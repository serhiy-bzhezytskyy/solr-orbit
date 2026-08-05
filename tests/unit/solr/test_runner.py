# SPDX-License-Identifier: Apache-2.0
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Solr runners (solrorbit/worker_coordinator/runner.py)"""

import asyncio
import json
import unittest
from unittest.mock import MagicMock

from solrorbit.worker_coordinator.runner import (
    _flatten_document,
    _translate_ndjson_stream,
    _translate_ndjson_batch,
    SolrBulkIndex,
    SolrSearch,
    SolrBinaryBulkIndex,
    RawRequest,
    SolrCreateAlias,
    SolrCreateCollection,
    SolrDeleteAlias,
    SolrDeleteCollection,
)
from solrorbit.conversion.field import normalize_field_name
from solrorbit.conversion.query import translate_opensearch_query


# Backward compatibility aliases for tests
def _normalize_field_name(field):
    """Test compatibility wrapper."""
    return normalize_field_name(field)


def _translate_query_node(node):
    """Test compatibility wrapper — returns only the q string."""
    return translate_opensearch_query({"query": node})["q"]


def _run(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestFieldNameNormalization(unittest.TestCase):
    """Test OpenSearch to Solr field name normalization with underscore convention."""

    def test_raw_suffix_to_underscore(self):
        """Test that .raw suffix is converted to underscore."""
        self.assertEqual("country_code_raw", _normalize_field_name("country_code.raw"))
        self.assertEqual("name_raw", _normalize_field_name("name.raw"))
        self.assertEqual("title_raw", _normalize_field_name("title.raw"))

    def test_keyword_suffix_to_underscore(self):
        """Test that .keyword suffix is converted to underscore."""
        self.assertEqual("country_code_keyword", _normalize_field_name("country_code.keyword"))
        self.assertEqual("name_keyword", _normalize_field_name("name.keyword"))

    def test_sort_suffix_to_underscore(self):
        """Test that .sort suffix is converted to underscore."""
        self.assertEqual("title_sort", _normalize_field_name("title.sort"))
        self.assertEqual("name_sort", _normalize_field_name("name.sort"))

    def test_regular_fields_unchanged(self):
        """Test that regular field names are unchanged."""
        self.assertEqual("country_code", _normalize_field_name("country_code"))
        self.assertEqual("title", _normalize_field_name("title"))
        self.assertEqual("_id", _normalize_field_name("_id"))

    def test_term_query_with_raw_field(self):
        """Test that term queries with .raw fields are normalized to _raw."""
        query = {"term": {"country_code.raw": "US"}}
        result = _translate_query_node(query)
        # Should use country_code_raw (underscore convention)
        self.assertEqual("country_code_raw:US", result)

    def test_term_query_with_keyword_field(self):
        """Test that term queries with .keyword fields are normalized to _keyword."""
        query = {"term": {"name.keyword": "John"}}
        result = _translate_query_node(query)
        self.assertEqual("name_keyword:John", result)

    def test_range_query_with_raw_field(self):
        """Test that range queries with .raw fields are normalized to _raw."""
        query = {"range": {"population.raw": {"gte": 1000, "lte": 5000}}}
        result = _translate_query_node(query)
        self.assertEqual("population_raw:[1000 TO 5000]", result)

    def test_exists_query_with_raw_field(self):
        """Test that exists queries with .raw fields are normalized to _raw."""
        query = {"exists": {"field": "country_code.raw"}}
        result = _translate_query_node(query)
        self.assertEqual("country_code_raw:[* TO *]", result)


class TestTranslateNdjsonBatch(unittest.TestCase):
    def test_id_injected_from_action_line(self):
        lines = [
            '{"index": {"_id": "doc-1", "_index": "my-index"}}',
            '{"title": "hello", "body": "world"}',
        ]
        docs = _translate_ndjson_batch(lines)
        self.assertEqual(1, len(docs))
        self.assertEqual("doc-1", docs[0]["id"])
        self.assertEqual("hello", docs[0]["title"])

    def test_id_absent_when_action_has_no_id(self):
        lines = [
            '{"index": {"_index": "my-index"}}',
            '{"title": "no id"}',
        ]
        docs = _translate_ndjson_batch(lines)
        self.assertEqual(1, len(docs))
        self.assertNotIn("id", docs[0])

    def test_type_dropped(self):
        lines = [
            '{"index": {"_id": "1", "_type": "_doc", "_index": "idx"}}',
            '{"field": "value"}',
        ]
        docs = _translate_ndjson_batch(lines)
        self.assertNotIn("_type", docs[0])

    def test_index_not_stored_in_doc(self):
        lines = [
            '{"index": {"_id": "1", "_index": "my-collection"}}',
            '{"x": 1}',
        ]
        docs = _translate_ndjson_batch(lines)
        self.assertNotIn("_index", docs[0])
        self.assertNotIn("my-collection", docs[0].values())

    def test_multiple_pairs(self):
        lines = [
            '{"index": {"_id": "a"}}',
            '{"f": 1}',
            '{"index": {"_id": "b"}}',
            '{"f": 2}',
        ]
        docs = _translate_ndjson_batch(lines)
        self.assertEqual(2, len(docs))
        self.assertEqual("a", docs[0]["id"])
        self.assertEqual("b", docs[1]["id"])

    def test_malformed_json_skipped(self):
        lines = [
            "not json",
            '{"f": 1}',
        ]
        docs = _translate_ndjson_batch(lines)
        self.assertEqual(0, len(docs))

    def test_empty_lines_ignored(self):
        lines = ["", '{"index": {"_id": "1"}}', '{"f": 1}', ""]
        docs = _translate_ndjson_batch(lines)
        self.assertEqual(1, len(docs))


class TestSolrBulkIndex(unittest.TestCase):
    def _params(self, corpus_lines):
        return {
            "host": "localhost",
            "port": 8983,
            "collection": "test",
            "corpus": corpus_lines,
            "bulk-size": 500,
        }

    def test_bulk_index_returns_weight(self):
        mock_sc = MagicMock()
        mock_sc.add.return_value = None

        lines = [
            '{"index": {"_id": "1"}}',
            '{"title": "doc"}',
        ]
        runner = SolrBulkIndex()
        result = _run(runner(mock_sc, self._params(lines)))

        self.assertEqual(1, result["weight"])
        self.assertEqual("docs", result["unit"])
        self.assertTrue(result["success"])

    def test_bulk_index_reports_errors(self):
        import pysolr
        mock_sc = MagicMock()
        mock_sc.add.side_effect = pysolr.SolrError("Indexing error")

        lines = [
            '{"index": {"_id": "1"}}',
            '{"title": "doc"}',
        ]
        runner = SolrBulkIndex()
        result = _run(runner(mock_sc, self._params(lines)))
        self.assertFalse(result["success"])
        self.assertGreater(result["error-count"], 0)

    def test_simple_ndjson_format(self):
        """Test simple NDJSON (one doc per line, no action lines)."""
        mock_sc = MagicMock()
        mock_sc.add.return_value = None

        # Simple NDJSON: just document lines, no action lines
        lines = [
            '{"vendor_id": "1", "trip_distance": 1.2}',
            '{"vendor_id": "2", "trip_distance": 3.5}',
            '{"vendor_id": "1", "trip_distance": 0.8}',
        ]
        runner = SolrBulkIndex()
        result = _run(runner(mock_sc, self._params(lines)))

        self.assertEqual(3, result["weight"])
        self.assertTrue(result["success"])
        # Verify add was called with 3 docs (collection + docs batch as positional args)
        self.assertEqual(1, mock_sc.add.call_count)
        added_docs = mock_sc.add.call_args[0][1]
        self.assertEqual(3, len(added_docs))


class TestFlattenDocument(unittest.TestCase):
    """
    Solr documents are flat. noaa nests an eight-field station object with a location object inside
    it, and writes its *RANGE fields as {gte,lte} pairs — none of which Solr accepts as it stands.
    """

    def test_a_nested_object_becomes_underscore_joined_fields(self):
        doc = {"date": "2016-01-01", "station": {"id": "AE1", "elevation": 34.0}}
        self.assertEqual(
            {"date": "2016-01-01", "station_id": "AE1", "station_elevation": 34.0},
            _flatten_document(doc))

    def test_the_names_match_what_query_normalisation_produces(self):
        # The query side turns station.location.lat into station_location_lat; the two have to agree
        # or the operations would search fields the documents never wrote.
        flat = _flatten_document({"station": {"location": {"lat": 25.333, "lon": 55.517}}})
        self.assertEqual("station_location_lat", normalize_field_name("station.location.lat"))
        self.assertIn("station_location_lat", flat)
        self.assertIn("station_location_lon", flat)

    def test_a_lat_lon_object_also_yields_a_spatial_value(self):
        # A Solr spatial field takes "lat,lon"; the components are kept too, because an RPT field
        # cannot expose them as a ValueSource.
        flat = _flatten_document({"station": {"location": {"lat": 25.333, "lon": 55.517}}})
        self.assertEqual("25.333,55.517", flat["station_location"])
        self.assertEqual(25.333, flat["station_location_lat"])
        self.assertEqual(55.517, flat["station_location_lon"])

    def test_an_object_that_is_not_a_point_gets_no_combined_value(self):
        flat = _flatten_document({"TRANGE": {"gte": 18.8, "lte": 29.3}})
        self.assertNotIn("TRANGE", flat)
        self.assertEqual({"TRANGE_gte": 18.8, "TRANGE_lte": 29.3}, flat)

    def test_a_dotted_key_normalises_like_a_queried_field_name(self):
        # big5 writes a literal "aws.cloudwatch" key holding an object. The query side turns
        # aws.cloudwatch.log_stream into aws_cloudwatch_log_stream, so the document has to arrive
        # under that name — otherwise the operations search a field nothing wrote.
        flat = _flatten_document({"aws.cloudwatch": {"log_stream": "madeye"}})
        self.assertIn("aws_cloudwatch_log_stream", flat)
        self.assertEqual("madeye", flat["aws_cloudwatch_log_stream"])
        self.assertEqual("aws_cloudwatch_log_stream", normalize_field_name("aws.cloudwatch.log_stream"))

    def test_an_at_prefixed_key_survives(self):
        # @timestamp holds no dot, so normalisation leaves it as it is — and the operations name it
        # exactly that way.
        self.assertEqual({"@timestamp": "x"}, _flatten_document({"@timestamp": "x"}))

    def test_a_flat_document_is_unchanged(self):
        doc = {"TAVG": 22.9, "id": "1"}
        self.assertEqual(doc, _flatten_document(doc))

    def test_a_list_of_objects_is_left_alone(self):
        # Solr's answer to that is a child document, a different shape than flattening.
        doc = {"readings": [{"v": 1}, {"v": 2}]}
        self.assertEqual(doc, _flatten_document(doc))


class TestNestedCorpusThroughTheTranslator(unittest.TestCase):
    def test_a_real_noaa_document_comes_out_flat_and_dated(self):
        lines = ['{"index": {"_id": "0"}}',
                 '{"date": "2016-01-01T00:00:00", "TAVG": 22.9, "station": {"id": "AE1", '
                 '"location": {"lat": 25.333, "lon": 55.517}}, "TRANGE": {"gte": 18.8, "lte": 29.3}}']
        doc, target = next(iter(_translate_ndjson_stream(lines)))
        self.assertEqual("AE1", doc["station_id"])
        self.assertEqual("25.333,55.517", doc["station_location"])
        self.assertEqual(18.8, doc["TRANGE_gte"])
        # A timestamp with no zone: OpenSearch reads it as UTC, Solr rejects it outright, so every
        # noaa document failed to index until the T-separated form was handled alongside the
        # space-separated one pmc writes.
        self.assertEqual("2016-01-01T00:00:00Z", doc["date"])

    def test_both_zoneless_timestamp_forms_are_given_a_zone(self):
        for written in ("2016-01-01 00:00:00", "2016-01-01T00:00:00"):
            lines = ['{"index": {"_id": "x"}}', json.dumps({"date": written})]
            doc, _ = next(iter(_translate_ndjson_stream(lines)))
            self.assertEqual("2016-01-01T00:00:00Z", doc["date"], msg="from %r" % written)


class _RecordingBulkClient:
    """
    Records which collection each batch went to.

    A real class rather than a Mock, so a call to a method the runner is not supposed to use fails
    instead of quietly succeeding.
    """

    def __init__(self):
        self.batches = []      # (collection, [ids])
        self.committed = []

    def add(self, collection, docs, **kwargs):
        self.batches.append((collection, [d.get("id") for d in docs]))

    def commit(self, collection, **kwargs):
        self.committed.append(collection)


class TestSolrBulkIndexTargets(unittest.TestCase):
    """
    A workload can feed several collections from one corpus — http_logs has a document set per
    month, each with its own target-collection — and the bulk action line is where that target
    arrives. Reading only _id from it sent all 247M documents to whichever collection the operation
    happened to name.
    """

    def _params(self, lines, **extra):
        return {"host": "localhost", "port": 8983, "corpus": lines, "bulk-size": 500, **extra}

    def test_documents_go_to_the_collection_their_action_line_names(self):
        client = _RecordingBulkClient()
        lines = [
            '{"index": {"_index": "logs-181998", "_id": "a"}}', '{"status": 200}',
            '{"index": {"_index": "logs-191998", "_id": "b"}}', '{"status": 404}',
            '{"index": {"_index": "logs-181998", "_id": "c"}}', '{"status": 200}',
        ]
        _run(SolrBulkIndex()(client, self._params(lines, collection="ignored")))
        by_collection = {c: ids for c, ids in client.batches}
        self.assertEqual({"logs-181998": ["a", "c"], "logs-191998": ["b"]}, by_collection)

    def test_operation_collection_is_the_fallback_when_the_action_names_none(self):
        client = _RecordingBulkClient()
        lines = ['{"index": {"_id": "a"}}', '{"status": 200}']
        _run(SolrBulkIndex()(client, self._params(lines, collection="fallback")))
        self.assertEqual([("fallback", ["a"])], client.batches)

    def test_plain_ndjson_uses_the_operation_collection(self):
        client = _RecordingBulkClient()
        _run(SolrBulkIndex()(client, self._params(['{"id": "a", "status": 200}'], collection="plain")))
        self.assertEqual(["plain"], [c for c, _ in client.batches])

    def test_no_collection_anywhere_is_an_error(self):
        from solrorbit import exceptions
        client = _RecordingBulkClient()
        lines = ['{"index": {"_id": "a"}}', '{"status": 200}']
        with self.assertRaises(exceptions.DataError):
            _run(SolrBulkIndex()(client, self._params(lines)))

    def test_commit_reaches_every_collection_written_to(self):
        client = _RecordingBulkClient()
        lines = [
            '{"index": {"_index": "logs-181998", "_id": "a"}}', '{"status": 200}',
            '{"index": {"_index": "logs-191998", "_id": "b"}}', '{"status": 404}',
        ]
        _run(SolrBulkIndex()(client, self._params(lines, collection="ignored", commit=True)))
        self.assertEqual({"logs-181998", "logs-191998"}, set(client.committed))

    def test_a_full_batch_flushes_per_collection(self):
        # bulk-size counts per target, so two collections each reaching the size flush separately
        # rather than one flush of the combined count.
        client = _RecordingBulkClient()
        lines = []
        for i in range(4):
            coll = "logs-181998" if i % 2 == 0 else "logs-191998"
            lines += ['{"index": {"_index": "%s", "_id": "%d"}}' % (coll, i), '{"status": 200}']
        _run(SolrBulkIndex()(client, self._params(lines, collection="ignored", **{"bulk-size": 2})))
        # Each collection gets exactly its own two documents, in one flush each.
        self.assertEqual([("logs-181998", ["0", "2"]), ("logs-191998", ["1", "3"])],
                         sorted(client.batches))


class _RecordingChainClient:
    """Records the raw requests a chained update makes, and any pysolr-style adds."""

    def __init__(self, status_code=200):
        self.requests = []     # (method, path, headers)
        self.adds = []
        self.committed = []
        self.status_code = status_code

    def raw_request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, headers or {}))
        resp = MagicMock()
        resp.status_code = self.status_code
        resp.text = "" if self.status_code < 400 else "boom"
        return resp

    def add(self, collection, docs, **kwargs):
        self.adds.append((collection, [d.get("id") for d in docs]))

    def commit(self, collection, **kwargs):
        self.committed.append(collection)


class TestBulkIndexUpdateChain(unittest.TestCase):
    """
    An operation may route documents through a named update chain — what an OpenSearch workload calls
    an ingest pipeline, and how http_logs' four pipeline operations are ported.

    The request cannot go through pysolr: it builds the path as handler + "/" + "?commit=…", so a
    handler carrying a query string arrives as "update?update.chain=x/?commit=true" and Solr answers
    'unknown UpdateRequestProcessorChain: x/'.
    """

    def _params(self, **extra):
        lines = ['{"index": {"_id": "a"}}', '{"status": 200}']
        return {"collection": "logs", "corpus": lines, "bulk-size": 10, **extra}

    def test_no_chain_goes_through_the_normal_add(self):
        client = _RecordingChainClient()
        _run(SolrBulkIndex()(client, self._params()))
        self.assertEqual([("logs", ["a"])], client.adds)
        self.assertEqual([], client.requests)

    def test_a_chain_is_sent_as_update_chain_on_the_path(self):
        client = _RecordingChainClient()
        _run(SolrBulkIndex()(client, self._params(pipeline="grok-pipeline")))
        self.assertEqual([], client.adds, msg="a chained update must not go through pysolr")
        self.assertEqual(1, len(client.requests))
        method, path, headers = client.requests[0]
        self.assertEqual("POST", method)
        self.assertIn("/solr/logs/update?", path)
        self.assertIn("update.chain=grok-pipeline", path)
        self.assertEqual("application/json", headers.get("Content-type"))

    def test_update_chain_is_accepted_as_a_spelling(self):
        client = _RecordingChainClient()
        _run(SolrBulkIndex()(client, self._params(**{"update-chain": "baseline-pipeline"})))
        self.assertIn("update.chain=baseline-pipeline", client.requests[0][1])

    def test_a_chained_batch_that_fails_is_counted_as_an_error(self):
        client = _RecordingChainClient(status_code=400)
        result = _run(SolrBulkIndex()(client, self._params(pipeline="grok-pipeline")))
        self.assertFalse(result["success"])
        self.assertEqual(1, result["error-count"])

    def test_the_binary_runner_also_carries_the_chain(self):
        client = _RecordingChainClient()
        _run(SolrBinaryBulkIndex()(client, self._params(pipeline="grok-pipeline")))
        self.assertIn("update.chain=grok-pipeline", client.requests[0][1])
        self.assertEqual("application/javabin", client.requests[0][2].get("Content-type"))


class _RecordingRawClient:
    """Records the raw request and answers with a payload the test chooses."""

    def __init__(self, payload=None, status_code=200):
        self.calls = []
        self.payload = payload if payload is not None else {}
        self.status_code = status_code

    def raw_request(self, method, path, body=None, headers=None):
        self.calls.append((method, path, body, headers or {}))
        resp = MagicMock()
        resp.status_code = self.status_code
        resp.json.return_value = self.payload
        resp.text = ""
        return resp


class TestRawRequestFormAndInPayloadErrors(unittest.TestCase):
    """
    Some Solr handlers read their input from request parameters rather than a JSON body, and some
    report failure inside a 200. big5's 46 PPL operations become SQL, and Solr's /sql handler does
    both: it answers "stmt parameter cannot be null" to a JSON body, and puts an EXCEPTION entry in
    its result-set when a statement fails.
    """

    def _params(self, **extra):
        return {"method": "POST", "path": "/solr/c/sql", "body": {"stmt": "SELECT id FROM c"}, **extra}

    def test_form_sends_an_encoded_body_with_the_matching_content_type(self):
        client = _RecordingRawClient()
        _run(RawRequest()(client, self._params(form=True)))
        method, path, body, headers = client.calls[0]
        self.assertEqual("stmt=SELECT+id+FROM+c", body)
        self.assertEqual("application/x-www-form-urlencoded", headers.get("Content-type"))

    def test_without_form_the_body_is_left_as_a_dict_for_json(self):
        client = _RecordingRawClient()
        _run(RawRequest()(client, self._params()))
        self.assertEqual({"stmt": "SELECT id FROM c"}, client.calls[0][2])

    def test_an_exception_inside_a_200_is_reported_as_a_failure(self):
        # A status check alone would call this a success on a query that did not run.
        client = _RecordingRawClient(payload={"result-set": {"docs": [
            {"EXCEPTION": "stmt parameter cannot be null", "EOF": True}]}})
        result = _run(RawRequest()(client, self._params(form=True)))
        self.assertEqual(200, result["http-status"])
        self.assertFalse(result["success"])
        self.assertEqual(1, result["error-count"])

    def test_a_clean_result_set_is_a_success(self):
        client = _RecordingRawClient(payload={"result-set": {"docs": [
            {"id": "1"}, {"EOF": True, "RESPONSE_TIME": 4}]}})
        result = _run(RawRequest()(client, self._params(form=True)))
        self.assertTrue(result["success"])
        self.assertEqual(0, result["error-count"])

    def test_a_4xx_is_a_failure_even_with_no_payload(self):
        client = _RecordingRawClient(status_code=400)
        result = _run(RawRequest()(client, self._params()))
        self.assertFalse(result["success"])
        self.assertEqual(1, result["error-count"])

    def test_a_response_that_is_not_json_does_not_raise(self):
        client = _RecordingRawClient()
        client.raw_request = lambda *a, **k: type("R", (), {
            "status_code": 200, "text": "not json",
            "json": lambda self: (_ for _ in ()).throw(ValueError("no json"))})()
        result = _run(RawRequest()(client, self._params()))
        self.assertTrue(result["success"])


class TestSolrSearch(unittest.TestCase):
    def _base_params(self):
        return {
            "host": "localhost",
            "port": 8983,
            "collection": "test",
        }

    def test_classic_mode(self):
        mock_results = MagicMock()
        mock_results.hits = 42
        mock_sc = MagicMock()
        mock_sc.search.return_value = mock_results

        params = {**self._base_params(), "q": "hello world", "rows": 10}
        runner = SolrSearch()
        result = _run(runner(mock_sc, params))

        self.assertEqual(42, result["hits"])
        self.assertEqual(1, result["weight"])

    def test_json_dsl_mode(self):
        """Mode 2: body with a Solr-style string query → POST to /query endpoint via raw_request."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": {"numFound": 7}}
        mock_sc = MagicMock()
        mock_sc.raw_request.return_value = mock_resp

        # Solr JSON DSL uses a string for the 'query' key, not a dict
        params = {**self._base_params(), "body": {"query": "*:*", "limit": 5}}
        runner = SolrSearch()
        result = _run(runner(mock_sc, params))

        self.assertEqual(7, result["hits"])
        mock_sc.raw_request.assert_called_once()

    def test_dict_query_body_posted_to_solr(self):
        """Body with dict query (Solr JSON DSL) is POSTed to /query endpoint via raw_request."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": {"numFound": 3}}
        mock_sc = MagicMock()
        mock_sc.raw_request.return_value = mock_resp

        params = {**self._base_params(), "body": {"query": {"match_all": {}}, "size": 20}}
        runner = SolrSearch()
        result = _run(runner(mock_sc, params))

        self.assertEqual(3, result["hits"])
        mock_sc.raw_request.assert_called_once()


class TestSolrCreateCollection(unittest.TestCase):
    def test_two_step_sequence(self):
        import tempfile
        mock_sc = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            params = {
                "collection": "my-coll",
                "configset": "my-config",
                "configset-path": tmpdir,
            }
            runner = SolrCreateCollection()
            _run(runner(mock_sc, params))

        # Verify upload_configset called before create_collection
        mock_sc.upload_configset.assert_called_once_with("my-config", tmpdir)
        mock_sc.create_collection.assert_called_once()

    def test_create_collection_passes_tlog_pull_replicas(self):
        """Runner should pass tlog-replicas and pull-replicas to create_collection."""
        mock_sc = MagicMock()

        params = {
            "collection": "my-coll",
            "configset": "my-config",
            "num-shards": 2,
            "replication-factor": 1,
            "tlog-replicas": 2,
            "pull-replicas": 1,
        }
        runner = SolrCreateCollection()
        _run(runner(mock_sc, params))

        mock_sc.create_collection.assert_called_once_with(
            "my-coll", "my-config", 2, 1, 2, 1
        )

    def test_create_collection_defaults_tlog_pull_to_zero(self):
        """Runner defaults tlog-replicas and pull-replicas to 0 when omitted."""
        mock_sc = MagicMock()

        params = {
            "collection": "my-coll",
            "configset": "my-config",
        }
        runner = SolrCreateCollection()
        _run(runner(mock_sc, params))

        mock_sc.create_collection.assert_called_once_with(
            "my-coll", "my-config", 1, 1, 0, 0
        )


class TestSolrDeleteCollection(unittest.TestCase):
    def test_delete_ignores_missing_by_default(self):
        from solrorbit.client import CollectionNotFoundError
        mock_sc = MagicMock()
        mock_sc.delete_collection.side_effect = CollectionNotFoundError("not found")

        params = {
            "collection": "missing-coll",
            "ignore-missing": True,
        }
        runner = SolrDeleteCollection()
        # Should not raise
        _run(runner(mock_sc, params))


class _RecordingAliasClient:
    """
    A stand-in for the admin client that records alias calls.

    Deliberately a real class rather than a Mock: a bare Mock answers to any attribute name, so a
    runner calling a method that does not exist would still pass. Only the methods the alias runners
    are supposed to use are defined here, and the recorded aliases are readable state.
    """

    def __init__(self):
        self.aliases = {}
        self.deleted = []

    def create_alias(self, name, collections):
        if not isinstance(collections, str):
            collections = ",".join(collections)
        self.aliases[name] = collections

    def delete_alias(self, name, ignore_missing=True):
        self.deleted.append((name, ignore_missing))
        self.aliases.pop(name, None)


class TestSolrAlias(unittest.TestCase):
    """
    A workload written against an index pattern such as logs-* needs an alias: Solr has no wildcard
    collection name and asking for one is a 404, while a standard alias over the matching collections
    searches all of their shards as one whole.
    """

    def test_create_alias_accepts_a_list(self):
        client = _RecordingAliasClient()
        _run(SolrCreateAlias()(client, {"alias": "logs", "collections": ["logs-1", "logs-2"]}))
        self.assertEqual({"logs": "logs-1,logs-2"}, client.aliases)

    def test_create_alias_accepts_a_comma_separated_string(self):
        client = _RecordingAliasClient()
        _run(SolrCreateAlias()(client, {"alias": "logs", "collections": "logs-1,logs-2"}))
        self.assertEqual({"logs": "logs-1,logs-2"}, client.aliases)

    def test_create_alias_without_collections_is_an_error(self):
        from solrorbit import exceptions
        client = _RecordingAliasClient()
        with self.assertRaises(exceptions.DataError):
            _run(SolrCreateAlias()(client, {"alias": "logs"}))

    def test_alias_name_may_come_from_the_collection_param(self):
        # The converter writes the target under "collection" for every other operation, so accepting
        # it here keeps a converted workload from having to special-case the alias.
        client = _RecordingAliasClient()
        _run(SolrCreateAlias()(client, {"collection": "logs", "collections": ["logs-1"]}))
        self.assertIn("logs", client.aliases)

    def test_delete_alias_passes_ignore_missing_through(self):
        client = _RecordingAliasClient()
        _run(SolrCreateAlias()(client, {"alias": "logs", "collections": ["logs-1"]}))
        _run(SolrDeleteAlias()(client, {"alias": "logs", "ignore-missing": False}))
        self.assertEqual([("logs", False)], client.deleted)
        self.assertEqual({}, client.aliases)

    def test_delete_alias_ignores_missing_by_default(self):
        client = _RecordingAliasClient()
        _run(SolrDeleteAlias()(client, {"alias": "never-existed"}))
        self.assertEqual([("never-existed", True)], client.deleted)

    def test_both_runners_are_registered(self):
        from solrorbit.worker_coordinator.runner import register_default_runners, runner_for
        register_default_runners()
        self.assertIsNotNone(runner_for("create-alias"))
        self.assertIsNotNone(runner_for("delete-alias"))


class TestRunnerRegistrationSmoke(unittest.TestCase):
    """
    Verify that Solr runners work end-to-end through the MultiClientRunner wrapper.

    The wrapper (created by register_runner) does client_extractor=lambda c: c["default"]
    before calling the runner's __call__.  Tests that bypass register_runner and call
    __call__ directly would miss this extraction and test the wrong calling convention.
    These smoke-tests go through runner_for() so the full wrapper stack is exercised.
    """

    def setUp(self):
        from solrorbit.worker_coordinator.runner import register_default_runners
        register_default_runners()

    def _run_via_framework(self, op_type, clients_dict, params):
        """Look up a registered runner and invoke it the same way execute_single does."""
        from solrorbit.worker_coordinator.runner import runner_for
        wrapped = runner_for(op_type)
        return _run(wrapped(clients_dict, params))

    def test_delete_collection_via_framework(self):
        """SolrDeleteCollection receives SolrClient directly after MultiClientRunner extraction."""
        mock_sc = MagicMock()
        params = {"collection": "smoke-coll", "ignore-missing": True}
        # Pass the dict — the wrapper extracts ["default"] before calling __call__
        self._run_via_framework("delete-collection", {"default": mock_sc}, params)
        mock_sc.delete_collection.assert_called_once_with("smoke-coll")

    def test_bulk_index_via_framework(self):
        """SolrBulkIndex receives SolrClient directly after MultiClientRunner extraction."""
        mock_sc = MagicMock()
        mock_sc.add.return_value = None
        lines = ['{"index": {"_id": "1"}}', '{"title": "doc"}']
        params = {"collection": "smoke-coll", "corpus": lines, "bulk-size": 500}
        result = self._run_via_framework("bulk-index", {"default": mock_sc}, params)
        self.assertTrue(result["success"])
        mock_sc.add.assert_called_once()

    def test_search_via_framework(self):
        """SolrSearch receives SolrClient directly after MultiClientRunner extraction."""
        mock_results = MagicMock()
        mock_results.hits = 5
        mock_sc = MagicMock()
        mock_sc.search.return_value = mock_results
        params = {"collection": "smoke-coll", "q": "*:*"}
        result = self._run_via_framework("search", {"default": mock_sc}, params)
        self.assertEqual(5, result["hits"])


class TestIntegerCoercion(unittest.TestCase):
    """
    A corpus may write a fraction where the mapping declares an integer.

    OpenSearch coerces it by truncating toward zero — measured: '800.94' is indexed as 800, '-1.5' as
    -1, '2.5' as 2, '3.5' as 3, so it truncates rather than rounds. Solr rejects the document instead
    ("For input string: 800.94"), and no shipped processor closes the gap: ParseIntFieldUpdateProcessor
    *skips* a value it cannot parse, leaving the fraction to reach the field. clickbench's FlashMinor2 is
    declared short and written with fractions, so the corpus could not be indexed at all.
    """

    def test_a_fractional_string_is_truncated_toward_zero(self):
        from solrorbit.worker_coordinator.runner import _coerce_integer_fields
        doc = _coerce_integer_fields(
            {"a": "800.94", "b": "-1.5", "c": "2.5", "d": "3.5"}, {"a", "b", "c", "d"})
        self.assertEqual({"a": 800, "b": -1, "c": 2, "d": 3}, doc)

    def test_a_float_is_truncated_too(self):
        from solrorbit.worker_coordinator.runner import _coerce_integer_fields
        self.assertEqual({"a": 800}, _coerce_integer_fields({"a": 800.94}, {"a"}))

    def test_a_field_the_schema_does_not_call_integral_keeps_its_fraction(self):
        # Coercing by value rather than by declaration would silently truncate a real float field.
        from solrorbit.worker_coordinator.runner import _coerce_integer_fields
        self.assertEqual({"price": "1.5"}, _coerce_integer_fields({"price": "1.5"}, {"other"}))

    def test_a_whole_value_and_a_string_are_left_alone(self):
        from solrorbit.worker_coordinator.runner import _coerce_integer_fields
        doc = _coerce_integer_fields({"n": 7, "s": "text", "v": "1.2.3"}, {"n", "s", "v"})
        self.assertEqual({"n": 7, "s": "text", "v": "1.2.3"}, doc)

    def test_the_schema_is_not_read_when_no_value_carries_a_fraction(self):
        # Every workload before clickbench has no such value; a schema request per batch would put it
        # in the measured path for all of them.
        from solrorbit.worker_coordinator.runner import _coerce_integer_fields
        calls = []

        def resolve():
            calls.append(1)
            return {"a"}

        _coerce_integer_fields({"a": 7, "b": "text"}, resolve)
        self.assertEqual([], calls)
        _coerce_integer_fields({"a": "7.5"}, resolve)
        self.assertEqual([1], calls)

    def test_plain_ndjson_is_prepared_the_same_way_as_bulk_pairs(self):
        # The plain branch used to flatten nothing, repair no timestamp and coerce nothing, so a corpus
        # published without action lines was prepared differently from the same documents with them.
        from solrorbit.worker_coordinator.runner import _translate_ndjson_stream
        lines = ['{"outer": {"inner": 1}, "when": "2013-07-15 09:21:41", "n": "2.5"}']
        docs = [doc for doc, _ in _translate_ndjson_stream(lines, lambda: {"n"})]
        self.assertEqual(1, docs[0]["outer_inner"])
        self.assertEqual("2013-07-15T09:21:41Z", docs[0]["when"])
        self.assertEqual(2, docs[0]["n"])


class TestSolrBinarySearch(unittest.TestCase):
    """
    A search whose response comes back in Solr's binary format.

    Mocks here return a real encoded response rather than a bare Mock: a Mock says yes to any
    attribute, so a runner reading the wrong one would still look like it worked.
    """

    def _client_returning(self, payload):
        from solrorbit.utils.javabin import JavaBinWriter, VERSION
        writer = JavaBinWriter()
        writer._byte(VERSION)
        writer._value(payload)
        encoded = writer._out.getvalue()

        response = MagicMock()
        response.content = encoded
        response.raise_for_status.return_value = None
        client = MagicMock()
        client.raw_request.return_value = response
        return client, response

    def test_the_hit_count_is_read_from_the_decoded_response(self):
        from solrorbit.worker_coordinator.runner import SolrBinarySearch
        client, _ = self._client_returning(
            {"response": {"numFound": 42, "start": 0, "docs": []}})
        result = _run(SolrBinarySearch()(client, {"collection": "c", "body": {"query": "*:*"}}))
        self.assertEqual(42, result["hits"])
        self.assertTrue(result["success"])

    def test_the_binary_response_writer_is_requested(self):
        # Without wt=javabin Solr answers JSON and the measurement is not the binary transport at all.
        from solrorbit.worker_coordinator.runner import SolrBinarySearch
        client, _ = self._client_returning({"response": {"numFound": 1, "start": 0, "docs": []}})
        _run(SolrBinarySearch()(client, {"collection": "c", "body": {"query": "*:*"}}))
        path = client.raw_request.call_args[0][1]
        self.assertIn("wt=javabin", path)

    def test_a_query_without_a_body_goes_through_select_with_its_params(self):
        from solrorbit.worker_coordinator.runner import SolrBinarySearch
        client, _ = self._client_returning({"response": {"numFound": 7, "start": 0, "docs": []}})
        result = _run(SolrBinarySearch()(
            client, {"collection": "c", "q": "process_name:kernel", "rows": 3}))
        path = client.raw_request.call_args[0][1]
        self.assertIn("/solr/c/select", path)
        self.assertIn("rows=3", path)
        self.assertEqual(7, result["hits"])

    def test_proto_search_resolves_to_the_binary_search_runner(self):
        # Unregistered, the operation failed to load with no runner at all, so three big5 operations
        # could never run.
        from solrorbit.worker_coordinator.runner import register_default_runners, runner_for
        register_default_runners()
        self.assertIsNotNone(runner_for("proto-search"))
        self.assertIsNotNone(runner_for("binary-search"))


if __name__ == "__main__":
    unittest.main()
