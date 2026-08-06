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

"""Unit tests for solrorbit/conversion/workload_converter.py"""

import json
import os
import tempfile
import unittest

from solrorbit.conversion import workload_converter as wc
from solrorbit.conversion.workload_converter import (
    CONVERTED_MARKER,
    _jinja_restore,
    _jinja_substitute,
    _parse_jinja_fragment,
    convert_opensearch_workload,
    detect_workload_format_from_file,
    is_already_converted,
)
from solrorbit.conversion.query import (
    translate_to_solr_json_dsl,
    _convert_aggregations_to_facets,
    _auto_interval_to_solr_gap,
    _calendar_interval_to_solr_gap,
)


class TestDetectWorkloadFormatFromFile(unittest.TestCase):
    def _make_workload(self, tmpdir, workload_dict):
        path = os.path.join(tmpdir, "workload.json")
        with open(path, "w") as f:
            json.dump(workload_dict, f)
        return tmpdir

    def test_detects_opensearch_format_by_indices_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_workload(tmpdir, {"indices": [{"name": "my-index"}], "challenges": []})
            self.assertTrue(detect_workload_format_from_file(tmpdir))

    def test_detects_solr_format_by_collections_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_workload(tmpdir, {"collections": [{"name": "my-coll"}], "challenges": []})
            self.assertFalse(detect_workload_format_from_file(tmpdir))

    def test_raises_if_no_workload_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                detect_workload_format_from_file(tmpdir)


class TestIsAlreadyConverted(unittest.TestCase):
    def test_returns_false_when_no_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(is_already_converted(tmpdir))

    def test_returns_true_when_marker_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = os.path.join(tmpdir, CONVERTED_MARKER)
            with open(marker, "w") as f:
                f.write("# converted")
            self.assertTrue(is_already_converted(tmpdir))


class TestConvertOpensearchWorkload(unittest.TestCase):
    """Integration tests for the main conversion function."""

    def _make_source_workload(self, tmpdir, workload_dict):
        path = os.path.join(tmpdir, "workload.json")
        with open(path, "w") as f:
            json.dump(workload_dict, f)

    def test_renames_indices_to_collections(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {
                "indices": [{"name": "my-index"}],
                "challenges": [],
            })
            result = convert_opensearch_workload(src, dst)
            self.assertEqual(0, len(result["issues"]))

            with open(os.path.join(dst, "workload.json")) as f:
                out = json.load(f)
            self.assertIn("collections", out)
            self.assertNotIn("indices", out)
            self.assertEqual("my-index", out["collections"][0]["name"])

    def test_renames_operation_types(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {
                "indices": [],
                "challenges": [
                    {
                        "name": "default",
                        "schedule": [
                            {
                                "operation": {
                                    "name": "index-docs",
                                    "operation-type": "bulk",
                                },
                            },
                            {
                                "operation": {
                                    "name": "run-search",
                                    "operation-type": "search",
                                },
                            },
                        ],
                    }
                ],
            })
            convert_opensearch_workload(src, dst)
            with open(os.path.join(dst, "workload.json")) as f:
                out = json.load(f)
            schedule = out["challenges"][0]["schedule"]
            self.assertEqual("bulk-index", schedule[0]["operation"]["operation-type"])
            self.assertEqual("search", schedule[1]["operation"]["operation-type"])

    def test_translates_search_body_to_solr_json_dsl(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {
                "indices": [],
                "challenges": [
                    {
                        "name": "default",
                        "schedule": [
                            {
                                "operation": {
                                    "name": "search-all",
                                    "operation-type": "search",
                                    "body": {"query": {"match_all": {}}, "size": 10},
                                }
                            }
                        ],
                    }
                ],
            })
            convert_opensearch_workload(src, dst)
            with open(os.path.join(dst, "workload.json")) as f:
                out = json.load(f)
            body = out["challenges"][0]["schedule"][0]["operation"]["body"]
            # Body should be Solr JSON DSL (query is a string, not a dict)
            self.assertIsInstance(body["query"], str)
            self.assertEqual("*:*", body["query"])
            self.assertEqual(10, body["limit"])

    def test_unsupported_ops_are_skipped(self):
        # create-snapshot used to be listed here; Solr has a runner for it now, so an operation with
        # no Solr equivalent at all is needed to exercise the skip.
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {
                "indices": [],
                "challenges": [
                    {
                        "name": "default",
                        "schedule": [
                            {
                                "operation": {
                                    "name": "settings",
                                    "operation-type": "put-settings",
                                }
                            }
                        ],
                    }
                ],
            })
            result = convert_opensearch_workload(src, dst)
            self.assertIn("settings", result["skipped"])

    def test_backup_operations_are_converted_rather_than_skipped(self):
        # They have runners, so skipping them would drop a snapshot workload's whole point.
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {
                "indices": [],
                "challenges": [
                    {
                        "name": "default",
                        "schedule": [
                            {
                                "operation": {
                                    "name": "snap",
                                    "operation-type": "create-snapshot",
                                }
                            }
                        ],
                    }
                ],
            })
            result = convert_opensearch_workload(src, dst)
            self.assertNotIn("snap", result["skipped"])

    def test_nested_fragment_directory_is_converted(self):
        # A fragment can be collected by a nested path, as http_logs does with
        # test_procedures/intra_segment/. Listing only the top level dropped it from the output, the
        # collect call then rendered to nothing, and the workload failed to load on a dangling comma.
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {"indices": [], "challenges": []})
            nested = os.path.join(src, "test_procedures", "intra_segment")
            os.makedirs(nested)
            with open(os.path.join(nested, "schedule.json"), "w") as f:
                f.write('{"name": "nested", "operation": "match-all"}')
            convert_opensearch_workload(src, dst)
            self.assertTrue(
                os.path.isfile(os.path.join(dst, "test_procedures", "intra_segment", "schedule.json")),
                msg="nested fragment was not carried over",
            )

    def test_target_index_in_corpora_is_renamed(self):
        # The loader defaults target-collection only for a single-collection workload. http_logs has
        # 21, so each document spec names its own and the key has to be renamed or validation fails.
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {
                "indices": [{"name": "logs-1"}, {"name": "logs-2"}],
                "corpora": [{
                    "name": "http_logs",
                    "documents": [
                        {"target-index": "logs-1", "source-file": "a.json.bz2", "document-count": 1},
                        {"target-index": "logs-2", "source-file": "b.json.bz2", "document-count": 2},
                    ],
                }],
                "challenges": [],
            })
            convert_opensearch_workload(src, dst)
            with open(os.path.join(dst, "workload.json")) as f:
                out = f.read()
            self.assertNotIn("target-index", out)
            self.assertEqual(2, out.count('"target-collection"'))

    def test_a_templated_collection_body_becomes_a_configset_path(self):
        # http_logs declares all eight collections as "body": "{{ index_body }}". The literal-value
        # rewrite cannot see that — it matches the rendered file name — so every collection came out
        # with no configset and the run died on the first create-collection with "Can not find the
        # specified config set".
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            path = os.path.join(src, "workload.json")
            with open(path, "w") as f:
                f.write('{\n  "indices": [\n'
                        '    {"name": "logs-1", "body": "{{ index_body }}"},\n'
                        '    {"name": "logs-2", "body": "{{ index_body }}"}\n'
                        '  ],\n  "challenges": []\n}')
            convert_opensearch_workload(src, dst)
            with open(os.path.join(dst, "workload.json")) as f:
                out = f.read()
            self.assertNotIn('"body"', out)
            self.assertIn('"configset-path": "configsets/logs-1"', out)
            self.assertIn('"configset-path": "configsets/logs-2"', out)

    def test_a_string_body_outside_the_collections_list_is_left_alone(self):
        # Only the collections list is rewritten by position; anything else keeping a string-valued
        # "body" is not a configset reference.
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            path = os.path.join(src, "workload.json")
            with open(path, "w") as f:
                f.write('{\n  "indices": [{"name": "logs-1", "body": "{{ index_body }}"}],\n'
                        '  "corpora": [{"name": "elsewhere", "body": "keep-me"}],\n'
                        '  "challenges": []\n}')
            convert_opensearch_workload(src, dst)
            with open(os.path.join(dst, "workload.json")) as f:
                out = f.read()
            self.assertIn('"body": "keep-me"', out)
            self.assertIn('"configset-path": "configsets/logs-1"', out)

    def test_a_separator_the_converter_inserted_does_not_count_as_the_source_comma(self):
        """
        force_merge.json writes `"request-timeout": {{ … }}{%- if … %}, "key": value {%- endif %}`.

        The expression before the block becomes a placeholder with no comma after it, so the separator
        pass inserts one — and that comma is the converter's own. Reading it as the source's dropped
        the real separator and left two values side by side, which broke every workload that
        force-merges.

        ⚠️ Driven through the whole conversion, not the fragment helpers: a shared fragment reaches the
        output by a different path, and the isolated round trip stays valid either way. That is why
        the first version of this test passed with the fix reverted.
        """
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {
                "indices": [{"name": "c"}],
                "challenges": [{"name": "default", "schedule": [
                    {"operation": "x"},
                ]}],
            })
            os.makedirs(os.path.join(src, "test_procedures"))
            with open(os.path.join(src, "test_procedures", "default.json"), "w") as f:
                # The path is relative to the fragment's own directory, so from test_procedures/ a
                # shared fragment beside the workload is two levels up.
                f.write('{\n  "name": "default",\n  "schedule": [\n'
                        '    {{ benchmark.collect(parts="../../common_operations/force_merge.json") }}\n'
                        '  ]\n}')
            os.makedirs(os.path.join(os.path.dirname(src), "common_operations"), exist_ok=True)
            shared = os.path.join(os.path.dirname(src), "common_operations", "force_merge.json")
            with open(shared, "w") as f:
                f.write('{\n    "operation": {\n'
                        '        "operation-type": "force-merge",\n'
                        '        "request-timeout": {{ request_timeout | default(60) | tojson }}'
                        '{%- if max_num_segments is defined %},\n'
                        '        "max-num-segments": {{ max_num_segments | tojson }}\n'
                        '        {%- endif %}\n    }\n}')
            try:
                convert_opensearch_workload(src, dst)
                out_path = os.path.join(dst, "common_operations", "force_merge.json")
                self.assertTrue(os.path.isfile(out_path), "the shared fragment was not carried over")
                written = open(out_path).read()
                # The separator before the conditional has to survive: without it the rendered
                # fragment holds two values with nothing between them.
                jinja2 = __import__("jinja2")
                env = jinja2.Environment()
                for context in ({}, {"max_num_segments": 1}):
                    try:
                        json.loads("[" + env.from_string(written).render(**context) + "]")
                    except json.JSONDecodeError as error:
                        self.fail("the converted fragment does not render with %s: %s"
                                  % (context, error))
            finally:
                os.remove(shared)

    def test_writes_converted_marker(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {"indices": [], "challenges": []})
            convert_opensearch_workload(src, dst)
            self.assertTrue(os.path.isfile(os.path.join(dst, CONVERTED_MARKER)))

    def test_idempotent_after_marker(self):
        """is_already_converted returns True after a successful conversion."""
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {"indices": [], "challenges": []})
            convert_opensearch_workload(src, dst)
            self.assertTrue(is_already_converted(dst))

    def test_returns_output_dir_in_result(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            self._make_source_workload(src, {"indices": [], "challenges": []})
            result = convert_opensearch_workload(src, dst)
            self.assertEqual(os.path.abspath(dst), result["output_dir"])


class TestTranslateToSolrJsonDsl(unittest.TestCase):
    """Tests for translate_to_solr_json_dsl() in query.py."""

    def test_match_all_query(self):
        body = {"query": {"match_all": {}}, "size": 20}
        result = translate_to_solr_json_dsl(body)
        self.assertEqual("*:*", result["query"])
        self.assertEqual(20, result["limit"])
        self.assertNotIn("filter", result)

    def test_term_query(self):
        body = {"query": {"term": {"vendor_id": "CMT"}}}
        result = translate_to_solr_json_dsl(body)
        self.assertIn("vendor_id", result["query"])

    def test_range_query(self):
        body = {"query": {"range": {"fare_amount": {"gte": 5, "lte": 100}}}}
        result = translate_to_solr_json_dsl(body)
        self.assertIn("fare_amount", result["query"])
        self.assertIn("TO", result["query"])

    def test_bool_with_filter_goes_to_fq(self):
        body = {
            "query": {
                "bool": {
                    "must": [{"match_all": {}}],
                    "filter": [{"term": {"payment_type": "CRD"}}],
                }
            }
        }
        result = translate_to_solr_json_dsl(body)
        self.assertIn("filter", result)
        self.assertTrue(len(result["filter"]) > 0)

    def test_sort_is_extracted(self):
        body = {"query": {"match_all": {}}, "sort": [{"fare_amount": "desc"}]}
        result = translate_to_solr_json_dsl(body)
        self.assertIn("sort", result)
        self.assertIn("desc", result["sort"])

    def test_terms_aggregation_converted_to_facet(self):
        body = {
            "query": {"match_all": {}},
            "aggs": {
                "vendors": {
                    "terms": {"field": "vendor_id", "size": 5}
                }
            }
        }
        result = translate_to_solr_json_dsl(body)
        self.assertIn("facet", result)
        facet = result["facet"]["vendors"]
        self.assertEqual("terms", facet["type"])
        self.assertEqual("vendor_id", facet["field"])
        self.assertEqual(5, facet["limit"])

    def test_date_histogram_converted_to_range_facet(self):
        body = {
            "query": {"match_all": {}},
            "aggs": {
                "pickup_by_month": {
                    "date_histogram": {
                        "field": "pickup_datetime",
                        "calendar_interval": "month",
                    }
                }
            }
        }
        result = translate_to_solr_json_dsl(body)
        facet = result["facet"]["pickup_by_month"]
        self.assertEqual("range", facet["type"])
        self.assertEqual("pickup_datetime", facet["field"])
        self.assertEqual("+1MONTH", facet["gap"])

    def test_avg_metric_aggregation(self):
        body = {
            "query": {"match_all": {}},
            "aggs": {"avg_fare": {"avg": {"field": "fare_amount"}}}
        }
        result = translate_to_solr_json_dsl(body)
        self.assertEqual("avg(fare_amount)", result["facet"]["avg_fare"])

    def test_empty_body_returns_star_star(self):
        self.assertEqual({"query": "*:*"}, translate_to_solr_json_dsl({}))
        self.assertEqual({"query": "*:*"}, translate_to_solr_json_dsl(None))

    def test_non_dict_query_value_ignored(self):
        """If body['query'] is already a string (Solr native), return body unchanged."""
        body = {"query": "vendor_id:CMT", "limit": 5}
        result = translate_to_solr_json_dsl(body)
        # query is not a dict, so we just get q=*:* and limit from size (not present here)
        self.assertEqual("*:*", result["query"])


class TestConvertAggregationsToFacets(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual({}, _convert_aggregations_to_facets({}))
        self.assertEqual({}, _convert_aggregations_to_facets(None))

    def test_nested_agg_within_terms(self):
        aggs = {
            "by_vendor": {
                "terms": {"field": "vendor_id", "size": 10},
                "aggs": {
                    "avg_fare": {"avg": {"field": "fare_amount"}}
                }
            }
        }
        result = _convert_aggregations_to_facets(aggs)
        self.assertIn("by_vendor", result)
        self.assertIn("facet", result["by_vendor"])
        self.assertEqual("avg(fare_amount)", result["by_vendor"]["facet"]["avg_fare"])

    def test_histogram_aggregation(self):
        aggs = {"fare_hist": {"histogram": {"field": "fare_amount", "interval": 5}}}
        result = _convert_aggregations_to_facets(aggs)
        self.assertEqual("range", result["fare_hist"]["type"])
        self.assertEqual(5, result["fare_hist"]["gap"])

    def test_unsupported_agg_skipped_with_warning(self):
        aggs = {"my_geohash": {"geohash_grid": {"field": "location", "precision": 3}}}
        with self.assertLogs("solrorbit.conversion.query", level="WARNING") as log:
            result = _convert_aggregations_to_facets(aggs)
        self.assertEqual({}, result)
        self.assertTrue(any("geohash_grid" in msg for msg in log.output))

    def test_value_count_metric(self):
        aggs = {"doc_count": {"value_count": {"field": "vendor_id"}}}
        result = _convert_aggregations_to_facets(aggs)
        self.assertEqual("countvals(vendor_id)", result["doc_count"])


class TestRangeAggregationWithExplicitRanges(unittest.TestCase):
    """
    An explicit list of buckets, each with its own width. Solr's range facet takes the same thing
    under the same key — this was reported as an unsupported aggregation and dropped, which silently
    emptied noaa's three range-auto-date-histo operations: the facet vanished and the operation was
    left as a bare match-all that still looked like it ran.
    """

    def test_ranges_are_carried_across(self):
        aggs = {"tmax": {"range": {"field": "TMAX", "ranges": [
            {"to": -10}, {"from": -10, "to": 0}, {"from": 30}]}}}
        result = _convert_aggregations_to_facets(aggs)["tmax"]
        self.assertEqual("range", result["type"])
        self.assertEqual("TMAX", result["field"])
        self.assertEqual([{"to": -10}, {"from": -10, "to": 0}, {"from": 30}], result["ranges"])
        # gap/start/end belong to the fixed-width form and must not be invented here.
        for key in ("gap", "start", "end"):
            self.assertNotIn(key, result)

    def test_a_nested_aggregation_survives(self):
        aggs = {"tmax": {"range": {"field": "TMAX", "ranges": [{"from": 0, "to": 10}]},
                         "aggs": {"tmin": {"min": {"field": "TMIN"}}}}}
        result = _convert_aggregations_to_facets(aggs)["tmax"]
        self.assertEqual({"tmin": "min(TMIN)"}, result["facet"])

    def test_a_range_agg_listing_nothing_is_skipped(self):
        aggs = {"tmax": {"range": {"field": "TMAX", "ranges": []}}}
        self.assertEqual({}, _convert_aggregations_to_facets(aggs))


class TestAutoDateHistogram(unittest.TestCase):
    """
    auto_date_histogram states a bucket *target* and lets the engine pick an interval. Solr takes the
    interval, so it is computed from the same two inputs. It used to skip the whole operation.
    """

    def test_the_interval_comes_from_the_span_and_the_target(self):
        year = ("2016-01-01T00:00:00Z", "2017-01-01T00:00:00Z")
        # A year over 20 buckets wants ~18 days; the coarsest ladder step that fits is a month.
        self.assertEqual("+1MONTH", _auto_interval_to_solr_gap(20, year))
        # A year over 400 buckets wants ~22 hours, so a day.
        self.assertEqual("+1DAY", _auto_interval_to_solr_gap(400, year))
        # A year over 2 buckets wants half a year, above every step, so the coarsest.
        self.assertEqual("+1YEAR", _auto_interval_to_solr_gap(2, year))

    def test_without_bounds_the_coarsest_interval_is_chosen(self):
        # A fine gap over an unknown range would produce a bucket per document.
        self.assertEqual("+1YEAR", _auto_interval_to_solr_gap(20, None))
        self.assertEqual("+1YEAR", _auto_interval_to_solr_gap(
            20, ("REPLACE_WITH_CORPUS_START", "REPLACE_WITH_CORPUS_END")))

    def test_a_nonsense_target_falls_back_rather_than_raising(self):
        # The fallback target is 10 buckets, so a year wants ~37 days and the ladder gives a quarter.
        year = ("2016-01-01T00:00:00Z", "2017-01-01T00:00:00Z")
        self.assertEqual("+3MONTHS", _auto_interval_to_solr_gap("not a number", year))
        self.assertEqual("+3MONTHS", _auto_interval_to_solr_gap(None, year))

    def test_it_converts_instead_of_being_skipped(self):
        aggs = {"date": {"auto_date_histogram": {"field": "date", "buckets": 20}}}
        result = _convert_aggregations_to_facets(
            aggs, ("2016-01-01T00:00:00Z", "2017-01-01T00:00:00Z"))
        self.assertEqual("range", result["date"]["type"])
        self.assertEqual("+1MONTH", result["date"]["gap"])
        self.assertEqual("2016-01-01T00:00:00Z", result["date"]["start"])

    def test_nested_inside_a_range_aggregation(self):
        # noaa's actual shape: an explicit range list with an auto histogram inside each bucket.
        aggs = {"tmax": {"range": {"field": "TMAX", "ranges": [{"from": 0, "to": 10}]},
                         "aggs": {"date": {"auto_date_histogram": {"field": "date", "buckets": 20}}}}}
        result = _convert_aggregations_to_facets(
            aggs, ("2016-01-01T00:00:00Z", "2017-01-01T00:00:00Z"))
        self.assertEqual([{"from": 0, "to": 10}], result["tmax"]["ranges"])
        self.assertEqual("+1MONTH", result["tmax"]["facet"]["date"]["gap"])


class TestCalendarIntervalToSolrGap(unittest.TestCase):
    def test_known_intervals(self):
        self.assertEqual("+1DAY", _calendar_interval_to_solr_gap("day"))
        self.assertEqual("+1MONTH", _calendar_interval_to_solr_gap("month"))
        self.assertEqual("+1YEAR", _calendar_interval_to_solr_gap("year"))
        self.assertEqual("+1HOUR", _calendar_interval_to_solr_gap("hour"))

    def test_unknown_defaults_to_month(self):
        self.assertEqual("+1MONTH", _calendar_interval_to_solr_gap("fortnight"))

    def test_case_insensitive(self):
        self.assertEqual("+1MONTH", _calendar_interval_to_solr_gap("MONTH"))


class TestJinjaSubstituteRoundTrip(unittest.TestCase):
    """
    A workload file is a Jinja template, so the converter replaces each template token with a
    JSON-safe placeholder, parses, converts, and puts the tokens back. These are the two shapes
    that made the placeholder itself unparseable, both taken from http_logs — the first workload
    to use them, and the reason its operations file was copied verbatim instead of converted.
    """

    def _round_trip(self, source):
        modified, tokens = _jinja_substitute(source)
        # The point of the substitution is that what comes out is parseable JSON.
        parsed = json.loads(modified)
        return modified, _jinja_restore(json.dumps(parsed), tokens)

    def test_expression_inside_a_string_literal_keeps_the_literal_intact(self):
        # "now-{{p}}d/d" — the expression is part of a larger string. Substituting the expression
        # alone puts the placeholder's own quotes mid-literal: "now-"__J_0__"d/d".
        source = '{"gte": "now-{{ p | default(30) }}d/d", "lt": "now/d"}'
        modified, restored = self._round_trip(source)
        self.assertNotIn('"now-"', modified)
        self.assertEqual(json.loads(source.replace("{{ p | default(30) }}", "30")),
                         json.loads(restored.replace("{{ p | default(30) }}", "30")))

    def test_for_loop_is_taken_as_one_block(self):
        # The loop *generates* the array elements, so its opening tag is not a value on its own.
        source = ('{"processors": [\n'
                  '  {% for i in range(1, 101) %}\n'
                  '  {"rename_field": {"field": "status", "target_field": "status_{{ i }}"}}'
                  '{% if not loop.last %},{% endif %}\n'
                  '  {% endfor %}\n'
                  ']}')
        modified, restored = self._round_trip(source)
        self.assertEqual(1, len(json.loads(modified)["processors"]))
        self.assertIn("{% for i in range(1, 101) %}", restored)
        self.assertIn("{% endfor %}", restored)
        self.assertIn("status_{{ i }}", restored)

    def test_conditional_generating_a_key_value_pair(self):
        # A tag can generate a whole pair, not a value, so its placeholder lands where the object
        # expects "key": value. This is upstream's common_operations/force_merge.json, which every
        # workload collects — so until this was fixed, every one of them took the text-only fallback.
        source = ('{"operation": {\n'
                  '  "operation-type": "force-merge",\n'
                  '  "request-timeout": {{ request_timeout | default(60) | tojson }}'
                  '{%- if max_num_segments is defined %},\n'
                  '  "max-num-segments": {{ max_num_segments | tojson }}\n'
                  '  {%- endif %}\n'
                  '}}')
        _, restored = self._round_trip(source)
        for tag in ("{%- if max_num_segments is defined %}", "{%- endif %}",
                    "{{ max_num_segments | tojson }}"):
            self.assertIn(tag, restored)
        self.assertNotIn("__J_", restored)
        self.assertNotIn("null", restored)

    def test_restored_template_renders_the_same_as_the_source(self):
        # Round-tripping to text that parses is not enough: what runs is the *rendered* template,
        # and both branches of the conditional have to come out unchanged.
        jinja2 = __import__("jinja2")
        source = ('{"operation": {\n'
                  '  "request-timeout": {{ request_timeout | default(60) | tojson }}'
                  '{%- if max_num_segments is defined %},\n'
                  '  "max-num-segments": {{ max_num_segments | tojson }}\n'
                  '  {%- endif %}\n'
                  '}}')
        _, restored = self._round_trip(source)
        env = jinja2.Environment()
        for context in ({}, {"max_num_segments": 1}):
            self.assertEqual(json.loads(env.from_string(source).render(**context)),
                             json.loads(env.from_string(restored).render(**context)),
                             msg=f"differs with context {context}")

    def test_placeholder_moved_into_another_string_is_still_restored(self):
        # Translating a range query to Solr syntax moves the placeholder into a longer string, where
        # it no longer has quotes of its own. http_logs' "range" operation shipped with a raw
        # __J_20__ marker in its query until restore also matched the bare form.
        from solrorbit.conversion.query import translate_to_solr_json_dsl
        source = ('{"query": {"range": {"@timestamp": '
                  '{"gte": "now-{{ \'15-05-1998\' | days_ago(now) }}d/d", "lt": "now/d"}}}}')
        modified, tokens = _jinja_substitute(source)
        translated = translate_to_solr_json_dsl(json.loads(modified))
        restored = _jinja_restore(json.dumps(translated), tokens)
        self.assertNotIn("__J_", restored)
        self.assertIn("{{ '15-05-1998' | days_ago(now) }}", restored)

    def test_marker_indices_do_not_collide_by_prefix(self):
        # __J_1__ must not match inside __J_11__, or restoring in index order would corrupt the rest.
        tokens = [("E%d" % i, True) for i in range(13)]
        restored = _jinja_restore('"a[__J_1__ TO __J_11__ TO __J_12__]"', tokens)
        self.assertEqual('"a[E1 TO E11 TO E12]"', restored)

    def test_a_conditional_after_a_completed_pair(self):
        # big5 writes `"field": "agent.name" {% if … %}, "execution_hint": … {% endif %}` — the block
        # sits after a finished pair and carries its own leading comma, so what the placeholder needs
        # is the separator BEFORE it, not a key to belong to.
        source = ('{"cardinality": {\n'
                  '  "field": "agent.name"\n'
                  '  {% if v %}\n'
                  '    , "execution_hint": "ordinals"\n'
                  '  {% endif %}\n'
                  '}}')
        _, restored = self._round_trip(source)
        self.assertIn("{% if v %}", restored)
        self.assertIn('"execution_hint": "ordinals"', restored)
        self.assertNotIn("__J_", restored)
        self.assertNotIn("null", restored)

    def test_a_conditional_standing_in_for_the_pair_after_a_comma(self):
        # The mirror shape, also big5: `"field": "@timestamp",\n {% if … %} "calendar_interval" …`.
        # Here the comma IS in the source and introduces the pair the block generates, so dropping it
        # would leave two values side by side. The two cases need opposite repairs.
        source = ('{"date_histogram": {\n'
                  '  "field": "@timestamp",\n'
                  '  {% if v %}\n'
                  '    "calendar_interval": "hour"\n'
                  '  {% else %}\n'
                  '    "interval": "hour"\n'
                  '  {% endif %}\n'
                  '}}')
        _, restored = self._round_trip(source)
        self.assertIn('"field": "@timestamp",', restored)
        self.assertIn("{% else %}", restored)
        self.assertNotIn("__J_", restored)

    def test_both_conditional_shapes_render_as_the_source_does(self):
        # Parsing is not the point; what runs is the rendered template, and both branches of each
        # conditional have to come out unchanged.
        jinja2 = __import__("jinja2")
        sources = [
            ('{"a": {"field": "x"\n{% if v %}, "hint": "y"\n{% endif %}}}'),
            ('{"a": {"field": "x",\n{% if v %}"i": "hour"\n{% else %}"j": "hour"\n{% endif %}}}'),
        ]
        env = jinja2.Environment()
        for source in sources:
            _, restored = self._round_trip(source)
            for context in ({"v": True}, {"v": False}):
                self.assertEqual(json.loads(env.from_string(source).render(**context)),
                                 json.loads(env.from_string(restored).render(**context)),
                                 msg="%r with %s" % (source, context))

    def test_a_conditional_between_array_elements(self):
        # A fragment file is a bare sequence wrapped in [ … ] before parsing, and big5's schedule
        # closes with `} {% endif %}` after its last task — a tag at the top level, between elements.
        source = ('{"operation": "a"}\n'
                  '{% if v %}\n'
                  ', {"operation": "b"}\n'
                  '{% endif %}')
        modified, tokens = _jinja_substitute(source)
        parsed = json.loads("[" + modified + "]")
        restored = _jinja_restore(json.dumps(parsed), tokens)
        self.assertIn("{% if v %}", restored)
        self.assertNotIn("__J_", restored)

    def test_plain_quoted_expression_still_round_trips(self):
        # The pre-existing shape, to show the two additions did not displace it.
        source = '{"clients": "{{ bulk_indexing_clients | default(8) }}"}'
        _, restored = self._round_trip(source)
        self.assertEqual(source, restored)


class TestHttpLogsShapedFragmentParses(unittest.TestCase):
    """
    Both defects above surfaced as the same symptom — a file the converter could not parse, so it
    fell back to copying it verbatim, OpenSearch operation types and all. This drives the parse
    entry point rather than the substitution, because that fallback is what the user sees.
    """

    def test_fragment_with_both_shapes_parses(self):
        fragment = ('{"name": "range", "body": {"query": {"range": {"@timestamp":\n'
                    '  {"gte": "now-{{ p | default(30) }}d/d", "lt": "now/d"}}}}},\n'
                    '{"name": "renames", "body": {"response_processors": [\n'
                    '  {% for i in range(1, 101) %}\n'
                    '  {"rename_field": {"target_field": "status_{{ i }}"}}'
                    '{% if not loop.last %},{% endif %}\n'
                    '  {% endfor %}\n'
                    ']}}')
        parsed, _ = _parse_jinja_fragment(fragment, wrap_array=True)
        self.assertEqual(["range", "renames"], [entry["name"] for entry in parsed])


if __name__ == "__main__":
    unittest.main()


class TestPipedAndProtoOperations:
    """A piped query and a binary-protocol search are queries, not opaque requests."""

    def test_a_piped_raw_request_becomes_a_solr_sql_operation(self):
        op = {"name": "ppl-term", "operation-type": "raw-request", "path": "/_plugins/_ppl",
              "method": "POST",
              "body": {"query": "source = big5 | where `process.name` = 'kernel' | head 10"}}
        issues, skipped = [], []
        wc._TARGET_COLLECTION = "big5"
        assert wc._convert_operation(op, issues, skipped, "", "") is True
        assert op["path"] == "/solr/big5/sql"
        assert op["form"] is True
        assert op["body"]["stmt"] == \
            "select id from big5 where (process_name = 'kernel') limit 10"

    def test_a_piped_query_sql_cannot_express_is_reported_not_mistranslated(self):
        op = {"name": "ppl-histo", "operation-type": "raw-request", "path": "/_plugins/_ppl",
              "method": "POST",
              "body": {"query": "source = big5 | stats count() by span(`@timestamp`, 1d)"}}
        issues, skipped = [], []
        wc._TARGET_COLLECTION = "big5"
        wc._convert_operation(op, issues, skipped, "", "")
        assert op["path"] == "/_plugins/_ppl"
        assert any("ppl-histo" in issue for issue in issues)

    def test_a_non_piped_raw_request_is_left_alone(self):
        op = {"name": "health", "operation-type": "raw-request", "method": "GET",
              "path": "/solr/admin/collections?action=CLUSTERSTATUS"}
        wc._convert_operation(op, [], [], "", "")
        assert op["path"] == "/solr/admin/collections?action=CLUSTERSTATUS"
        assert "form" not in op

    def test_a_proto_search_body_is_translated_like_a_search_body(self):
        # Left untranslated the operation shipped OpenSearch DSL to Solr, which answers it as a
        # syntactically valid query matching nothing.
        op = {"name": "grpc-term", "operation-type": "proto-search",
              "body": {"query": {"term": {"process.name": {"value": "kernel"}}}}}
        wc._TARGET_COLLECTION = "big5"
        wc._convert_operation(op, [], [], "", "")
        assert op["operation-type"] == "proto-search"
        assert "term" not in json.dumps(op["body"])
        assert op["collection"] == "big5"

    def test_a_piped_query_inside_a_jinja_fragment_is_translated(self):
        # The parsed value of a literal carrying an expression is a bare placeholder, so translating
        # it produced nothing at all and every piped operation was reported untranslatable.
        text = ('{"name": "ppl-default", "operation-type": "raw-request", '
                '"path": "/_plugins/_ppl", "method": "POST", '
                '"body": {"query": "source = {{index_name | default(\'big5\')}} | head 10"}}')
        wc._TARGET_COLLECTION = "big5"
        converted = wc._convert_operations_text(text) if hasattr(wc, "_convert_operations_text") \
            else None
        if converted is None:
            parsed, tokens = wc._parse_jinja_fragment(text)
            wc._convert_operation(parsed, [], [], "", "")
            converted = wc._serialise_jinja_fragment(parsed, tokens)
        # The expression survives in both the path and the statement: substituting the collection the
        # conversion happened to be given would pin an index the workload lets its caller override.
        assert "/solr/{{index_name | default('big5')}}/sql" in converted
        assert "select id from {{index_name | default('big5')}} limit 10" in converted
        assert "head 10" not in converted


class TestQueryStringAndPostFilter(unittest.TestCase):
    """Two query forms that fell through to *:* or were dropped, each measuring the wrong thing."""

    def test_a_query_string_with_an_inline_field_groups_its_terms(self):
        # Untranslated this fell through to *:*: 3,482,624 documents where the query selects 298,029.
        # Solr binds a bare field reference to the next term only and rejects the rest with "no field
        # name specified in query", so the terms are grouped.
        body = translate_to_solr_json_dsl(
            {"query": {"query_string": {"query": "message: monkey jackal bear"}}})
        self.assertEqual("message:(monkey jackal bear)", body["query"])

    def test_a_single_term_query_string_needs_no_grouping(self):
        body = translate_to_solr_json_dsl({"query": {"query_string": {"query": "message: monkey"}}})
        self.assertEqual("message:monkey", body["query"])

    def test_a_query_string_field_name_is_flattened(self):
        body = translate_to_solr_json_dsl(
            {"query": {"query_string": {"query": "log.file.path: /var/log/x"}}})
        self.assertEqual("log_file_path:/var/log/x", body["query"])

    def test_a_query_string_over_several_fields_uses_edismax(self):
        body = translate_to_solr_json_dsl(
            {"query": {"query_string": {"query": "monkey", "fields": ["message", "process.name"]}}})
        self.assertIn('qf="message process_name"', body["query"])

    def test_a_query_string_already_carrying_operators_is_left_intact(self):
        body = translate_to_solr_json_dsl(
            {"query": {"query_string": {"query": "message:(a b) AND process.name:kernel"}}})
        self.assertEqual("message:(a b) AND process_name:kernel", body["query"])

    def test_a_post_filter_becomes_a_tagged_filter(self):
        # Dropped entirely, the operation reported 103,349 hits where upstream reports 4,199.
        body = translate_to_solr_json_dsl({
            "query": {"match": {"message": "monkey"}},
            "post_filter": {"term": {"cloud.region": "us-east-1"}},
        })
        self.assertTrue(any("{!tag=postfilter}" in f for f in body["filter"]))
        self.assertTrue(any("cloud_region" in f for f in body["filter"]))

    def test_the_facets_exclude_the_post_filter(self):
        # A post_filter narrows the hits *after* the aggregations: facets computed over the narrowed
        # set are the one thing a post_filter exists to prevent. Measured against both engines, the
        # five buckets agree to the document only with the exclusion in place.
        body = translate_to_solr_json_dsl({
            "query": {"match": {"message": "monkey"}},
            "post_filter": {"term": {"cloud.region": "us-east-1"}},
            "aggs": {"by_region": {"terms": {"field": "cloud.region", "size": 5}}},
        })
        self.assertEqual("postfilter", body["facet"]["by_region"]["domain"]["excludeTags"])

    def test_a_body_with_no_post_filter_gets_no_exclusion(self):
        body = translate_to_solr_json_dsl({
            "query": {"match": {"message": "monkey"}},
            "aggs": {"by_region": {"terms": {"field": "cloud.region", "size": 5}}},
        })
        self.assertNotIn("domain", body["facet"]["by_region"])


class TestTupleAggregations(unittest.TestCase):
    """
    multi_terms and composite group by a tuple of fields; Solr states that as nested terms facets.

    Skipped, the aggregation vanished and the operation stayed a valid search reporting a hit count and
    no buckets at all — clickbench has 14 of these across 16 operations, which is the shape of a silent
    zero rather than a failure.
    """

    def test_multi_terms_nests_one_level_per_field(self):
        body = translate_to_solr_json_dsl(
            {"aggregations": {"a": {"multi_terms": {
                "terms": [{"field": "WatchID"}, {"field": "ClientIP"}], "size": 10}}}})
        outer = body["facet"]["a"]
        self.assertEqual(("terms", "WatchID", 10), (outer["type"], outer["field"], outer["limit"]))
        inner = outer["facet"]["ClientIP"]
        self.assertEqual(("terms", "ClientIP", 10), (inner["type"], inner["field"], inner["limit"]))

    def test_a_composite_reads_its_fields_from_its_sources(self):
        body = translate_to_solr_json_dsl(
            {"aggregations": {"c": {"composite": {"sources": [
                {"x": {"terms": {"field": "URLHash", "order": "desc"}}},
                {"y": {"terms": {"field": "EventDate", "order": "asc"}}}]}}}})
        outer = body["facet"]["c"]
        self.assertEqual("URLHash", outer["field"])
        self.assertEqual("index desc", outer["sort"])
        self.assertEqual("index asc", outer["facet"]["EventDate"]["sort"])

    def test_a_composite_is_not_truncated_by_a_size(self):
        # It paginates upstream rather than truncating, so a Solr facet must ask for every bucket:
        # a default limit of 10 would have reported ten buckets of a set with hundreds.
        body = translate_to_solr_json_dsl(
            {"aggregations": {"c": {"composite": {"sources": [
                {"x": {"terms": {"field": "A"}}}]}}}})
        self.assertEqual(-1, body["facet"]["c"]["limit"])

    def test_a_multi_terms_size_is_honoured(self):
        body = translate_to_solr_json_dsl(
            {"aggregations": {"a": {"multi_terms": {
                "terms": [{"field": "A"}], "size": 25}}}})
        self.assertEqual(25, body["facet"]["a"]["limit"])

    def test_a_metric_sub_aggregation_lands_on_the_innermost_level(self):
        # A metric belongs to the tuple, so it must be computed inside the last grouping level, not
        # beside the first.
        body = translate_to_solr_json_dsl(
            {"aggregations": {"a": {
                "multi_terms": {"terms": [{"field": "A"}, {"field": "B"}], "size": 5},
                "aggregations": {"m": {"avg": {"field": "N"}}}}}})
        self.assertNotIn("m", body["facet"]["a"].get("facet", {}))
        self.assertEqual("avg(N)", body["facet"]["a"]["facet"]["B"]["facet"]["m"])

    def test_a_tuple_aggregation_naming_no_field_is_reported_not_emitted(self):
        body = translate_to_solr_json_dsl({"aggregations": {"a": {"multi_terms": {"terms": []}}}})
        self.assertNotIn("facet", body)


class TestRawSearchAndDroppedAggregations(unittest.TestCase):
    """A raw request to the search endpoint, and an aggregation that cannot be carried."""

    def test_a_raw_request_to_the_search_endpoint_becomes_a_search(self):
        # clickbench declares all 45 of its DSL operations this way. Left as raw requests they would
        # have sent OpenSearch query DSL to a Solr path that does not exist.
        op = {"name": "dsl-q01", "operation-type": "raw-request", "path": "/_search",
              "method": "POST", "body": {"query": {"match_all": {}}, "size": 0}}
        wc._TARGET_COLLECTION = "clickbench"
        wc._convert_operation(op, [], [], "", "")
        self.assertEqual("search", op["operation-type"])
        self.assertEqual("*:*", op["body"]["query"])
        self.assertNotIn("path", op)
        self.assertEqual("clickbench", op["collection"])

    def test_a_raw_request_to_another_endpoint_is_left_alone(self):
        op = {"name": "flush", "operation-type": "raw-request", "method": "POST",
              "path": "/clickbench/_flush", "body": {}}
        wc._convert_operation(op, [], [], "", "")
        self.assertEqual("raw-request", op["operation-type"])
        self.assertEqual("/clickbench/_flush", op["path"])

    def test_an_untranslatable_aggregation_is_reported_not_silently_dropped(self):
        # The operation stays a valid search reporting a hit count and no buckets, which reads as a
        # working search that measures nothing. 16 clickbench operations were in this state.
        op = {"name": "dsl-q29", "operation-type": "raw-request", "path": "/_search",
              "method": "POST",
              "body": {"aggregations": {"c": {"composite": {"sources": [
                  {"k": {"terms": {"script": {"source": "..."}}}}]}}}}}
        issues = []
        wc._TARGET_COLLECTION = "clickbench"
        wc._convert_operation(op, issues, [], "", "")
        self.assertTrue(any("dsl-q29" in i and "hit count only" in i for i in issues))

    def test_a_translatable_aggregation_raises_no_issue(self):
        op = {"name": "dsl-ok", "operation-type": "raw-request", "path": "/_search",
              "method": "POST",
              "body": {"aggregations": {"t": {"terms": {"field": "CounterID", "size": 5}}}}}
        issues = []
        wc._TARGET_COLLECTION = "clickbench"
        wc._convert_operation(op, issues, [], "", "")
        self.assertEqual([], issues)
        self.assertIn("facet", op["body"])

    def test_a_composite_source_computed_by_a_script_is_refused_whole(self):
        # Emitting the remaining sources would bucket by a different tuple than upstream measures.
        body = translate_to_solr_json_dsl({"aggregations": {"c": {"composite": {"sources": [
            {"k": {"terms": {"script": {"source": "..."}}}},
            {"d": {"terms": {"field": "EventDate"}}}]}}}})
        self.assertNotIn("facet", body)

    def test_a_composite_source_may_bucket_by_time(self):
        body = translate_to_solr_json_dsl({"aggregations": {"c": {"composite": {"sources": [
            {"M": {"date_histogram": {"field": "EventTime", "fixed_interval": "1m"}}}]}}}})
        facet = body["facet"]["c"]
        self.assertEqual("range", facet["type"])
        self.assertEqual("EventTime", facet["field"])
        self.assertEqual("+1MINUTE", facet["gap"])

    def test_a_wildcard_query_is_translated(self):
        # Untranslated it fell through to *:* and matched the whole corpus.
        body = translate_to_solr_json_dsl(
            {"query": {"wildcard": {"URL": {"value": "*google*"}}}})
        self.assertEqual("URL:*google*", body["query"])

    def test_a_prefix_query_gets_its_trailing_metacharacter(self):
        body = translate_to_solr_json_dsl({"query": {"prefix": {"URL": {"value": "http://x"}}}})
        self.assertEqual("URL:http://x*", body["query"])

    def test_a_cardinality_aggregation_uses_the_exact_form(self):
        # Upstream's cardinality is a HyperLogLog estimate — measured on big5 it answered 5,958 where
        # the true distinct count is 5,909. Solr's unique() is exact, and that is the deliberate choice.
        body = translate_to_solr_json_dsl(
            {"aggregations": {"u": {"cardinality": {"field": "UserID"}}}})
        self.assertEqual("unique(UserID)", body["facet"]["u"])


class TestRangeAndEmptyTerm(unittest.TestCase):
    """
    Two query spellings OpenSearch's own query builder serialises, both mistranslated.

    Found by sending the ported bodies to a live Solr rather than by reading them: one produced a query
    matching the whole corpus, the other a syntax error for 18 operations.
    """

    def test_a_from_to_range_keeps_its_bounds(self):
        # Reading only gte/lte lost both and left field:[* TO *], which matches every document that has
        # the field: clickbench's q44 reported 1,498,137 where the query selects 663.
        body = translate_to_solr_json_dsl({"query": {"range": {
            "RegionID": {"from": 200, "to": 300, "include_lower": True, "include_upper": True}}}})
        self.assertEqual("RegionID:[200 TO 300]", body["query"])

    def test_an_exclusive_from_to_range_uses_braces(self):
        body = translate_to_solr_json_dsl({"query": {"range": {
            "A": {"from": 1, "to": 9, "include_lower": False, "include_upper": False}}}})
        self.assertEqual("A:{1 TO 9}", body["query"])

    def test_gt_and_lt_are_exclusive_too(self):
        # Rendering them as an inclusive range widened it by one value at each end.
        body = translate_to_solr_json_dsl({"query": {"range": {"A": {"gt": 1, "lt": 9}}}})
        self.assertEqual("A:{1 TO 9}", body["query"])

    def test_gte_and_lte_stay_inclusive(self):
        body = translate_to_solr_json_dsl({"query": {"range": {"A": {"gte": 1, "lte": 9}}}})
        self.assertEqual("A:[1 TO 9]", body["query"])

    def test_a_half_open_range_keeps_its_one_bound(self):
        body = translate_to_solr_json_dsl({"query": {"range": {"A": {"from": 5}}}})
        self.assertEqual("A:[5 TO *]", body["query"])

    def test_an_empty_term_becomes_a_query_that_selects_nothing(self):
        # `field:` is not a query — Solr answers 'Encountered " ")"' — and the generated configset removes
        # blank values before indexing, mirroring OpenSearch, so no document has one. Measured against
        # both engines, the whole bool query then reports 83,461 on each.
        body = translate_to_solr_json_dsl({"query": {"bool": {
            "must": [{"exists": {"field": "M"}}],
            "must_not": [{"term": {"M": {"value": ""}}}]}}})
        self.assertNotIn("M:)", body["query"])
        self.assertIn("M:[* TO *]", body["query"])

    def test_a_non_empty_term_is_unaffected(self):
        body = translate_to_solr_json_dsl({"query": {"term": {"M": {"value": "x"}}}})
        self.assertEqual("M:x", body["query"])

    def test_a_metadata_value_count_counts_documents(self):
        # clickbench counts values of _index, which every document has once. Solr has no such field and
        # answered 'undefined field: "_index"' — a 400 for 21 operations. `count(*)` is SQL's spelling,
        # not the JSON Facet API's, which answers a SyntaxError; countvals(id) is the same number.
        body = translate_to_solr_json_dsl(
            {"aggregations": {"c": {"value_count": {"field": "_index"}}}})
        self.assertEqual("countvals(id)", body["facet"]["c"])

    def test_a_value_count_over_a_real_field_is_unaffected(self):
        body = translate_to_solr_json_dsl(
            {"aggregations": {"c": {"value_count": {"field": "URL"}}}})
        self.assertEqual("countvals(URL)", body["facet"]["c"])


class TestSerialisedNumbersAndBoost(unittest.TestCase):
    """
    What OpenSearch's own query builder writes into a query, beyond the query itself.

    Both of these came from clickbench's serialised DSL, which is what a real workload ships rather than
    the hand-written form a test usually uses.
    """

    def test_a_whole_json_number_loses_its_fraction(self):
        # Every number is serialised as a JSON double, so a term list over an integer field arrives as
        # [-1.0, 6.0]. OpenSearch coerces; Solr refuses with "Invalid Number: -1.0 for field ...".
        body = translate_to_solr_json_dsl(
            {"query": {"terms": {"TraficSourceID": [-1.0, 6.0]}}})
        self.assertEqual(["{!terms f=TraficSourceID}-1,6"], body["filter"])

    def test_a_fractional_number_keeps_its_fraction(self):
        # The field is then not an integer one, so truncating would change the query.
        body = translate_to_solr_json_dsl({"query": {"term": {"A": {"value": 3.5}}}})
        self.assertEqual("A:3.5", body["query"])

    def test_a_boost_beside_the_field_does_not_eat_the_term_list(self):
        # Dict order put `boost` first, the loop returned on it, and the whole term list was lost to *:*.
        body = translate_to_solr_json_dsl(
            {"query": {"terms": {"A": ["x", "y"], "boost": 1.0}}})
        self.assertEqual(["{!terms f=A}x,y"], body["filter"])
        self.assertEqual("*:*", body["query"])

    def test_a_boolean_is_left_as_it_is(self):
        # A bool is a subclass of int, not of float, so the coercion never sees one.
        from solrorbit.conversion.query import _numeric_literal
        self.assertIs(True, _numeric_literal(True))
        self.assertEqual(1, _numeric_literal(1.0))

    def test_the_fq_path_coerces_its_numbers_too(self):
        # There are two term-list paths — the fast top-level one and the bool-clause one — and only one
        # was fixed at first. Exercise the fq builder directly, since a body reaching either path gives
        # the same answer and would not tell them apart.
        from solrorbit.conversion.query import _translate_node_for_fq
        self.assertEqual("{!terms f=T}-1,6",
                        _translate_node_for_fq({"terms": {"T": [-1.0, 6.0]}}))

    def test_the_fq_path_ignores_a_boost_beside_the_field(self):
        from solrorbit.conversion.query import _translate_node_for_fq
        self.assertEqual("{!terms f=A}x,y",
                        _translate_node_for_fq({"terms": {"boost": 1.0, "A": ["x", "y"]}}))

    def test_the_date_bounds_come_from_a_from_to_range_too(self):
        # A range facet without bounds is refused outright — "Missing required parameter: 'start'" — and
        # reading only gte/lte left them empty for every serialised query.
        from solrorbit.conversion.query import _date_bounds_from_query
        start, end = _date_bounds_from_query({"query": {"bool": {"filter": [
            {"range": {"EventDate": {"from": "2013-07-01", "to": "2013-07-15"}}}]}}})
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)

    def test_a_terms_query_inside_a_bool_filter_still_reaches_fq(self):
        # The top-level fast path is guarded by len(query) == 1, which a serialised query with a boost
        # fails, so the clause path has to handle it as well.
        body = translate_to_solr_json_dsl({"query": {"bool": {"filter": [
            {"terms": {"T": [-1.0, 6.0], "boost": 1.0}}]}}})
        self.assertEqual(["{!terms f=T}-1,6"], body["filter"])


class TestNestedNegativeClause(unittest.TestCase):
    """A nested must_not with nothing positive beside it."""

    def test_a_negative_only_group_gets_something_to_subtract_from(self):
        # `+(-(URL:*x*))` matches nothing in Lucene, where `-(URL:*x*)` beside a positive clause matches
        # 564. Measured on a live pair: the operation reported 0 hits against upstream's 128, and 564 once
        # the group had *:* to subtract from — which is the exact count over the corpus.
        body = translate_to_solr_json_dsl({"query": {"bool": {"must": [
            {"wildcard": {"Title": {"wildcard": "*Google*"}}},
            {"bool": {"must_not": [{"wildcard": {"URL": {"wildcard": "*.google.*"}}}]}}]}}})
        self.assertEqual("+(Title:*Google*) +(*:* -(URL:*.google.*))", body["query"])

    def test_a_group_with_a_positive_clause_is_left_alone(self):
        body = translate_to_solr_json_dsl({"query": {"bool": {"must": [
            {"bool": {"must": [{"term": {"A": {"value": "x"}}}],
                      "must_not": [{"term": {"B": {"value": "y"}}}]}}]}}})
        self.assertNotIn("*:* +", body["query"])
        self.assertIn("A:x", body["query"])
