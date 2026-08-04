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
