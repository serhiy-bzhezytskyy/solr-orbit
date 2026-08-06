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

"""Unit tests for solrorbit/conversion/detector.py"""

import unittest

from solrorbit.conversion.query import translate_to_solr_json_dsl
from solrorbit.conversion.workload_converter import _convert_operation


class TestAggregationOnlyBodies(unittest.TestCase):
    """
    An aggregation-only body carries no "query" at all. translate_to_solr_json_dsl has always
    handled that; the converter decided whether to CALL it by looking for a dict "query", so such
    bodies were copied through in OpenSearch syntax, where "size" and "aggs" mean nothing to Solr.
    The operation then loads, answers, and silently has no aggregation - which is why these tests
    drive _convert_operation rather than the translation directly.
    """

    def test_an_aggregation_without_a_query_is_converted(self):
        op = {"name": "articles_monthly_agg", "operation-type": "search",
              "body": {"size": 0, "aggs": {"over_time": {
                  "date_histogram": {"field": "timestamp", "calendar_interval": "month"}}}}}

        _convert_operation(op, [], [], "/tmp", None)

        self.assertNotIn("aggs", op["body"])
        self.assertNotIn("size", op["body"])
        self.assertEqual(0, op["body"]["limit"])
        self.assertEqual("range", op["body"]["facet"]["over_time"]["type"])
        self.assertEqual("+1MONTH", op["body"]["facet"]["over_time"]["gap"])

    def test_a_sort_only_body_is_converted(self):
        op = {"name": "sorted", "operation-type": "search",
              "body": {"sort": [{"timestamp": "asc"}]}}

        _convert_operation(op, [], [], "/tmp", None)

        self.assertEqual("timestamp asc", op["body"]["sort"])


class TestDateHistogramBounds(unittest.TestCase):
    """
    Solr range facets need explicit bounds; OpenSearch derives them from the data. Bounds that miss
    the corpus produce an empty facet, and an empty facet reads as a working operation - so a guess
    is worse than an obvious placeholder. pmc's timestamps are 2010-2016 and the old hardcoded
    default spanned 2016-2027, which would have returned almost nothing.
    """

    def test_bounds_come_from_the_operations_own_range(self):
        body = {
            "query": {"range": {"dropoff_datetime": {
                "gte": "2015-01-01 00:00:00", "lt": "2016-01-01 00:00:00"}}},
            "size": 0,
            "aggs": {"over_time": {
                "date_histogram": {"field": "dropoff_datetime", "calendar_interval": "month"}}},
        }

        facet = translate_to_solr_json_dsl(body)["facet"]["over_time"]

        self.assertEqual("2015-01-01T00:00:00Z", facet["start"])
        self.assertEqual("2016-01-01T00:00:00Z", facet["end"])

    def test_a_range_inside_a_bool_filter_is_found(self):
        body = {
            "query": {"bool": {"filter": {"range": {"ts": {"gte": "2015-01-01", "lte": "2015-02-01"}}}}},
            "aggs": {"over_time": {"date_histogram": {"field": "ts", "calendar_interval": "day"}}},
        }

        facet = translate_to_solr_json_dsl(body)["facet"]["over_time"]

        self.assertEqual("2015-01-01", facet["start"])
        self.assertEqual("2015-02-01", facet["end"])

    def test_without_a_range_the_bounds_are_an_unmissable_placeholder(self):
        body = {"size": 0, "aggs": {"over_time": {
            "date_histogram": {"field": "timestamp", "calendar_interval": "month"}}}}

        facet = translate_to_solr_json_dsl(body)["facet"]["over_time"]

        # A date literal here would be a guess that silently empties the facet.
        self.assertEqual("REPLACE_WITH_CORPUS_START", facet["start"])
        self.assertEqual("REPLACE_WITH_CORPUS_END", facet["end"])


class TestPaginatedOperations(unittest.TestCase):
    """
    "pages" and "results-per-page" say how far a search walks. A plain search ignores them, so a
    25-page walk became a single request measuring something else.
    """

    def test_a_search_with_pages_becomes_a_paginated_search(self):
        op = {"name": "scroll", "operation-type": "search", "pages": 25,
              "results-per-page": 100, "body": {"query": {"match_all": {}}}}

        kept = _convert_operation(op, [], [], "/tmp", None)

        self.assertTrue(kept)
        self.assertEqual("paginated-search", op["operation-type"])
        self.assertEqual(25, op["pages"])
        self.assertEqual(100, op["results-per-page"])

    def test_a_plain_search_stays_a_search(self):
        op = {"name": "term", "operation-type": "search",
              "body": {"query": {"term": {"body": "physician"}}}}

        _convert_operation(op, [], [], "/tmp", None)

        self.assertEqual("search", op["operation-type"])


class TestNestedQueries(unittest.TestCase):
    """
    A `nested` clause asks about one object of a nested field at a time. Solr's equivalent is a block
    join over child documents, and every one of these assertions records a spelling that was measured
    against OpenSearch on nested's 1,000-document sample.
    """

    CUTOFF = "2012-01-01T00:00:00Z"

    def _nested_body(self, **inner):
        clause = {"path": "answers", "query": {"range": {"answers.date": {"lte": self.CUTOFF}}}}
        clause.update(inner)
        return {"query": {"bool": {"must": [{"match": {"tag": "php"}}, {"nested": clause}]}}}

    def test_a_nested_clause_becomes_a_referenced_block_join(self):
        # ⚠️ The child query cannot be inlined. Measured against upstream's 1 hit: written inline inside
        # the bool clause Solr answered 0 (the local-params parser consumes the whole clause), as the
        # entire q it answered 80 (the tag clause vanished), and referenced through a parameter it
        # answered 1. So the parameter is not a stylistic choice.
        out = translate_to_solr_json_dsl(self._nested_body())

        self.assertIn("_query_:", out["query"])
        self.assertIn("v=$nq0", out["query"])
        self.assertIn("{!parent which='_nested_parent_:true'", out["query"])
        self.assertIn("+(tag:php)", out["query"])
        self.assertIn("nq0", out["params"])

    def test_the_child_query_carries_the_path_and_the_clause(self):
        # ⚠️ The path alone is not the child query: with a childFilter of _nested_path_:answers a
        # document returned 2 children where inner_hits returned 1, the extra being a 2015 answer the
        # query excludes.
        out = translate_to_solr_json_dsl(self._nested_body())

        self.assertIn("+_nested_path_:answers", out["params"]["nq0"])
        self.assertIn(self.CUTOFF, out["params"]["nq0"])

    def test_a_nested_leaf_is_named_for_the_child_not_the_parents_copy(self):
        # ⚠️ Two fields carry the leaf: the child's answer_date and the parent's flattened answers_date.
        # A child query naming the parent's copy asks for a field no child document has, and Solr
        # answers 0 rather than failing.
        out = translate_to_solr_json_dsl(self._nested_body())

        self.assertIn("answer_date:", out["params"]["nq0"])
        self.assertNotIn("answers_date:", out["params"]["nq0"])

    def test_inner_hits_becomes_a_child_transformer_with_the_query_and_its_own_fl(self):
        out = translate_to_solr_json_dsl(self._nested_body(inner_hits={"size": 3}))

        self.assertIn("[child childFilter=$nq0", out["fields"])
        self.assertIn("limit=3", out["fields"])
        # ⚠️ Without its own fl a child is rendered through the outer one: measured, outer fl=qid,tag —
        # parent fields a child does not have — returned {} per child while the count stayed right.
        self.assertIn("fl=answer_*", out["fields"])

    def test_a_body_with_no_inner_hits_asks_for_no_children(self):
        self.assertNotIn("fields", translate_to_solr_json_dsl(self._nested_body()))

    def test_a_nested_sort_becomes_childfield_over_a_separate_block_join(self):
        body = {"query": {"match": {"tag": "php"}},
                "sort": [{"answers.date": {"mode": "max", "order": "desc",
                                           "nested": {"path": "answers"}}}]}

        out = translate_to_solr_json_dsl(body)

        # ⭐ The block join goes in its own parameter, not in q: with it in q the sort returned 74 of the
        # sample's 80 matching documents, dropping the 6 questions with no answers. With q the parent
        # query and the join in bjq it returned all 80 in upstream's order.
        self.assertEqual("childfield(answer_date,$bjq) desc", out["sort"])
        self.assertEqual("tag:php", out["query"])
        self.assertIn("{!parent which='_nested_parent_:true'}", out["params"]["bjq"])

    def test_an_ordinary_sort_is_untouched_by_the_nested_handling(self):
        out = translate_to_solr_json_dsl({"sort": [{"timestamp": {"order": "desc"}}]})

        self.assertEqual("timestamp desc", out["sort"])
        self.assertNotIn("params", out)

    def test_a_nested_aggregation_becomes_a_facet_over_the_child_documents(self):
        body = {"size": 0, "aggs": {"answers": {"nested": {"path": "answers"}, "aggs": {
            "date_histo": {"date_histogram": {"field": "answers.date",
                                              "calendar_interval": "month"}}}}}}

        facet = translate_to_solr_json_dsl(body)["facet"]

        # ⛔ Left untranslated this logged "Unsupported aggregation type 'nested'" and the operation
        # became a search with a hit count and no facet — the shape of a silent zero. Measured with the
        # domain: all 91 monthly buckets identical to upstream's, 1,977 answers counted on both sides.
        self.assertIn("date_histo", facet)
        self.assertEqual("_nested_path_:answers", facet["date_histo"]["domain"]["query"])
        # ⚠️ And the field is the child's. An aggregation names its field in a value, not a key, so
        # renaming keys alone left the facet computed over the children while naming a field only the
        # parent has — every bucket would have been empty.
        self.assertEqual("answer_date", facet["date_histo"]["field"])

    def test_a_nested_aggregation_does_not_swallow_an_ordinary_one(self):
        body = {"size": 0, "aggs": {"by_tag": {"terms": {"field": "tag", "size": 5}}}}

        facet = translate_to_solr_json_dsl(body)["facet"]

        self.assertEqual("terms", facet["by_tag"]["type"])
        self.assertNotIn("domain", facet["by_tag"])


class TestNestedScoping(unittest.TestCase):
    """
    A child document is a document of its own in Solr and shares the collection with its parent, so a
    query that does not exclude the children counts them. Measured on nested's sample: match_all
    answered 1,000 upstream and 2,977 in Solr — 1,000 questions plus 1,977 answers.
    """

    def setUp(self):
        from solrorbit.conversion import query
        self._was = query._HAS_NESTED_FIELDS
        self.addCleanup(query.set_nested_fields_present, self._was)

    def test_a_nested_workload_scopes_every_operation_to_top_level_documents(self):
        from solrorbit.conversion.query import set_nested_fields_present
        set_nested_fields_present(True)

        out = translate_to_solr_json_dsl({"query": {"match_all": {}}})

        # ⚠️ The parent marker is the wrong filter here: it marks a document that *has* children, and 40
        # of the sample's 1,000 questions have none, so filtering on it answered 960. "Has no nested
        # path" answered 1,000 exactly.
        self.assertIn("-_nested_path_:[* TO *]", out["filter"])
        self.assertNotIn("_nested_parent_:true", " ".join(out["filter"]))

    def test_a_workload_with_no_nested_field_gets_no_such_filter(self):
        from solrorbit.conversion.query import set_nested_fields_present
        set_nested_fields_present(False)

        out = translate_to_solr_json_dsl({"query": {"match_all": {}}})

        # ⚠️ Emitting it unconditionally is not harmless: a collection with no such field answers
        # `undefined field: "_nested_path_"` and refuses the query outright — measured against a
        # collection from another workload.
        self.assertNotIn("filter", out)


if __name__ == "__main__":
    unittest.main()
