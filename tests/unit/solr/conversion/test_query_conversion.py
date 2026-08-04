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


if __name__ == "__main__":
    unittest.main()
