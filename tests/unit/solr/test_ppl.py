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

from solrorbit.conversion.ppl import facet_body_for_piped_query, translate_ppl_to_sql


class TestPipedQueryTranslation:
    def test_head_becomes_a_limit(self):
        assert translate_ppl_to_sql("source = big5 | head 10") == "select id from big5 limit 10"

    def test_where_becomes_a_predicate_with_the_field_flattened(self):
        statement = translate_ppl_to_sql(
            "source = big5 | where `log.file.path` = '/var/log/x' | head 5")
        assert statement == "select id from big5 where (log_file_path = '/var/log/x') limit 5"

    def test_a_leading_search_expression_is_a_filter_not_a_scan(self):
        # `source = big5 match(...)` filters; dropping the expression would measure a full scan.
        statement = translate_ppl_to_sql("source = big5 match(`process.name`, 'kernel') | head 10")
        assert "process_name = 'kernel'" in statement

    def test_a_multi_word_search_value_is_parenthesised(self):
        # Solr SQL passes the right-hand side to the query parser, where a bare multi-word value is a
        # phrase: unparenthesised this matched 0 documents where the piped query matched 298,029.
        statement = translate_ppl_to_sql(
            "source = big5 query_string(['message'], 'monkey jackal bear') | head 10")
        assert "message = '(monkey jackal bear)'" in statement

    def test_a_single_word_search_value_is_left_alone(self):
        statement = translate_ppl_to_sql("source = big5 match(`process.name`, 'kernel') | head 10")
        assert "'(kernel)'" not in statement

    def test_a_timestamp_gets_the_iso_separator(self):
        # Solr's date field rejects a space where the 'T' belongs.
        statement = translate_ppl_to_sql(
            "source = big5 | where `@timestamp` >= '2023-01-01 00:00:00' | head 10")
        assert "'2023-01-01T00:00:00Z'" in statement

    def test_stats_becomes_a_group_by_with_the_keys_selected(self):
        statement = translate_ppl_to_sql(
            "source = big5 | stats count() by `process.name`, `cloud.region`")
        assert statement.startswith("select process_name, cloud_region, count(*) as ")
        assert " group by process_name, cloud_region" in statement

    def test_distinct_count_becomes_count_distinct(self):
        statement = translate_ppl_to_sql("source = big5 | stats dc(`agent.name`)")
        assert "count(distinct agent_name)" in statement

    def test_sorting_by_an_aggregation_resolves_to_its_alias(self):
        # A piped query sorts by its own spelling of the aggregation; SQL answers "Column 'count()'
        # not found in any table" unless the SELECT list gave it an alias to sort by.
        statement = translate_ppl_to_sql(
            "source = big5 | stats count() by `process.name` | sort - `count()` | head 10")
        assert "`count()`" not in statement
        assert " order by " in statement

    def test_an_explicit_alias_is_used_for_the_sort(self):
        statement = translate_ppl_to_sql(
            "source = big5 | stats count() as c by `cloud.region` | sort - c | head 50")
        assert "count(*) as c" in statement
        assert statement.endswith("order by c desc limit 50")

    def test_sort_direction_is_read_from_the_sign(self):
        assert "order by `@timestamp` asc" in translate_ppl_to_sql(
            "source = big5 | sort + `@timestamp` | head 10")
        assert "order by `@timestamp` desc" in translate_ppl_to_sql(
            "source = big5 | sort - `@timestamp` | head 10")

    def test_a_field_sql_cannot_name_bare_is_quoted(self):
        assert "`@timestamp`" in translate_ppl_to_sql("source = big5 | sort + `@timestamp` | head 1")

    def test_a_jinja_filter_pipe_is_not_a_stage_separator(self):
        # A workload file is a Jinja template and a filter uses the same pipe: splitting on the bare
        # pipe cut `{{index_name | default('big5')}}` in half.
        statement = translate_ppl_to_sql("source = {{index_name | default('big5')}} | head 10")
        assert statement == "select id from {{index_name | default('big5')}} limit 10"

    def test_the_collection_argument_overrides_the_index_named_in_the_query(self):
        assert translate_ppl_to_sql("source = big5 | head 3", collection="c1") == \
            "select id from c1 limit 3"

    def test_a_projection_becomes_the_select_list(self):
        # `fields a, b` names the columns to return.
        statement = translate_ppl_to_sql(
            "source = cb | where SearchPhrase != '' | fields SearchPhrase | head 10")
        assert statement == "select SearchPhrase from cb where (SearchPhrase is not null) limit 10"

    def test_a_projection_returns_only_what_it_projects(self):
        # An earlier version added the sort key to the SELECT list, believing Solr required it there. It
        # does not — measured against a live node — and adding it changed what the operation returns: the
        # rows carried a column upstream does not project, so the two result shapes could not be compared.
        statement = translate_ppl_to_sql(
            "source = cb | sort EventTime | fields SearchPhrase | head 10")
        assert statement == "select SearchPhrase from cb order by EventTime asc limit 10"

    def test_a_query_without_head_uses_the_measured_piped_page_size(self):
        # Measured against a live node: `source = x | fields y` answers with size=10000, total=10000.
        # A default of 10 under-reported such a query by three orders of magnitude.
        statement = translate_ppl_to_sql("source = cb | fields UserID")
        assert statement.endswith("limit 10000")

    def test_an_explicit_head_is_not_overridden_by_the_default(self):
        assert translate_ppl_to_sql("source = cb | head 25").endswith("limit 25")

    def test_a_stats_option_is_dropped_rather_than_selected(self):
        # Upstream writes `stats {% if … %}bucket_nullable = false {% endif %}count()`. The option is a
        # directive to the piped engine about empty buckets, not a value: left in place it glued itself
        # to the aggregation, `count()` stopped matching, and Solr answered "No match found for function
        # signature". Eight of big5's piped operations are written this way.
        statement = translate_ppl_to_sql(
            "source = big5 | stats {% if v %}bucket_nullable = false {% endif %}"
            "count() as country by `cloud.region` | sort - country | head 50")
        assert statement == ("select cloud_region, count(*) as country from big5 "
                            "group by cloud_region order by country desc limit 50")

    def test_a_stats_option_before_a_distinct_count_is_dropped_too(self):
        statement = translate_ppl_to_sql(
            "source = big5 | stats {% if v %}bucket_nullable = false {% endif %}dc(`agent.name`)")
        assert "count(distinct agent_name)" in statement
        assert "bucket_nullable" not in statement

    def test_bang_equal_becomes_the_standard_spelling(self):
        # Solr's SQL layer runs at a conformance level that rejects it: "Bang equal '!=' is not allowed
        # under the current SQL conformance level".
        statement = translate_ppl_to_sql("source = cb | where AdvEngineID != 0 | stats count()")
        assert "AdvEngineID <> 0" in statement
        assert "!=" not in statement

    def test_a_like_function_becomes_the_infix_operator(self):
        # Solr answers 'Encountered "like" at line 1' to the function form.
        statement = translate_ppl_to_sql("source = cb | where like(URL, '%google%') | head 10")
        assert "URL like '%google%'" in statement

    def test_a_predicate_over_an_aggregate_becomes_having(self):
        # A `where` after a `stats` filters the buckets. As a WHERE clause Solr answered "Column 'c' not
        # found in any table" — the alias does not exist until the grouping has happened.
        statement = translate_ppl_to_sql(
            "source = cb | stats count() as c by CounterID | where c > 100000 | sort - c | head 25")
        assert " having " in statement
        assert statement.index(" group by ") < statement.index(" having ")

    def test_a_having_names_the_expression_not_the_alias(self):
        # Solr will not resolve the alias in a HAVING either, but accepts the aggregation itself.
        statement = translate_ppl_to_sql(
            "source = cb | stats count() as c by CounterID | where c > 100000")
        assert "having (count(*) > 100000)" in statement

    def test_a_predicate_over_a_plain_field_stays_in_where(self):
        statement = translate_ppl_to_sql(
            "source = cb | where URL != '' | stats count() as c by CounterID")
        assert "where (URL is not null)" in statement
        assert " having " not in statement

    def test_an_emptiness_test_becomes_a_presence_test(self):
        # `field <> ''` does not exclude a document that has no such field: Solr matched all 1,498,137
        # documents where 1,300,738 have no SearchPhrase at all and the true answer is 197,399. In a store
        # that omits an empty value, "not empty" is "present".
        statement = translate_ppl_to_sql("source = cb | where SearchPhrase != '' | stats count()")
        assert "SearchPhrase is not null" in statement
        assert "<> ''" not in statement

    def test_a_non_empty_comparison_is_not_turned_into_a_presence_test(self):
        statement = translate_ppl_to_sql("source = cb | where AdvEngineID != 0 | stats count()")
        assert "AdvEngineID <> 0" in statement
        assert "is not null" not in statement

    def test_an_averaged_integer_column_is_cast(self):
        # Solr's SQL layer averages an integer column with integer arithmetic: avg(ResolutionWidth)
        # answered 1513 where the true mean is 1513.4086448702622 — which is what upstream reports and
        # what a Solr JSON facet reports for the same field.
        statement = translate_ppl_to_sql("source = cb | stats avg(ResolutionWidth)")
        assert "avg(cast(ResolutionWidth as double))" in statement

    def test_a_sum_is_cast_too_so_it_cannot_overflow_silently(self):
        # A total that does not fit a signed 64-bit integer comes back as Long.MAX_VALUE with no error:
        # sum(UserID) answered 9223372036854775807 where the true total is 3.789e+24. A clamped maximum
        # reads as a real number, which is worse than a failure.
        statement = translate_ppl_to_sql("source = cb | stats sum(UserID)")
        assert "sum(cast(UserID as double))" in statement

    def test_a_count_is_not_cast(self):
        statement = translate_ppl_to_sql("source = cb | stats count()")
        assert "count(*)" in statement
        assert "cast" not in statement

    def test_no_more_sorts_than_grouping_keys(self):
        # "If multiple sorts are specified there must be a sort for each bucket." Measured: one bucket
        # with two sorts is refused, two buckets with two sorts is accepted. A piped query sorts by the
        # metric then tie-breaks on every grouping key, which exceeds that.
        one = translate_ppl_to_sql(
            "source = cb | stats count() as PageViews by URL | sort - PageViews, URL | head 10")
        assert one.endswith("order by PageViews desc limit 10")
        two = translate_ppl_to_sql(
            "source = cb | stats count() as PageViews by URLHash, EventDate "
            "| sort - PageViews, URLHash, EventDate | head 10")
        assert two.endswith("order by PageViews desc, URLHash asc limit 10")

    def test_a_query_that_does_not_group_keeps_every_sort(self):
        statement = translate_ppl_to_sql("source = cb | sort - EventTime, SearchPhrase | head 10")
        assert statement.endswith("order by EventTime desc, SearchPhrase asc limit 10")

    def test_a_string_length_has_no_sql_spelling(self):
        # Three spellings tried against a live node: length() and strlen() are unknown functions, and
        # char_length() parses but "avg aggregation not supported for string".
        assert translate_ppl_to_sql(
            "source = cb | stats avg(length(URL)) as l by CounterID") is None

    def test_an_aggregate_over_a_computed_expression_has_no_sql_spelling(self):
        # Solr SQL answers a bare "null" to sum(ResolutionWidth+1) while accepting sum(ResolutionWidth).
        # A bare null is exactly the failure that reads as an empty result.
        assert translate_ppl_to_sql("source = cb | stats sum(ResolutionWidth+1)") is None
        assert translate_ppl_to_sql("source = cb | stats sum(ResolutionWidth)") is not None

    def test_a_date_span_has_no_sql_spelling(self):
        # Solr SQL rejects DATE_TRUNC over a date column, so there is nothing faithful to emit.
        assert translate_ppl_to_sql(
            "source = big5 | stats count() by span(`@timestamp`, 1d)") is None

    def test_a_computed_case_has_no_sql_spelling(self):
        assert translate_ppl_to_sql(
            "source = big5 | eval b = case(`metrics.size` < 10, 'r1') | stats count() by b") is None

    def test_an_unhandled_operator_is_reported_rather_than_dropped(self):
        # Silently ignoring a stage would emit a statement measuring something else.
        assert translate_ppl_to_sql("source = big5 | dedup `process.name` | head 10") is None

    def test_a_query_with_no_source_clause_is_refused(self):
        assert translate_ppl_to_sql("search index=big5 | head 10") is None


class TestFacetBodyForPipedQuery:
    """
    Shapes the SQL surface cannot express but the JSON Facet API can.

    ⭐ The point of this surface: Solr SQL answers a bare null to sum(ResolutionWidth+1), which reads as an
    empty result. Its facet spelling is sum(sum(ResolutionWidth,1)) and it is exact — measured, Solr
    answers 2,268,791,624 where upstream answers 2,147,483,647 (Integer.MAX_VALUE, because upstream clamps
    at 32 bits) and the true total counted over the corpus is 2,268,791,624.
    """

    def test_an_aggregate_over_a_computed_expression_becomes_a_facet_function(self):
        body = facet_body_for_piped_query(
            "source = cb | stats sum(ResolutionWidth), sum(ResolutionWidth+1)")
        assert body["facet"]["agg_1"] == "sum(ResolutionWidth)"
        assert body["facet"]["agg_2"] == "sum(sum(ResolutionWidth,1))"
        assert body["limit"] == 0

    def test_each_arithmetic_operator_has_its_function(self):
        for expression, expected in (("ClientIP-1", "sub(ClientIP,1)"),
                                     ("A*2", "product(A,2)"),
                                     ("A/2", "div(A,2)")):
            body = facet_body_for_piped_query("source = cb | stats sum(%s)" % expression)
            assert body["facet"]["agg_1"] == "sum(%s)" % expected

    def test_a_grouping_stats_is_left_to_the_sql_surface(self):
        # A grouping by fields is a terms facet, which the SQL translator already carries; answering here
        # too would give two spellings of the same operation.
        assert facet_body_for_piped_query("source = cb | stats count() by URL") is None

    def test_a_date_format_eval_becomes_a_range_facet_at_the_named_resolution(self):
        # '%Y-%m-%d %H:%i:00' names minutes, so the gap is a minute. Measured against both engines: 1,436
        # non-empty minute buckets here, and every one of upstream's 10 rows agrees.
        body = facet_body_for_piped_query(
            "source = cb | where CounterID = 62 and EventDate >= '2013-07-14' "
            "and EventDate <= '2013-07-15' | eval M = date_format(EventTime, '%Y-%m-%d %H:%i:00') "
            "| stats count() as PageViews by M | sort M | head 10")
        facet = body["facet"]["M"]
        assert facet["type"] == "range"
        assert facet["field"] == "EventTime"
        assert facet["gap"] == "+1MINUTE"
        assert facet["mincount"] == 1
        assert facet["facet"]["PageViews"] == "countvals(id)"

    def test_the_range_facet_takes_its_bounds_from_the_query_filter(self):
        # Solr refuses a range facet with no start; a guessed bound that misses the data produces an empty
        # facet, which reads as a working operation.
        body = facet_body_for_piped_query(
            "source = cb | where EventDate >= '2013-07-14' and EventDate <= '2013-07-15' "
            "| eval M = date_format(EventTime, '%Y-%m-%d %H:00:00') | stats count() by M")
        assert body["facet"]["M"]["start"] == "2013-07-14T00:00:00Z"
        assert body["facet"]["M"]["gap"] == "+1HOUR"

    def test_the_filter_is_a_solr_query_not_a_sql_predicate(self):
        # ⚠️ An earlier version of this test allowed an empty body, so it passed while the filter still
        # carried `CounterID = 62 and …` — a predicate Solr does not understand. It asserts the clauses now.
        body = facet_body_for_piped_query(
            "source = cb | where CounterID = 62 and EventDate >= '2013-07-14' "
            "and EventDate <= '2013-07-15' "
            "| eval M = date_format(EventTime, '%Y-%m-%d %H:%i:00') | stats count() by M")
        assert body is not None
        assert "CounterID:62" in body["filter"]
        assert all(" = " not in clause for clause in body["filter"])
        assert any(clause.startswith("EventDate:[") for clause in body["filter"])

    def test_an_ungrouped_aggregation_needs_no_filter_key(self):
        body = facet_body_for_piped_query("source = cb | stats sum(ResolutionWidth+1)")
        assert "filter" not in body

    def test_an_eval_that_is_not_a_date_format_is_refused(self):
        # A computed value has no facet spelling; emitting the rest would measure a different query. Two
        # forms, because the first alone left the refusal to the *grouping* check rather than to the eval
        # one, so a mutation removing the eval guard still passed.
        assert facet_body_for_piped_query(
            "source = cb | eval const = 1 | stats count() by const") is None
        assert facet_body_for_piped_query(
            "source = cb | eval const = 1 | stats count()") is None
        assert facet_body_for_piped_query(
            "source = cb | eval m = extract(minute from EventTime) | stats count()") is None

    def test_a_query_with_no_stats_is_refused(self):
        assert facet_body_for_piped_query("source = cb | head 10") is None
