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

from solrorbit.conversion.ppl import translate_ppl_to_sql


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
        assert statement == ("select SearchPhrase from cb where (SearchPhrase != '') limit 10")

    def test_a_projection_also_selects_what_the_sort_needs(self):
        # Solr SQL answers "Column 'EventTime' not found in any table" for an ORDER BY over a column
        # the SELECT list omits, where the piped form sorts on a field it does not return.
        statement = translate_ppl_to_sql(
            "source = cb | sort EventTime | fields SearchPhrase | head 10")
        assert statement == "select SearchPhrase, EventTime from cb order by EventTime asc limit 10"

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
