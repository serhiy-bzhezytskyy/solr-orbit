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

"""
Piped Processing Language to Solr SQL translation.

OpenSearch answers a piped query at ``POST /_plugins/_ppl``; Solr's equivalent surface is its SQL
module at ``/solr/<collection>/sql``. Both compile a declarative statement down to the engine's own
aggregation and search primitives, so a piped query becomes a SELECT rather than a request body.

Two properties of the Solr side shape this translation, both measured rather than assumed:

* ``/sql`` reads its statement from a **request parameter**. A JSON body carrying ``stmt`` is
  answered with "stmt parameter cannot be null"; the same content form-encoded works. The emitted
  operation therefore sets ``form``, which the raw-request runner honours.
* ``/sql`` reports a failure **inside an HTTP 200**, as an ``EXCEPTION`` entry in ``result-set``.
  A caller that checks only the status code reads a failure as a success.

Not every piped query has a SQL spelling. Solr's SQL layer rejects ``DATE_TRUNC`` over a date column
("No match found for function signature DATE_TRUNC(<CHARACTER>, <JavaType(class java.util.Date)>)")
and rejects ``CASE`` in a GROUP BY, so a query bucketing by ``span()`` or by ``case()`` has no
translation on this surface. Those are reported rather than mistranslated: a query that silently
measures something else is worse than one that is absent, and the same aggregations are already
covered by the JSON operations, which reach them through a Solr facet.
"""

import logging
import re

logger = logging.getLogger(__name__)

# A piped query names its fields in OpenSearch's dotted spelling, optionally backquoted; Solr's are
# the flattened ones the schema generator emits.
_BACKQUOTED = re.compile(r"`([^`]+)`")

# Operators that carry no SQL equivalent on this surface. Bucketing by a date span or by a computed
# case is the whole point of the query that uses one, so there is nothing partial to emit.
UNTRANSLATABLE = ("span(", "case(")

# An option a stats stage may carry before its aggregations, optionally wrapped in a masked Jinja
# conditional. `\x00J\d+\x00` is the mask the Jinja pass leaves behind.
_STATS_OPTION = re.compile(
    r"(?:\x00J\d+\x00\s*)*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:true|false)\s*(?:\x00J\d+\x00\s*)*",
    re.IGNORECASE)


def _quote_identifier(name):
    """Render a field name as Solr SQL refers to it.

    A flattened name needs no quoting except when it starts with a character SQL will not accept in
    a bare identifier — ``@timestamp`` is the one every workload carries.
    """
    flattened = name.replace(".", "_")
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", flattened):
        return flattened
    return "`%s`" % flattened


def _translate_fields(text):
    """Rewrite every backquoted field reference into its Solr spelling."""
    return _BACKQUOTED.sub(lambda match: _quote_identifier(match.group(1)), text)


# A workload file is a Jinja template, and a Jinja filter is written with the same pipe that separates
# the stages of a piped query: `source = {{index_name | default('big5')}} | head 10`. Splitting on the
# bare pipe cut the expression in half, so every expression is masked before the split and put back
# after it.
_JINJA_EXPRESSION = re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.DOTALL)
_MASK = "\x00J%d\x00"


def _mask_jinja(text):
    """Replace each Jinja expression with a mask that carries no pipe."""
    masked = []

    def replace(match):
        masked.append(match.group(0))
        return _MASK % (len(masked) - 1)

    return _JINJA_EXPRESSION.sub(replace, text), masked


def _unmask_jinja(text, masked):
    """Put the Jinja expressions back."""
    for index, expression in enumerate(masked):
        text = text.replace(_MASK % index, expression)
    return text


def _split_stages(query):
    """Split a piped query into its source clause and the stages after it.

    Splitting on the pipe is safe here because a pipe never appears inside a literal in the piped
    queries the upstream workloads carry; a query where it did would need a real tokenizer.
    """
    parts = [part.strip() for part in query.split("|")]
    return parts[0], parts[1:]


def _parse_source(clause):
    """Read the source clause, which names the index and may carry a leading search expression.

    ``source = big5 match(`process.name`, 'kernel')`` filters as well as naming the index: the
    expression after the index name is a predicate, and dropping it would turn a filtered query into
    a full scan measuring something else.
    """
    body = clause[len("source"):].lstrip()
    if body.startswith("="):
        body = body[1:].lstrip()
    match = re.match(r"^([A-Za-z0-9_.\-*\x00]+)\s*(.*)$", body, re.DOTALL)
    if not match:
        return body, ""
    return match.group(1), match.group(2).strip()


def _full_text_value(literal):
    """Render a full-text search value the way Solr SQL reads it as a set of terms.

    Solr SQL passes the right-hand side of an equality on an analysed field to the query parser, and
    a bare multi-word value arrives as a phrase: ``message = 'monkey jackal bear'`` matched 0 documents
    where the piped query matched, while the same words parenthesised matched. The parentheses are what
    make it a boolean query over the terms, which is what a piped ``match``/``query_string`` means.
    """
    if len(literal) >= 2 and literal[0] == "'" and literal[-1] == "'":
        inner = literal[1:-1]
        if " " in inner.strip() and not inner.strip().startswith("("):
            return "'(%s)'" % inner
    return literal


def _translate_predicate(expression):
    """Translate a piped predicate into a SQL one.

    The piped forms in use are comparisons, ``and``/``or``/parentheses, and two search functions:
    ``match(field, 'text')`` and ``query_string(['field'], 'text')``. Both reduce to equality against
    the analysed field, which is how Solr SQL reaches the same Lucene query — a full-text predicate
    on an analysed field is a match, not a term comparison.
    """
    text = _translate_fields(expression)

    def _match(match):
        field, value = match.group(1).strip(), match.group(2).strip()
        return "%s = %s" % (_quote_identifier(field.strip("`")), _full_text_value(value))

    text = re.sub(r"match\(\s*([^,()]+?)\s*,\s*('(?:[^']|'')*')\s*\)", _match, text)

    def _query_string(match):
        fields = [f.strip().strip("`").strip("'\"")
                  for f in match.group(1).split(",") if f.strip()]
        value = _full_text_value(match.group(2).strip())
        clauses = ["%s = %s" % (_quote_identifier(f), value) for f in fields]
        return "(%s)" % " or ".join(clauses) if len(clauses) > 1 else clauses[0]

    text = re.sub(r"query_string\(\s*\[([^\]]*)\]\s*,\s*('(?:[^']|'')*')\s*\)",
                  _query_string, text)

    # A piped query writes a timestamp as 'YYYY-MM-DD HH:MM:SS'; Solr's date field parses the ISO
    # spelling, and a space where the 'T' belongs is rejected.
    text = re.sub(r"'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})'", r"'\1T\2Z'", text)
    return text


def _translate_stats(clause):
    """Translate a ``stats`` stage into a SELECT list, a GROUP BY, and the aggregation aliases.

    Returns (select_expressions, group_by_expressions, aliases). ``count()`` becomes ``count(*)``,
    ``dc()`` (distinct count) becomes ``count(distinct …)``, and ``by a, b`` becomes the grouping
    keys, which must also appear in the SELECT list for the result rows to carry them.

    A later stage may sort by the aggregation, which a piped query names by its own spelling —
    ``sort - `count()```. SQL has no such column ("Column 'count()' not found in any table"), so every
    aggregation is given an alias and the mapping is returned for the sort stage to resolve.
    """
    body = clause[len("stats"):].strip()
    group_by = []
    lowered = body.lower()
    marker = lowered.rfind(" by ")
    if marker != -1:
        group_by = [g.strip() for g in body[marker + 4:].split(",") if g.strip()]
        body = body[:marker].strip()

    # A stats stage may carry an option before its aggregations — upstream writes
    # `stats {% if … %}bucket_nullable = false {% endif %}count()` — which is a directive to the piped
    # engine about empty buckets, not a value SQL can select. Left in place it glued itself to the
    # aggregation, so `count()` no longer matched and Solr answered "No match found for function
    # signature". SQL has no equivalent knob, and a Solr facet reports the same non-empty buckets.
    body = _STATS_OPTION.sub("", body).strip()

    select, aliases = [], {}
    for index, aggregation in enumerate(_split_top_level(body)):
        aggregation = aggregation.strip()
        piped_name = aggregation
        alias = None
        as_marker = re.search(r"\s+as\s+(.+)$", aggregation, re.IGNORECASE)
        if as_marker:
            alias = as_marker.group(1).strip().strip("`")
            aggregation = aggregation[:as_marker.start()].strip()
            piped_name = aggregation
        expression = re.sub(r"^count\(\s*\)$", "count(*)", aggregation, flags=re.IGNORECASE)
        expression = re.sub(r"^dc\(\s*(.+?)\s*\)$", r"count(distinct \1)", expression,
                            flags=re.IGNORECASE)
        expression = _translate_fields(expression)
        if alias is None:
            alias = "agg_%d" % (index + 1)
        select.append("%s as %s" % (expression, _quote_identifier(alias)))
        aliases[piped_name] = alias

    return select, [_translate_fields(g) for g in group_by], aliases


def _split_top_level(text):
    """Split on commas that are not inside parentheses."""
    parts, depth, current = [], 0, []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return [p for p in parts if p.strip()]


def _translate_sort(clause, aliases=None):
    """Translate a ``sort`` stage. A leading ``+``/``-`` is the direction; SQL states it as a word.

    ``aliases`` maps a piped aggregation spelling to the alias the SELECT list gave it, so a sort by
    the aggregation resolves to a column that exists.
    """
    body = clause[len("sort"):].strip()
    keys = []
    for key in _split_top_level(body):
        key = key.strip()
        direction = "asc"
        if key.startswith("-"):
            direction, key = "desc", key[1:].strip()
        elif key.startswith("+"):
            key = key[1:].strip()
        bare = key.strip("`")
        if aliases and bare in aliases:
            keys.append("%s %s" % (_quote_identifier(aliases[bare]), direction))
        else:
            keys.append("%s %s" % (_translate_fields(key), direction))
    return keys


def source_index(query):
    """Return the index the query's source clause names, or None.

    The name may be a Jinja expression, which is the point: a workload lets the index be overridden at
    load time, and substituting the collection the conversion happened to be given would pin it.
    """
    masked, tokens = _mask_jinja(query)
    source, _ = _split_stages(masked)
    if not source.lower().startswith("source"):
        return None
    index, _ = _parse_source(source)
    return _unmask_jinja(index, tokens) or None


def translate_ppl_to_sql(query, collection=None):
    """Translate a piped query into a Solr SQL statement.

    Returns the statement, or None when the query uses an operator Solr SQL has no spelling for.
    ``collection`` overrides the index named in the source clause, for a port whose collection is
    named differently from the upstream index.
    """
    for operator in UNTRANSLATABLE:
        if operator in query:
            logger.info(
                "Piped query uses %s, which Solr SQL does not accept over a date column or in a "
                "GROUP BY; leaving the operation untranslated rather than changing what it measures.",
                operator.rstrip("("),
            )
            return None

    query, masked = _mask_jinja(query)
    source, stages = _split_stages(query)
    if not source.lower().startswith("source"):
        logger.info("Piped query does not begin with a source clause: %s", query[:80])
        return None

    index, leading_predicate = _parse_source(source)
    table = collection or index

    where, select, group_by, order_by, limit = [], [], [], [], None
    aliases = {}
    if leading_predicate:
        where.append(_translate_predicate(leading_predicate))

    for stage in stages:
        verb = stage.split()[0].lower() if stage.split() else ""
        if verb == "where":
            where.append(_translate_predicate(stage[len("where"):].strip()))
        elif verb == "stats":
            select, group_by, aliases = _translate_stats(stage)
        elif verb == "sort":
            order_by = _translate_sort(stage, aliases)
        elif verb == "head":
            rest = stage[len("head"):].strip()
            limit = int(rest) if rest.isdigit() else None
        else:
            logger.info("Piped query uses an unhandled operator '%s': %s", verb, stage[:60])
            return None

    # A statement that aggregates selects its grouping keys alongside its aggregations; one that does
    # not selects the identifier, since a piped query with no stats returns whole rows and Solr SQL
    # has no `select *`.
    if select:
        projection = ", ".join(group_by + select)
    else:
        projection = "id"

    statement = "select %s from %s" % (projection, table)
    if where:
        statement += " where %s" % " and ".join("(%s)" % w for w in where)
    if group_by:
        statement += " group by %s" % ", ".join(group_by)
    if order_by:
        statement += " order by %s" % ", ".join(order_by)
    # Solr SQL requires a limit on an unsorted select; a piped query without `head` still returns a
    # bounded page upstream, so the default matches the piped default rather than being unbounded.
    statement += " limit %d" % (limit if limit is not None else 10)
    return _unmask_jinja(statement, masked)
