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
OpenSearch Query DSL to Solr Query Syntax Translation

This module handles translation of OpenSearch Query DSL (JSON-based query language)
to Solr's Lucene query syntax.

IMPORTANT: This module should ONLY be used when converting OpenSearch workloads.
Native Solr workloads should not go through this translation layer.
"""

import logging
import re
from datetime import datetime

from .field import normalize_field_name

logger = logging.getLogger(__name__)


def translate_opensearch_query(body: dict, nested: dict = None) -> dict:
    """
    Translate an OpenSearch query DSL dict to Solr query parameters.

    Supported patterns:
      - ``match_all``             → ``*:*``
      - ``term``                  → ``field:value``
      - ``terms``                 → ``field:(v1 v2 v3)`` or ``{!terms f=field}v1,v2,...``
      - ``match`` / ``match_phrase`` → ``field:value``
      - ``range``                 → ``field:[lo TO hi]``
      - ``exists``                → ``field:[* TO *]``
      - ``bool`` (must/should/must_not) → recursive translation in ``q``
      - ``bool.filter``           → Solr ``fq`` parameters (supports large term lists)
      - ``ids``                   → ``id:(id1 id2 ...)``

    Falls back to ``*:*`` for unrecognised patterns (logs warning).

    Args:
        body: OpenSearch query body dict with "query" key

    Returns:
        Dict with keys:
          - ``"q"``: Solr query string for the ``q`` parameter
          - ``"fq"``: list of Solr filter query strings for the ``fq`` parameter

    Examples:
        >>> translate_opensearch_query({"query": {"match_all": {}}})
        {'q': '*:*', 'fq': []}
        >>> translate_opensearch_query({"query": {"term": {"country": "US"}}})
        {'q': 'country:US', 'fq': []}
    """
    if not body or not isinstance(body, dict):
        return {"q": "*:*", "fq": []}
    query = body.get("query", {})
    fq_list = []

    # Top-level terms queries can be very large (thousands of values).
    # Route them directly to fq using {!terms f=...} for efficiency.
    if "terms" in query and len(query) == 1:
        fq_str = _translate_node_for_fq(query)
        if fq_str:
            fq_list.append(fq_str)
            return {"q": "*:*", "fq": fq_list}

    q = _translate_query_node(query, fq_list=fq_list, nested=nested)
    return {"q": q, "fq": fq_list}


def extract_sort_parameter(body: dict, nested: dict = None) -> str:
    """
    Extract a Solr sort string from an OpenSearch sort clause.

    Args:
        body: OpenSearch query body dict with optional "sort" key
        nested: Optional dict collecting referenced queries, for a sort over a nested field

    Returns:
        Solr sort parameter string (e.g., "name_raw desc, _score asc")
        or None if no sort clause present

    Examples:
        >>> extract_sort_parameter({"sort": [{"name.raw": "desc"}]})
        'name_raw desc'
    """
    if not isinstance(body, dict) or "sort" not in body:
        return None
    sort_clauses = body["sort"]
    if isinstance(sort_clauses, dict):
        sort_clauses = [sort_clauses]
    solr_sorts = []
    for clause in sort_clauses:
        if isinstance(clause, str):
            # Normalize field name before adding to sort
            field = normalize_field_name(clause.split()[0] if " " in clause else clause)
            suffix = " " + clause.split()[1] if " " in clause else ""
            solr_sorts.append(field + suffix)
        elif isinstance(clause, dict):
            for field, order_info in clause.items():
                if field == "_score":
                    continue
                order = "asc"
                if isinstance(order_info, dict):
                    order = order_info.get("order", "asc")
                elif isinstance(order_info, str):
                    order = order_info
                nested_sort = _nested_sort_expression(field, order_info, nested)
                if nested_sort:
                    solr_sorts.append("%s %s" % (nested_sort, order))
                    continue
                # Normalize field name
                field = normalize_field_name(field)
                solr_sorts.append(f"{field} {order}")
    return ", ".join(solr_sorts) if solr_sorts else None


def _nested_sort_expression(field: str, order_info, nested: dict):
    """The Solr sort expression for a sort over a nested field, or None if this is an ordinary sort.

    ⭐ ``childfield(field, $bjq)`` is the nested ``mode: max`` sort: it reads a child document's value
    for the parent the block-join query in ``$bjq`` produced. Where the block join is put matters —
    with it in ``q`` the sort returned 74 of the sample's 80 matching documents, dropping the 6
    questions with no answers (upstream's own ``nested`` + ``match_all`` returns the same 74, so that
    is a different query, not a Solr limitation). With ``q`` the parent query and the block join in a
    separate ``bjq`` parameter, it returned all 80 in upstream's order.

    ⚠️ The parent's flattened copy — ``field(answers_date,max)`` — is *also* correct, and only if it is
    written: measured on the sample with the copy written it gave upstream's order exactly, and with the
    copy merely declared in the schema it silently returned all 90 documents in id order. The
    block-join form does not depend on that copy existing, so it is the one emitted.
    """
    if not isinstance(order_info, dict):
        return None
    path = (order_info.get("nested") or {}).get("path")
    if not path:
        return None
    if nested is None:
        logger.warning(
            "sort over nested path '%s' cannot be translated here: it needs a block-join query passed "
            "by reference, and no parameter block is available", path)
        return None
    mode = order_info.get("mode", "max")
    if mode not in ("max", "min"):
        # avg/sum/median over a nested leaf have no childfield equivalent. Reported rather than
        # silently sorted by something else — a wrong sort order is not visible in a hit count.
        logger.warning("sort mode '%s' over nested path '%s' has no Solr childfield equivalent",
                       mode, path)
        return None
    prefix = str(path).replace(".", "_")
    leaf = field
    for candidate in (str(path) + ".", prefix + "_"):
        if leaf.startswith(candidate):
            leaf = leaf[len(candidate):]
            break
    child_field = child_field_name(path, str(leaf).replace(".", "_"))
    nested[_NESTED_SORT_PARAM] = "{!parent which='%s:true'}%s:%s" % (
        _PARENT_MARKER_FIELD, _CHILD_PATH_FIELD, prefix)
    # ⚠️ childfield picks the value the block join's scoring produced, which for a `desc` sort over a
    # date is the maximum. `min` is not expressible this way and is reported above.
    return "childfield(%s,$%s)" % (child_field, _NESTED_SORT_PARAM)


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _translate_query_node(node: dict, fq_list: list = None, nested: dict = None) -> str:
    """Recursively translate a single OpenSearch query node to Solr syntax.

    Args:
        node: OpenSearch query node dict
        fq_list: Optional list to collect Solr fq filter strings. When provided,
                 bool.filter clauses are appended here instead of inlined in q.
        nested: Optional dict collecting the referenced queries a ``nested`` clause needs, keyed by
                the parameter name the ``q`` refers to. Solr's block-join parser cannot take its child
                query inline inside a boolean clause, so it arrives by reference.
    """
    if not node or not isinstance(node, dict):
        return "*:*"

    if "match_all" in node:
        return "*:*"

    if "nested" in node:
        return _translate_nested_clause(node["nested"], nested)

    if "match_none" in node:
        return "-*:*"

    if "term" in node:
        for field, value in node["term"].items():
            v = value.get("value", value) if isinstance(value, dict) else value
            field = normalize_field_name(field)
            if v == "":
                # An empty term matches a document whose field is the empty string. `field:` is not a
                # query — Solr answers 'Encountered " ")"' — and the schema-generated configset removes
                # blank values before indexing, mirroring OpenSearch, so no document has one. The query
                # that selects none of them is what this means.
                return "-%s:[* TO *] AND %s:[* TO *]" % (field, field)
            return f"{field}:{_escape_solr_value(v)}"

    if "terms" in node:
        for field, values in node["terms"].items():
            # A serialised query carries `boost` beside the field, and dict order put it first: the loop
            # returned on it and the whole term list was lost to *:*. Only a list of values is a term
            # list, so anything else at this level is metadata.
            if field.startswith("_") or not isinstance(values, (list, tuple)):
                continue
            field = normalize_field_name(field)
            return _translate_terms_clause(field, values)

    if "geo_shape" in node:
        # A geo_shape query states its shape in GeoJSON. The two shapes the workloads use are an envelope,
        # which is a bounding box, and a polygon — so it is rewritten into the query that already carries
        # each, rather than translated a second time.
        for field, conf in node["geo_shape"].items():
            if field == "boost" or not isinstance(conf, dict):
                continue
            shape = conf.get("shape")
            if not isinstance(shape, dict):
                logger.warning("geo_shape on '%s' states no inline shape (an indexed-shape reference "
                              "has no Solr equivalent here)", field)
                return "*:*"
            kind = str(shape.get("type", "")).lower()
            coordinates = shape.get("coordinates")
            if kind == "envelope" and isinstance(coordinates, list) and len(coordinates) == 2:
                # GeoJSON's envelope is [[minLon, maxLat], [maxLon, minLat]] — the same two corners a
                # geo_bounding_box names top_left and bottom_right.
                return _translate_query_node(
                    {"geo_bounding_box": {field: {"top_left": coordinates[0],
                                                 "bottom_right": coordinates[1]}}}, fq_list=fq_list)
            if kind == "polygon" and isinstance(coordinates, list) and coordinates:
                # A GeoJSON polygon's first ring is its outer boundary; any further rings are holes, which
                # a conjunction of half-planes cannot express.
                if len(coordinates) > 1:
                    logger.warning("geo_shape polygon on '%s' has %d holes, which a half-plane "
                                  "conjunction cannot express", field, len(coordinates) - 1)
                    return "*:*"
                return _translate_query_node(
                    {"geo_polygon": {field: {"points": coordinates[0]}}}, fq_list=fq_list)
            logger.warning("geo_shape on '%s' states shape type '%s', which this converter does not "
                          "carry", field, kind or "(none)")
            return "*:*"

    if "geo_bounding_box" in node:
        # A range on an exact point field: measured on 2,000 points whose membership was computed
        # independently, this answers 380 against the exact 380, where the same box against an RPT index
        # answered 388 — an RPT query is approximate, bounded by the maxDistErr set at index time.
        for field, box in node["geo_bounding_box"].items():
            if field == "boost" or not isinstance(box, dict):
                continue
            top_left, bottom_right = box.get("top_left"), box.get("bottom_right")
            corners = _corner_pair(top_left, bottom_right)
            if corners is None:
                logger.warning("geo_bounding_box on '%s' has corners this converter cannot read: %s",
                              field, box)
                return "*:*"
            (min_lat, min_lon), (max_lat, max_lon) = corners
            field = normalize_field_name(field)
            return "%s:[%s,%s TO %s,%s]" % (field, min_lat, min_lon, max_lat, max_lon)

    if "geo_distance" in node:
        # {!geofilt} over the same field: measured 39 against the exact 39, where an RPT index answered 41.
        conf = node["geo_distance"]
        distance = conf.get("distance")
        for field, point in conf.items():
            if field in ("distance", "boost", "distance_type", "validation_method"):
                continue
            location = _lat_lon(point)
            if location is None or distance is None:
                logger.warning("geo_distance on '%s' has a form this converter cannot read: %s",
                              field, conf)
                return "*:*"
            kilometres = _distance_in_kilometres(distance)
            if kilometres is None:
                logger.warning("geo_distance states a distance with no unit this converter knows: %s",
                              distance)
                return "*:*"
            return "{!geofilt sfield=%s pt=%s,%s d=%s}" % (
                normalize_field_name(field), location[0], location[1], kilometres)

    if "geo_polygon" in node:
        # ⭐ Solr cannot take a WKT POLYGON without JTS — "Unsupported shape of this SpatialContext. Try
        # JTS or Geo3D" — and this build ships 14 modules, spatial-extras not among them. But a *convex*
        # ring is the intersection of half-planes, and that needs no shape library at all: a point is
        # inside when the cross product against every edge has the same sign, which {!frange} over the
        # lat/lon components states directly. Measured: 335 against the exact 335.
        for field, conf in node["geo_polygon"].items():
            if field == "boost" or not isinstance(conf, dict):
                continue
            points = [_lat_lon(point) for point in conf.get("points", [])]
            if not points or any(p is None for p in points):
                logger.warning("geo_polygon on '%s' has points this converter cannot read", field)
                return "*:*"
            # A closed ring repeats its first point; the half-plane form does not want the repeat.
            if len(points) > 1 and points[0] == points[-1]:
                points = points[:-1]
            if len(points) < 3:
                logger.warning("geo_polygon on '%s' has fewer than three distinct points", field)
                return "*:*"
            if not _is_convex(points):
                # A concave ring is not an intersection of half-planes, and emitting one anyway would
                # match a larger area than the query states.
                logger.warning(
                    "geo_polygon on '%s' is not convex, so it is not an intersection of half-planes; "
                    "Solr needs JTS for a general polygon and this build has no spatial-extras module.",
                    field)
                return "*:*"
            base = normalize_field_name(field)
            clauses = []
            for index in range(len(points)):
                lat1, lon1 = points[index]
                lat2, lon2 = points[(index + 1) % len(points)]
                clauses.append(
                    "{!frange l=0}sub(product(%s,sub(%s_lat,%s)),product(%s,sub(%s_lon,%s)))" % (
                        _number(lon2 - lon1), base, _number(lat1),
                        _number(lat2 - lat1), base, _number(lon1)))
            if fq_list is not None:
                fq_list.extend(clauses)
                return "*:*"
            return " AND ".join("(%s)" % clause for clause in clauses)

    if "wildcard" in node or "prefix" in node:
        # Both are Lucene's own query forms, so Solr states them directly. Untranslated they fell
        # through to *:* and the operation matched the whole corpus.
        kind = "wildcard" if "wildcard" in node else "prefix"
        for field, value in node[kind].items():
            pattern = value.get("value", value.get("wildcard", value)) if isinstance(value, dict) \
                else value
            if pattern is None:
                continue
            field = normalize_field_name(field)
            # A wildcard states its own metacharacters; a prefix implies a trailing one. Neither is
            # escaped, since the pattern is the query.
            return "%s:%s" % (field, pattern if kind == "wildcard" else "%s*" % pattern)

    if "query_string" in node or "simple_query_string" in node:
        # A query string is already Lucene syntax on the OpenSearch side, and Solr's default parser
        # reads the same syntax — what differs is the field spelling and how a default field is named.
        # Untranslated this fell through to *:* and the operation matched the whole corpus: 3,482,624
        # documents where the query selects 298,029.
        sub = node.get("query_string") or node.get("simple_query_string")
        if isinstance(sub, dict) and isinstance(sub.get("query"), str):
            query = sub["query"]
            fields = sub.get("fields")
            # `message: monkey jackal bear` names its field inline and then lists several terms. Solr
            # binds a bare field reference to the *next* term only, and answers "no field name
            # specified in query and no default specified via 'df' param" for the rest, so the terms
            # that follow are grouped. Both engines then report 298,029.
            inline = re.match(r"^\s*([A-Za-z_][\w.]*)\s*:\s*(.+)$", query, re.DOTALL)
            if inline and not re.search(r"[():\[\]]", inline.group(2)):
                terms = inline.group(2).strip()
                field = normalize_field_name(inline.group(1))
                return "%s:(%s)" % (field, terms) if " " in terms else "%s:%s" % (field, terms)
            query = re.sub(r"([A-Za-z_][\w.]*)\s*:\s*",
                           lambda m: "%s:" % normalize_field_name(m.group(1)), query)
            if isinstance(fields, list) and fields:
                # Several fields with no inline field reference: Solr states that with edismax's qf.
                if ":" not in query:
                    names = " ".join(normalize_field_name(f) for f in fields)
                    return "{!edismax qf=\"%s\"}%s" % (names, query)
            default_field = sub.get("default_field")
            if default_field and ":" not in query:
                return "%s:(%s)" % (normalize_field_name(default_field), query)
            return query
        logger.warning(
            "A query_string node carries no query string: %s. Falling back to q=*:*.", node)
        return "*:*"

    if "match" in node or "match_phrase" in node:
        sub = node.get("match") or node.get("match_phrase")
        if isinstance(sub, dict):
            for field, value in sub.items():
                # Skip empty field names or metadata fields
                if not field or field.startswith("_"):
                    continue
                v = value.get("query", value) if isinstance(value, dict) else value
                field = normalize_field_name(field)
                # For phrase queries, wrap in quotes if not already
                if "match_phrase" in node and not (isinstance(v, str) and v.startswith('"')):
                    return f'{field}:"{_escape_solr_phrase(v)}"'
                return f"{field}:{_escape_solr_value(v)}"
        # If sub is not a dict or has no valid fields, fall back
        logger.warning(
            "match/match_phrase query has invalid structure: %s. Using *:*",
            sub
        )
        return "*:*"

    if "range" in node:
        for field, bounds in node["range"].items():
            field = normalize_field_name(field)
            # `from`/`to` with `include_lower`/`include_upper` is the same range in the spelling
            # OpenSearch's own query builder serialises. Reading only gte/lte lost both bounds and left
            # `field:[* TO *]`, which matches every document that has the field — clickbench's q44
            # reported 1,498,137 where the query selects 663.
            lo = bounds.get("gte", bounds.get("gt", bounds.get("from")))
            hi = bounds.get("lte", bounds.get("lt", bounds.get("to")))
            lower_inclusive = "gt" not in bounds and bounds.get("include_lower", True) is not False
            upper_inclusive = "lt" not in bounds and bounds.get("include_upper", True) is not False
            lo = "*" if lo is None else lo
            hi = "*" if hi is None else hi
            # Convert dates if format is specified (common for date fields)
            os_format = bounds.get("format")
            lo = _convert_date_to_solr_format(lo, os_format)
            hi = _convert_date_to_solr_format(hi, os_format)
            # An exclusive bound is a brace in Solr's range syntax, so `gt`/`lt` and
            # include_lower/include_upper=false do not silently widen the range by one value.
            open_bracket = "[" if lower_inclusive else "{"
            close_bracket = "]" if upper_inclusive else "}"
            return f"{field}:{open_bracket}{lo} TO {hi}{close_bracket}"

    if "exists" in node:
        field = node["exists"].get("field", "*")
        field = normalize_field_name(field)
        return f"{field}:[* TO *]"

    if "ids" in node:
        values = node["ids"].get("values", [])
        if values:
            escaped = " ".join(_escape_solr_value(v) for v in values)
            return f"id:({escaped})"
        return "*:*"

    if "bool" in node:
        bool_q = node["bool"]
        parts = []

        def _add_to_q(clauses, prefix):
            if not clauses:
                return
            if isinstance(clauses, dict):
                clauses = [clauses]
            for clause in clauses:
                sub = _translate_query_node(clause, fq_list=fq_list, nested=nested)
                if sub and sub != "*:*":
                    # ⛔ A purely negative group matches nothing in Lucene: `+(-(URL:*x*))` returned 0
                    # where `-(URL:*x*)` beside it returned 564. A nested must_not therefore needs
                    # something for the negation to subtract from.
                    if sub.lstrip().startswith("-"):
                        sub = "*:* %s" % sub
                    parts.append(f"{prefix}({sub})")

        def _add_to_fq(clauses):
            """Translate bool.filter clauses to Solr fq parameters."""
            if not clauses:
                return
            if isinstance(clauses, dict):
                clauses = [clauses]
            for clause in clauses:
                fq_str = _translate_node_for_fq(clause)
                if fq_str:
                    fq_list.append(fq_str)

        _add_to_q(bool_q.get("must"), "+")
        _add_to_q(bool_q.get("must_not"), "-")

        # bool.filter → Solr fq when fq_list is available; otherwise inline in q
        if fq_list is not None:
            _add_to_fq(bool_q.get("filter"))
        else:
            _add_to_q(bool_q.get("filter"), "+")

        shoulds = bool_q.get("should", [])
        if isinstance(shoulds, dict):
            shoulds = [shoulds]
        should_parts = [_translate_query_node(s, fq_list=fq_list, nested=nested) for s in shoulds]
        should_parts = [s for s in should_parts if s and s != "*:*"]
        if should_parts:
            parts.append("(" + " ".join(should_parts) + ")")

        return " ".join(parts) if parts else "*:*"

    # Unknown / untranslatable query node
    logger.warning(
        "Cannot translate OpenSearch query type '%s' to Solr syntax. "
        "Falling back to q=*:* (results may not match workload intent). "
        "Consider rewriting this operation as a native Solr workload task.",
        list(node.keys()),
    )
    return "*:*"


# The fields the runner writes to mark a block of parent and child documents. Named the same way in
# the schema generator and the runner; a block join needs a query for each side and neither is
# expressible as a negation.
_CHILD_PATH_FIELD = "_nested_path_"
_PARENT_MARKER_FIELD = "_nested_parent_"

# The prefix of the referenced-parameter names a nested clause's child query is passed under.
_NESTED_PARAM_PREFIX = "nq"

# ⭐ The filter restricting a query to the documents upstream would have counted. A nested field's
# objects are separate documents in Solr and share the collection with their parents, so a query that
# does not exclude them counts both: measured on the 1,000-document sample, match_all answered 1,000
# upstream and 2,977 in Solr — 1,000 questions plus 1,977 answers.
#
# ⚠️ The parent marker is *not* the right filter for this. It marks a document that has children, and
# 40 of the sample's 1,000 questions have none — filtering on it answered 960. "Has no nested path" is
# the property that matches upstream's document set, and it answered 1,000 exactly. Measured on the
# other three operations, adding it changes nothing: term 90, nested 3, and the nested sort's 90 in the
# same order.
_TOP_LEVEL_ONLY = "-%s:[* TO *]" % _CHILD_PATH_FIELD

# Whether the workload being converted declares a nested field at all, set by the converter from the
# mapping.
#
# ⚠️ The scope filter cannot be emitted unconditionally: a collection with no nested field has no
# _nested_path_ to negate, and Solr answers `undefined field: "_nested_path_"` and refuses the query
# outright — measured against a collection from another workload. So it is emitted only where the
# mapping declares the field, which is also the only place child documents exist.
_HAS_NESTED_FIELDS = False


def set_nested_fields_present(present: bool):
    """Say whether the workload under conversion declares a nested field.

    The scope filter belongs to every operation of such a workload, including those with no nested
    clause of their own — match_all is exactly the operation that needed it — so it cannot be derived
    from the operation body and has to come from the mapping.
    """
    global _HAS_NESTED_FIELDS  # noqa: PLW0603 — same convention as the converter's target collection
    _HAS_NESTED_FIELDS = bool(present)

# The parameter a nested sort's block-join query is passed under.
_NESTED_SORT_PARAM = "bjq"


def _translate_nested_clause(conf: dict, nested: dict) -> str:
    """Translate one ``nested`` clause into a Solr block-join query over child documents.

    Upstream's ``{"nested": {"path": "answers", "query": ...}}`` selects parents having at least one
    *object* satisfying the query. Solr's equivalent is ``{!parent which=<parents>}<child query>``,
    where the child query carries both the path and the clause — the path alone is not enough:
    measured on the 1,000-document sample, a childFilter of ``_nested_path_:answers`` returned a 2015
    answer that the query excludes.

    ⚠️ The child query cannot be written inline. Three spellings were measured against upstream's 1 hit
    on the sample:

    * ``+(tag:vb6) +({!parent which=...}+_nested_path_:answers +answer_date:[* TO D])`` → **0**. The
      leading local-params parser consumes the whole clause and the boolean structure is lost.
    * ``{!parent which=...}...`` as the entire ``q`` → **80**, i.e. the tag clause vanished.
    * ``_query_:"{!parent which=... v=$nq0}"`` with the child query in a referenced parameter → **1**,
      agreeing with upstream.

    ⇒ So the child query is collected into *nested* and referenced. An inline ``v='...'`` form is not a
    substitute: a child value containing a single quote — ``answer_user:"O'Brien"`` — is a syntax error
    ("Missing end quote for string at pos 40") whether the quote is escaped or not, while the referenced
    form parses it.
    """
    if not isinstance(conf, dict):
        return "*:*"
    path = conf.get("path")
    inner = conf.get("query")
    if not path or not isinstance(inner, dict):
        logger.warning("nested clause states no path or query — cannot translate to a block join")
        return "*:*"
    if nested is None:
        # Nothing to carry the referenced parameter, so there is nowhere to put the child query. Saying
        # so is the point: a fallback to *:* here would widen the clause to every document.
        logger.warning(
            "nested clause on path '%s' cannot be translated in this context: Solr's block-join parser "
            "needs its child query passed by reference, and no parameter block is available", path)
        return "*:*"

    child_q = _translate_query_node(_rename_to_child_fields(inner, path),
                                    fq_list=None, nested=nested)
    path_field = str(path).replace(".", "_")
    param = "%s%d" % (_NESTED_PARAM_PREFIX, len(nested))
    nested[param] = "+%s:%s +(%s)" % (_CHILD_PATH_FIELD, path_field, child_q)
    return '_query_:"{!parent which=\'%s:true\' v=$%s}"' % (_PARENT_MARKER_FIELD, param)


def child_field_name(path: str, leaf: str) -> str:
    """The name a child document's field takes: the singular of the path, then the leaf.

    ⚠️ Two different fields carry a nested leaf, and they are not interchangeable. The child documents
    hold ``answer_date`` — the singular — and the parent holds a flattened multi-valued copy named
    ``answers_date``. A block-join child query must name the former; a `mode: max` sort over the parent
    may name either. Using the flattened name inside a child query asks for a field no child document
    has, and Solr answers 0 rather than failing.
    """
    prefix = str(path).replace(".", "_")
    singular = prefix[:-1] if prefix.endswith("s") and len(prefix) > 1 else "%s_item" % prefix
    return "%s_%s" % (singular, leaf)


def _rename_to_child_fields(node, path):
    """Rewrite every field name in a nested clause's query to the child document's spelling.

    Upstream writes a nested leaf fully qualified — ``answers.date`` — and normalisation alone turns
    that into ``answers_date``, which is the *parent's* flattened copy, not the child's field. The
    child query would then select nothing.
    """
    prefix = str(path) + "."
    prefix_underscored = str(path).replace(".", "_") + "_"

    def rename(name):
        if not isinstance(name, str):
            return name
        for candidate in (prefix, prefix_underscored):
            if name.startswith(candidate):
                return child_field_name(path, name[len(candidate):].replace(".", "_"))
        return name

    def walk(value):
        if isinstance(value, dict):
            out = {}
            for key, sub in value.items():
                # Only a field position is renamed. A leaf-clause key ("range", "gte", "boost") is not
                # a field name, and the field sits one level below the clause name.
                if key in _FIELD_KEYED_CLAUSES and isinstance(sub, dict):
                    out[key] = {rename(field): walk(spec) if isinstance(spec, (dict, list)) else spec
                                for field, spec in sub.items()}
                elif key == "field" and isinstance(sub, str):
                    # ⚠️ An aggregation names its field in a *value*, not a key: a date_histogram is
                    # {"field": "answers.date"}. Renaming keys alone left the facet computed over the
                    # domain of child documents while naming the parent's flattened copy, which no child
                    # document carries — every bucket would have been empty, and an empty facet reads
                    # like a working operation.
                    out[key] = rename(sub)
                else:
                    out[key] = walk(sub)
            return out
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(node)


# The clause types whose immediate keys are field names.
_FIELD_KEYED_CLAUSES = ("term", "terms", "match", "match_phrase", "range", "wildcard", "prefix")


def _translate_terms_clause(field: str, values: list) -> str:
    """
    Translate a terms list to the most efficient Solr syntax.

    - Small lists (≤100 terms): ``field:(v1 v2 ...)``  — standard Lucene OR clause
    - Large lists (>100 terms): ``field:(v1 v2 ...)``  for q context, still works
      but when used as an fq, callers should prefer ``{!terms f=field}v1,v2,...``
    """
    escaped = " ".join(
        f'"{_escape_solr_phrase(v)}"' if " " in str(v) else _escape_solr_value(v)
        for v in values
    )
    return f"{field}:({escaped})"


def _translate_node_for_fq(node: dict) -> str:
    """
    Translate a single query node to a Solr fq string.

    For terms clauses, uses the efficient ``{!terms f=field}v1,v2,...`` syntax
    which Solr handles as a cached bitset — ideal for large term lists in filters.

    For other clause types, delegates to _translate_query_node() without fq_list
    (filter sub-clauses are flattened into a single fq string).
    """
    if not node or not isinstance(node, dict):
        return None

    # terms → {!terms f=field}v1,v2,...  (Solr's efficient bitset filter)
    if "terms" in node:
        for field, values in node["terms"].items():
            # `boost` sits beside the field in a serialised query and is not a term list.
            if field.startswith("_") or not isinstance(values, (list, tuple)):
                continue
            field = normalize_field_name(field)
            if not values:
                return None
            # Join values as comma-separated (Solr {!terms} syntax). A whole JSON number is written
            # without its fraction, as everywhere else: Solr refuses "-1.0" for an integer field.
            joined = ",".join(str(_numeric_literal(v)) for v in values)
            return f"{{!terms f={field}}}{joined}"

    # range, term, exists, match etc. — translate normally
    return _translate_query_node(node, fq_list=None)


def _lat_lon(point):
    """Read a point in any of the spellings OpenSearch accepts, as (lat, lon).

    ⚠️ A two-element array is **[lon, lat]** — GeoJSON order — while a string is "lat,lon". Reading the
    array in the wrong order silently moves the query, which is why every form is handled here rather
    than at each call site.
    """
    if isinstance(point, dict):
        if "lat" in point and "lon" in point:
            return float(point["lat"]), float(point["lon"])
        return None
    if isinstance(point, (list, tuple)) and len(point) == 2:
        return float(point[1]), float(point[0])
    if isinstance(point, str):
        parts = [part.strip() for part in point.split(",")]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None
    return None


def _corner_pair(top_left, bottom_right):
    """Return ((min_lat, min_lon), (max_lat, max_lon)) from a bounding box's two corners."""
    upper, lower = _lat_lon(top_left), _lat_lon(bottom_right)
    if upper is None or lower is None:
        return None
    lats, lons = sorted((upper[0], lower[0])), sorted((upper[1], lower[1]))
    return (_number(lats[0]), _number(lons[0])), (_number(lats[1]), _number(lons[1]))


_DISTANCE_UNITS = {"km": 1.0, "kilometers": 1.0, "kilometres": 1.0,
                   "m": 0.001, "meters": 0.001, "metres": 0.001,
                   "mi": 1.609344, "miles": 1.609344,
                   "yd": 0.0009144, "yards": 0.0009144,
                   "ft": 0.0003048, "feet": 0.0003048,
                   "nmi": 1.852, "nauticalmiles": 1.852}


def _distance_in_kilometres(distance):
    """Convert a distance like "200km" to kilometres, which is what {!geofilt}'s d takes."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$", str(distance))
    if not match:
        return None
    amount, unit = float(match.group(1)), match.group(2).lower()
    if not unit:
        # OpenSearch's default for a bare number is metres.
        return _number(amount * 0.001)
    factor = _DISTANCE_UNITS.get(unit)
    return None if factor is None else _number(amount * factor)


def _is_convex(points):
    """Say whether a ring is convex, so it is the intersection of its edges' half-planes."""
    signs = []
    count = len(points)
    for index in range(count):
        lat1, lon1 = points[index]
        lat2, lon2 = points[(index + 1) % count]
        lat3, lon3 = points[(index + 2) % count]
        cross = (lon2 - lon1) * (lat3 - lat2) - (lat2 - lat1) * (lon3 - lon2)
        if cross:
            signs.append(cross > 0)
    return bool(signs) and (all(signs) or not any(signs))


def _number(value):
    """Render a float without a trailing .0, so a query reads as the workload wrote it."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(round(value, 9)) if isinstance(value, float) else str(value)


def _numeric_literal(value):
    """Render a JSON number the way a whole number should be written.

    OpenSearch serialises every number in a query as a JSON double, so a term list over an integer
    field arrives as ``[-1.0, 6.0]``. It coerces those to the field's type; Solr refuses — "Invalid
    Number: -1.0 for field TraficSourceID" — so a value that is whole is written without its fraction.
    A value that is not whole keeps it, since the field is then not an integer one.
    """
    # No bool guard: a bool is a subclass of int, not of float, so the float branch never sees one and a
    # guard for it would be code no test can reach.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _escape_solr_value(value) -> str:
    """Escape special Lucene/Solr query characters in a field value."""
    value = _numeric_literal(value)
    special = r'+-&&||!(){}[]^"~*?:\/'
    result = []
    for char in str(value):
        if char in special:
            result.append('\\' + char)
        else:
            result.append(char)
    return ''.join(result)


def _escape_solr_phrase(value) -> str:
    """
    Escape a phrase value for Solr phrase queries.

    For phrases, we only need to escape quotes (and backslashes).
    Other special characters are OK within quotes.
    """
    return str(value).replace('\\', '\\\\').replace('"', '\\"')


# The tag a post_filter's filter query carries, so the facets can exclude it by name.
_POST_FILTER_TAG = "postfilter"


def translate_to_solr_json_dsl(body: dict) -> dict:
    """
    Translate an OpenSearch query body to Solr JSON Query DSL format.

    Output is a valid Solr JSON Query DSL body for POSTing to the
    ``/solr/{collection}/query`` endpoint (Mode 2 of SolrSearch runner).
    The ``"query"`` value is always a string, making it Mode 2 compatible.

    Args:
        body: OpenSearch query body dict (may contain "query", "aggs", "size", "sort")

    Returns:
        Solr JSON Query DSL dict with keys:
          - ``query``: Lucene query string
          - ``filter``: list of filter query strings (omitted if empty)
          - ``limit``: number of results (omitted if not specified)
          - ``sort``: sort string (omitted if not specified)
          - ``facet``: Solr JSON Facet API dict (omitted if no aggregations)
    """
    if not body or not isinstance(body, dict):
        return {"query": "*:*"}

    fq_list = []
    # The referenced child queries a nested clause needs. Solr's block-join parser cannot take its
    # child query inline inside a boolean clause, so each one is collected here and passed in a params
    # block beside the query.
    nested = {}
    query_val = body.get("query")
    if isinstance(query_val, dict):
        translated = translate_opensearch_query(body, nested=nested)
        q = translated["q"]
        fq_list = translated["fq"]
    else:
        q = "*:*"

    result = {"query": q}

    # A post_filter narrows the hits *after* the aggregations have been computed, so the facets see
    # the unfiltered set and the hit count sees the filtered one. Dropped entirely, the operation
    # reported 103,349 hits where upstream reports 4,199. Solr's equivalent is a filter tagged and
    # excluded from the facets, which is what {!tag} plus a facet domain excludeTags does.
    post_filter = body.get("post_filter")
    if isinstance(post_filter, dict):
        post_fq = _translate_query_node(post_filter)
        if post_fq and post_fq != "*:*":
            fq_list = list(fq_list) + ["{!tag=%s}%s" % (_POST_FILTER_TAG, post_fq)]

    # Restrict the hits to the documents upstream counts. A child document is a document of its own in
    # Solr, so without this every operation of a nested workload counts the objects too — match_all
    # answered 2,977 against upstream's 1,000 on the sample.
    if _HAS_NESTED_FIELDS:
        fq_list = list(fq_list) + [_TOP_LEVEL_ONLY]

    if fq_list:
        result["filter"] = fq_list

    if "size" in body:
        result["limit"] = body["size"]

    sort_str = extract_sort_parameter(body, nested=nested)
    if sort_str:
        result["sort"] = sort_str

    # An inner_hits block asks for the matching child objects beside each parent. Solr's equivalent is
    # the [child] document transformer, and it needs both the query — not merely "is a child" — and its
    # own fl. Measured on the sample: a childFilter of _nested_path_:answers alone returned 2 children
    # for a document where inner_hits returned 1, including a 2015 answer the query excludes; and
    # without the inner fl the count was right and every child came back as {}.
    child_fl = _inner_hits_field_list(body, nested)
    if child_fl:
        result["fields"] = child_fl

    aggs = body.get("aggs") or body.get("aggregations")
    if aggs and isinstance(aggs, dict):
        facets, agg_domain = _nested_aggregation_domain(aggs)
        if agg_domain:
            # A `nested` aggregation counts *child* documents, so the facet is computed over the
            # children rather than the parents the query selects. Measured: over
            # q=_nested_path_:answers, all 91 monthly buckets are identical to upstream's, sum 1977.
            aggs = facets
        facets = _convert_aggregations_to_facets(aggs, _date_bounds_from_query(body))
        if facets:
            if agg_domain:
                for facet in facets.values():
                    if isinstance(facet, dict):
                        domain = dict(facet.get("domain") or {})
                        domain["query"] = agg_domain
                        facet["domain"] = domain
            # A post_filter is by definition not applied to the aggregations. Each top-level facet
            # excludes it by the tag the filter carries; without this the facets would be computed over
            # the narrowed set, which is the very thing a post_filter exists to avoid.
            if isinstance(post_filter, dict) and "filter" in result:
                for facet in facets.values():
                    if isinstance(facet, dict):
                        domain = dict(facet.get("domain") or {})
                        domain["excludeTags"] = _POST_FILTER_TAG
                        facet["domain"] = domain
            result["facet"] = facets

    # The child queries, referenced by the query, the sort and the [child] transformer alike.
    if nested:
        result["params"] = dict(nested)

    return result


def _nested_aggregation_domain(aggs: dict) -> tuple:
    """Unwrap a single ``nested`` aggregation, returning (its sub-aggregations, the child-set query).

    ⭐ A ``nested`` aggregation is not a bucket type at all: it changes the *set of documents* the
    aggregations beneath it are computed over, from the parents to the objects of one nested path. Solr
    states that as a facet domain, and since the objects are child documents the domain is a query
    selecting them.

    Left untranslated, this reported "Unsupported aggregation type 'nested'" and the operation became a
    search with a hit count and no facet at all — the shape of a silent zero.
    """
    if len(aggs) != 1:
        return aggs, None
    name, agg_def = next(iter(aggs.items()))
    if not isinstance(agg_def, dict) or "nested" not in agg_def:
        return aggs, None
    path = (agg_def.get("nested") or {}).get("path")
    sub = agg_def.get("aggs") or agg_def.get("aggregations")
    if not path or not isinstance(sub, dict):
        logger.warning("nested aggregation '%s' states no path or sub-aggregation", name)
        return aggs, None
    # The leaf a sub-aggregation names is qualified by the path upstream and belongs to the child
    # document here, so it is renamed the same way a nested query's fields are.
    return _rename_to_child_fields(sub, path), "%s:%s" % (_CHILD_PATH_FIELD,
                                                          str(path).replace(".", "_"))


def _inner_hits_field_list(body: dict, nested: dict) -> str:
    """The ``fl`` an ``inner_hits`` block needs, or None if the body asks for no inner hits.

    ⚠️ Both parts are required and each was measured missing:

    * without ``childFilter`` carrying the nested *query*, the transformer returned every child of the
      block — 2 where ``inner_hits`` returned 1, and the extra was a 2015 answer the query excludes.
      It was silent for 2 of the 3 documents checked, which is how it would have passed a spot check.
    * without its own ``fl``, a child is rendered through the *outer* one. Measured on the sample with
      the same childFilter and limit: outer ``fl=*`` returned full children, outer ``fl=id`` returned
      ``{"id": ...}`` and nothing else, and outer ``fl=qid,tag`` — parent fields a child does not
      have — returned ``{}`` per child while the count stayed right. So the emptiness follows from the
      outer list, not from omitting the inner one; naming the child's own fields makes it independent
      of what the parent asks for.

    With ``childFilter``, ``limit`` and ``fl=<singular>_*`` the inner hits agree with upstream on all
    3 matching documents of the sample: same parents, same child counts, same users in the same order.
    """
    found = _find_inner_hits(body.get("query"))
    if not found:
        return None
    path, conf = found
    size = conf.get("size", 3) if isinstance(conf, dict) else 3
    # The child query is already collected under a referenced parameter; the transformer refers to the
    # same one, so the children returned are exactly the children the query matched.
    param = "%s0" % _NESTED_PARAM_PREFIX if not nested else sorted(nested)[0]
    child_fields = child_field_name(path, "*")
    return "*,[child childFilter=$%s limit=%s fl=%s]" % (param, size, child_fields)


def _find_inner_hits(node):
    """The (path, inner_hits) of the first nested clause asking for inner hits, or None."""
    if isinstance(node, dict):
        conf = node.get("nested")
        if isinstance(conf, dict) and isinstance(conf.get("inner_hits"), dict):
            return conf.get("path"), conf["inner_hits"]
        for value in node.values():
            found = _find_inner_hits(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_inner_hits(item)
            if found:
                return found
    return None


def _date_bounds_from_query(body: dict) -> tuple:
    """
    The start and end a date range facet should span, taken from the operation's own query.

    Solr range facets need explicit bounds; OpenSearch date_histogram derives them from the data. If
    the operation filters on the same field it aggregates, those bounds are the right ones and they
    are exact. Otherwise there is nothing in the operation to derive them from, and a guess is
    dangerous rather than merely imprecise: bounds that miss the data produce an empty facet, and an
    empty facet looks like a working operation.
    """
    query = body.get("query") if isinstance(body, dict) else None
    if not isinstance(query, dict):
        return None, None
    # walk bool wrappers to find a range clause
    def find_range(node):
        if not isinstance(node, dict):
            return None
        if "range" in node and isinstance(node["range"], dict):
            return node["range"]
        for key in ("bool", "filter", "must", "query"):
            child = node.get(key)
            if isinstance(child, dict):
                found = find_range(child)
                if found:
                    return found
            elif isinstance(child, list):
                for item in child:
                    found = find_range(item)
                    if found:
                        return found
        return None

    ranges = find_range(query)
    if not ranges:
        return None, None
    for field, spec in ranges.items():
        if not isinstance(spec, dict):
            continue
        # `from`/`to` is the spelling OpenSearch's own query builder serialises. Reading only gte/lte
        # left the bounds empty, and a range facet without them is refused outright: "Missing required
        # parameter: 'start'".
        lower = spec.get("gte") or spec.get("gt") or spec.get("from")
        upper = spec.get("lte") or spec.get("lt") or spec.get("to")
        if lower and upper:
            return _to_solr_date(lower), _to_solr_date(upper)
    return None, None


def _to_solr_date(value):
    """A Solr date literal from an OpenSearch one: a space-separated timestamp becomes ISO-8601."""
    if isinstance(value, str) and len(value) == 19 and value[10] == " ":
        return value.replace(" ", "T") + "Z"
    return value


def _convert_aggregations_to_facets(aggs: dict, date_bounds: tuple = None) -> dict:
    """
    Convert OpenSearch aggregations to Solr JSON Facet API format.

    Supported aggregation types:
      - ``terms``          → ``{"type":"terms","field":...,"limit":n}``
      - ``date_histogram`` → ``{"type":"range","field":...,"gap":"..."}``
      - ``histogram``      → ``{"type":"range","field":...,"gap":n}``
      - ``avg``            → ``"avg(field)"`` function expression
      - ``sum``            → ``"sum(field)"`` function expression
      - ``min``            → ``"min(field)"`` function expression
      - ``max``            → ``"max(field)"`` function expression
      - ``value_count``    → ``"countvals(field)"`` function expression

    Nested aggregations within bucket aggs are recursively converted.
    Unsupported aggregation types are skipped with a WARN log.

    Args:
        aggs: OpenSearch aggregations dict (the value of "aggs" or "aggregations")

    Returns:
        Solr JSON Facet API dict suitable for the ``"facet"`` key in JSON Query DSL
    """
    if not aggs or not isinstance(aggs, dict):
        return {}

    result = {}
    for agg_name, agg_def in aggs.items():
        if not isinstance(agg_def, dict):
            continue
        entry = _convert_single_agg(agg_name, agg_def, date_bounds)
        if entry is not None:
            result[agg_name] = entry

    return result


def _convert_single_agg(agg_name: str, agg_def: dict, date_bounds: tuple = None):
    """Convert a single named OpenSearch aggregation to a Solr facet entry."""
    # --- bucket aggregations ---
    if "terms" in agg_def:
        terms_conf = agg_def["terms"]
        field = normalize_field_name(terms_conf.get("field", ""))
        if not field:
            logger.warning("terms agg '%s' has no field — skipping", agg_name)
            return None
        facet_def = {
            "type": "terms",
            "field": field,
            "limit": terms_conf.get("size", 10),
        }
        nested = agg_def.get("aggs") or agg_def.get("aggregations")
        if nested:
            sub = _convert_aggregations_to_facets(nested, date_bounds)
            if sub:
                facet_def["facet"] = sub
        return facet_def

    if "geo_distance" in agg_def:
        # Distance bands from an origin. Solr has no distance-range facet type, but each band is a
        # {!geofilt} annulus, and a query facet per band is exactly that: the inner edge subtracted from
        # the outer one. Measured on 2,000 points against an independently computed answer.
        conf = agg_def["geo_distance"]
        field = normalize_field_name(conf.get("field", ""))
        origin = _lat_lon(conf.get("origin"))
        if not field or origin is None:
            logger.warning("geo_distance agg '%s' has no field or origin — skipping", agg_name)
            return None
        unit = str(conf.get("unit", "m")).lower()
        factor = _DISTANCE_UNITS.get(unit)
        if factor is None:
            logger.warning("geo_distance agg '%s' states unit '%s', which is unknown", agg_name, unit)
            return None
        bands = {}
        for band in conf.get("ranges", []):
            if not isinstance(band, dict):
                continue
            lower, upper = band.get("from"), band.get("to")
            key = band.get("key") or "%s-%s" % (
                "*" if lower is None else _number(lower), "*" if upper is None else _number(upper))
            geofilt = "{!geofilt sfield=%s pt=%s,%s d=%%s}" % (field, origin[0], origin[1])
            # ⚠️ The leading `+` is required. Written as `outer -inner` the whole string is parsed by the
            # leading local-params parser and the negation is ignored: measured, that answered 130 — the
            # outer circle alone — where the band holds 91. `+outer -inner` answers 91. `AND NOT` is
            # ignored the same way.
            if upper is None:
                # An open upper edge: everything outside the inner one.
                bands[key] = {"type": "query",
                             "q": "+*:* -%s" % (geofilt % _number(float(lower) * factor))}
            elif lower is None:
                bands[key] = {"type": "query", "q": geofilt % _number(float(upper) * factor)}
            else:
                bands[key] = {"type": "query",
                             "q": "+%s -%s" % (geofilt % _number(float(upper) * factor),
                                              geofilt % _number(float(lower) * factor))}
        if not bands:
            logger.warning("geo_distance agg '%s' states no ranges — skipping", agg_name)
            return None
        # One entry per band, so the caller splices them in beside each other.
        return bands if len(bands) > 1 else next(iter(bands.values()))

    if "multi_terms" in agg_def or "composite" in agg_def:
        # Both group by a tuple of fields, which Solr states as nested terms facets, one level per
        # source. Skipped, the aggregation vanished and the operation stayed a valid search returning a
        # hit count and no buckets — clickbench has 14 of these across 16 operations.
        #
        # They differ from each other only in how they bound the result: multi_terms truncates to a
        # size, composite paginates with an after_key. A Solr facet truncates, so a composite's total
        # bucket set is reached by asking for all of them; the JSON operations for big5 measured that
        # both engines then report the identical set.
        histograms = {}
        if "multi_terms" in agg_def:
            conf = agg_def["multi_terms"]
            sources = [t.get("field") for t in conf.get("terms", []) if isinstance(t, dict)]
            size = conf.get("size", 10)
            orders = [None] * len(sources)
        else:
            conf = agg_def["composite"]
            sources, orders, histograms = [], [], {}
            for source in conf.get("sources", []):
                if not isinstance(source, dict):
                    continue
                for _, spec in source.items():
                    if not isinstance(spec, dict):
                        continue
                    terms = spec.get("terms")
                    if isinstance(terms, dict):
                        if "script" in terms:
                            # A source computed by a script: upstream ships a serialised Calcite
                            # expression, which has no Solr spelling. Reported rather than dropped —
                            # emitting the remaining sources would bucket by a different tuple.
                            logger.warning(
                                "composite agg '%s' has a script-computed source, which has no Solr "
                                "equivalent — the aggregation is not translated.", agg_name)
                            return None
                        sources.append(terms.get("field"))
                        orders.append(terms.get("order"))
                        continue
                    # A composite source may bucket by time rather than by term.
                    histogram = spec.get("date_histogram")
                    if isinstance(histogram, dict):
                        field = normalize_field_name(histogram.get("field", ""))
                        if not field:
                            logger.warning("composite agg '%s' has a date source with no field",
                                          agg_name)
                            return None
                        histograms[len(sources)] = histogram
                        sources.append(histogram.get("field"))
                        orders.append(histogram.get("order"))
            # A composite paginates rather than truncating, so no size caps its buckets.
            size = conf.get("size", -1)

        fields = [normalize_field_name(f) for f in sources if f]
        if not fields:
            logger.warning("%s agg '%s' names no field — skipping",
                          "multi_terms" if "multi_terms" in agg_def else "composite", agg_name)
            return None

        nested = agg_def.get("aggs") or agg_def.get("aggregations")
        inner = _convert_aggregations_to_facets(nested, date_bounds) if nested else None

        # Build outermost-first, so the innermost level carries the metric sub-aggregations.
        facet_def = None
        for depth in reversed(range(len(fields))):
            histogram = histograms.get(depth)
            if histogram:
                # A time source is a range facet over the field's own bounds, as for a date_histogram.
                interval = (histogram.get("fixed_interval") or histogram.get("calendar_interval")
                            or histogram.get("interval"))
                level = {"type": "range", "field": fields[depth],
                         "gap": _calendar_interval_to_solr_gap(interval) if interval else "+1DAY"}
                if date_bounds and date_bounds[0] and date_bounds[1]:
                    level["start"] = date_bounds[0]
                    level["end"] = date_bounds[1]
            else:
                level = {"type": "terms", "field": fields[depth], "limit": size}
            order = orders[depth] if depth < len(orders) else None
            if order in ("asc", "desc") and level["type"] == "terms":
                # A composite source states its own direction, over the term rather than the count.
                level["sort"] = "index %s" % order
            if facet_def is not None:
                level["facet"] = {fields[depth + 1]: facet_def}
            elif inner:
                level["facet"] = inner
            facet_def = level
        return facet_def

    if "date_histogram" in agg_def or "auto_date_histogram" in agg_def:
        auto = "auto_date_histogram" in agg_def
        dh_conf = agg_def["auto_date_histogram"] if auto else agg_def["date_histogram"]
        field = normalize_field_name(dh_conf.get("field", ""))
        if not field:
            logger.warning("date_histogram agg '%s' has no field — skipping", agg_name)
            return None
        if auto:
            # auto_date_histogram states a bucket *target* and lets the engine choose an interval that
            # lands near it. Solr's range facet takes the interval, so the interval is computed here
            # from the same two inputs the engine uses: the range being covered and the target count.
            # That makes it deterministic rather than adaptive — which is what a Solr-to-Solr
            # comparison wants anyway, and matches how the earlier workloads carried this aggregation.
            gap = _auto_interval_to_solr_gap(dh_conf.get("buckets", 10), date_bounds)
        else:
            interval = (
                dh_conf.get("calendar_interval")
                or dh_conf.get("fixed_interval")
                or dh_conf.get("interval", "month")
            )
            gap = _calendar_interval_to_solr_gap(interval)
        start, end = (date_bounds[0], date_bounds[1]) if date_bounds else (None, None)
        if not start or not end:
            # A guess here is worse than an obvious placeholder: bounds that miss the corpus give an
            # empty facet, which reads as a working operation. pmc's data is 2010-2016 and the old
            # default spanned 2016-2027, so the facet returned almost nothing.
            logger.warning(
                "date_histogram agg '%s' has no date range in its query; emitting placeholder "
                "start/end that MUST be set to the corpus range before the operation is used",
                agg_name)
            start, end = "REPLACE_WITH_CORPUS_START", "REPLACE_WITH_CORPUS_END"
        facet_def = {
            "type": "range",
            "field": field,
            "gap": gap,
            "mincount": 1,
            "start": start,
            "end": end,
        }
        nested = agg_def.get("aggs") or agg_def.get("aggregations")
        if nested:
            sub = _convert_aggregations_to_facets(nested, date_bounds)
            if sub:
                facet_def["facet"] = sub
        return facet_def

    if "range" in agg_def:
        # An explicit list of buckets, each with its own width. Solr's range facet takes the same
        # thing under the same key, and the inclusivity defaults line up: both engines include the
        # lower bound and exclude the upper. An omitted bound means unbounded on both sides.
        r_conf = agg_def["range"]
        field = normalize_field_name(r_conf.get("field", ""))
        if not field:
            logger.warning("range agg '%s' has no field — skipping", agg_name)
            return None
        ranges = []
        for entry in r_conf.get("ranges") or []:
            if not isinstance(entry, dict):
                continue
            bucket = {}
            if "from" in entry:
                bucket["from"] = entry["from"]
            if "to" in entry:
                bucket["to"] = entry["to"]
            if bucket:
                ranges.append(bucket)
        if not ranges:
            logger.warning("range agg '%s' lists no ranges — skipping", agg_name)
            return None
        facet_def = {"type": "range", "field": field, "ranges": ranges}
        nested = agg_def.get("aggs") or agg_def.get("aggregations")
        if nested:
            sub = _convert_aggregations_to_facets(nested, date_bounds)
            if sub:
                facet_def["facet"] = sub
        return facet_def

    if "histogram" in agg_def:
        h_conf = agg_def["histogram"]
        field = normalize_field_name(h_conf.get("field", ""))
        if not field:
            logger.warning("histogram agg '%s' has no field — skipping", agg_name)
            return None
        return {
            "type": "range",
            "field": field,
            "gap": h_conf.get("interval", 1),
            "mincount": 1,
            "start": 0,
            "end": 1000000,
        }

    # --- metric aggregations (function expressions) ---
    for metric_type in ("avg", "sum", "min", "max"):
        if metric_type in agg_def:
            field = normalize_field_name(agg_def[metric_type].get("field", ""))
            if not field:
                logger.warning("%s agg '%s' has no field — skipping", metric_type, agg_name)
                return None
            return f"{metric_type}({field})"

    if "value_count" in agg_def:
        raw_field = agg_def["value_count"].get("field", "")
        # A value_count over a *metadata* field is counting documents, not values: clickbench writes
        # value_count on _index, which every document has exactly once. Solr has no such field and
        # answered 'undefined field: "_index"' — a 400 for 21 operations — so it becomes the document
        # count it means.
        if str(raw_field).startswith("_"):
            # `count(*)` is SQL's spelling, not the JSON Facet API's — it answers
            # "SyntaxError: Expected ')' at position 6 in 'count(*)'". Counting the values of the unique
            # key is the same number, since every document has exactly one: measured 1,498,137 against a
            # corpus of 1,498,137.
            return "countvals(id)"
        field = normalize_field_name(raw_field)
        if not field:
            logger.warning("value_count agg '%s' has no field — skipping", agg_name)
            return None
        return f"countvals({field})"

    if "cardinality" in agg_def:
        # Skipped, the aggregation vanished and the operation stayed a valid search reporting a hit
        # count and nothing else — 8 of clickbench's are this.
        #
        # Solr has both an exact and an estimating form. `unique()` is exact, and upstream's
        # `cardinality` is a HyperLogLog *estimate*: measured on big5, upstream answered 5,958 where the
        # true distinct count is 5,909. So the two do not agree by construction, and the exact one is
        # chosen deliberately — a benchmark comparing engines should report what the field contains, and
        # the estimate is the side that has to justify itself.
        field = normalize_field_name(agg_def["cardinality"].get("field", ""))
        if not field:
            logger.warning("cardinality agg '%s' has no field — skipping", agg_name)
            return None
        return f"unique({field})"

    agg_type = next(iter(agg_def), "unknown")
    logger.warning(
        "Unsupported aggregation type '%s' (name='%s') — skipping in Solr conversion.",
        agg_type, agg_name,
    )
    return None


def _calendar_interval_to_solr_gap(interval: str) -> str:
    """Convert an OpenSearch calendar_interval or fixed_interval to a Solr range gap string."""
    mapping = {
        "minute": "+1MINUTE",
        "1m": "+1MINUTE",
        "hour": "+1HOUR",
        "1h": "+1HOUR",
        "day": "+1DAY",
        "1d": "+1DAY",
        "week": "+7DAYS",
        "1w": "+7DAYS",
        "month": "+1MONTH",
        "1m_month": "+1MONTH",  # avoid conflict with 1m (minute)
        "quarter": "+3MONTHS",
        "1q": "+3MONTHS",
        "year": "+1YEAR",
        "1y": "+1YEAR",
    }
    return mapping.get(str(interval).lower(), "+1MONTH")


def _auto_interval_to_solr_gap(target_buckets, date_bounds) -> str:
    """
    Choose the interval an ``auto_date_histogram`` would settle on, as a Solr range gap.

    OpenSearch is given a bucket *target* and picks an interval from a fixed ladder — the coarsest one
    that keeps the bucket count at or under the target. The same choice is made here, from the same
    two inputs: the range the query covers and the target. It is a deterministic interval rather than
    an adaptive one, which is what a range facet needs and what a Solr-to-Solr comparison wants.

    Without bounds there is nothing to divide, so the safe answer is the ladder's coarsest step: a gap
    too fine over an unknown range produces a facet with a bucket per document.
    """
    ladder = [
        (1, "+1SECOND"), (5, "+5SECONDS"), (10, "+10SECONDS"), (30, "+30SECONDS"),
        (60, "+1MINUTE"), (300, "+5MINUTES"), (600, "+10MINUTES"), (1800, "+30MINUTES"),
        (3600, "+1HOUR"), (10800, "+3HOURS"), (43200, "+12HOURS"),
        (86400, "+1DAY"), (604800, "+7DAYS"),
        (2592000, "+1MONTH"), (7776000, "+3MONTHS"), (31536000, "+1YEAR"),
    ]
    try:
        target = max(1, int(target_buckets))
    except (TypeError, ValueError):
        target = 10

    span = None
    if date_bounds and date_bounds[0] and date_bounds[1]:
        span = _date_span_seconds(date_bounds[0], date_bounds[1])
    if not span:
        return ladder[-1][1]

    wanted = span / target
    for seconds, gap in ladder:
        if seconds >= wanted:
            return gap
    return ladder[-1][1]


def _date_span_seconds(start, end):
    """Seconds between two ISO-8601 instants, or None if either is not one (a placeholder, say)."""
    import datetime

    def parse(value):
        text = str(value).strip().replace("Z", "+00:00")
        try:
            return datetime.datetime.fromisoformat(text)
        except ValueError:
            return None

    lo, hi = parse(start), parse(end)
    if lo is None or hi is None or hi <= lo:
        return None
    return (hi - lo).total_seconds()


def _convert_date_to_solr_format(date_str, os_format=None) -> str:
    """
    Convert an OpenSearch date string to Solr ISO 8601 format.

    Args:
        date_str: Date string in various OpenSearch formats
        os_format: Optional OpenSearch date format pattern (e.g., "dd/MM/yyyy")

    Returns:
        ISO 8601 date string for Solr (e.g., "2015-01-01T00:00:00Z")

    If the date is already in ISO format or conversion fails, returns the
    original string unchanged.
    """
    if not isinstance(date_str, str) or date_str in ("*", "now"):
        return date_str

    # Map OpenSearch date format patterns to Python strptime format
    OS_TO_PYTHON_FORMAT = {
        "dd/MM/yyyy": "%d/%m/%Y",
        "MM/dd/yyyy": "%m/%d/%Y",
        "yyyy-MM-dd": "%Y-%m-%d",
        "yyyy/MM/dd": "%Y/%m/%d",
        "dd-MM-yyyy": "%d-%m-%Y",
        "MM-dd-yyyy": "%m-%d-%Y",
        # Add more as needed
    }

    # If format is provided, use it to parse the date
    if os_format:
        python_fmt = OS_TO_PYTHON_FORMAT.get(os_format)
        if python_fmt:
            try:
                dt = datetime.strptime(date_str, python_fmt)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                logger.warning(f"Failed to parse date '{date_str}' with format '{os_format}'")
                return date_str
        else:
            logger.warning(f"Unknown OpenSearch date format: '{os_format}'")

    # Try common patterns if no format specified
    for python_fmt in OS_TO_PYTHON_FORMAT.values():
        try:
            dt = datetime.strptime(date_str, python_fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

    # If it's already in ISO-like format, return as-is
    # (handles cases like "2015-01-01T00:00:00Z" or partial ISO)
    if "T" in date_str or len(date_str) == 10:  # YYYY-MM-DD
        return date_str

    logger.warning(f"Could not parse date '{date_str}', using as-is")
    return date_str
