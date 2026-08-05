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

"""Unit tests for solrorbit/conversion/schema.py"""

import unittest

from solrorbit.conversion.schema import (
    translate_opensearch_mapping,
    generate_schema_xml,
)


class TestFieldTypeAndDocValues(unittest.TestCase):
    """
    Two things the type table got wrong, both corrected by hand in a workload before this.

    A geo_point became a string, on which nothing spatial works. And doc_values were given only to
    keyword fields, where upstream gives them to numeric, date and boolean too — noaa's
    station_elevation went without them on a field its operations filter by.
    """

    def test_a_geo_point_uses_the_spatial_field_type(self):
        fields, _ = translate_opensearch_mapping({"location": {"type": "geo_point"}})
        self.assertEqual("location_rpt", fields["location"]["type"])

    def test_a_spatial_field_gets_no_doc_values(self):
        # An RPT field cannot expose them, which is why geonames and noaa carry lat/lon separately.
        fields, _ = translate_opensearch_mapping({"location": {"type": "geo_point"}})
        self.assertNotIn("docValues", fields["location"])

    def test_numeric_date_and_boolean_get_doc_values(self):
        # OpenSearch defaults doc_values to true for all three — NumberFieldMapper, DateFieldMapper
        # and BooleanFieldMapper all declare docValuesParam(…, true).
        fields, _ = translate_opensearch_mapping({
            "n": {"type": "integer"}, "f": {"type": "float"},
            "d": {"type": "date"}, "b": {"type": "boolean"}, "k": {"type": "keyword"},
        })
        for name in ("n", "f", "d", "b", "k"):
            self.assertTrue(fields[name].get("docValues"), msg="%s should have docValues" % name)

    def test_a_mapping_that_switches_doc_values_off_says_so_explicitly(self):
        # ⚠️ Not merely absent: the generated string fieldType declares docValues="true", so a field
        # that omits the attribute inherits it. Silence would give doc values to the one field
        # upstream switches them off for.
        fields, _ = translate_opensearch_mapping({"m": {"type": "keyword", "doc_values": False}})
        self.assertIs(False, fields["m"]["docValues"])

    def test_match_only_text_is_analysed_text(self):
        # MatchOnlyTextFieldMapper extends TextFieldMapper — positions and norms off to save storage,
        # still analysed. Absent from the type table it fell through to string, and big5's message
        # field then held each log line as one term: message:monkey matched nothing where upstream
        # found 103,349 documents.
        fields, _ = translate_opensearch_mapping({"message": {"type": "match_only_text"}})
        self.assertEqual("text_general", fields["message"]["type"])

    def test_the_types_the_ported_workloads_use_are_all_known(self):
        # A type absent from the table falls through to string silently, which is how match_only_text
        # and geo_point both went wrong. These are the ones the six ported workloads actually declare.
        for os_type, expected in (("match_only_text", "text_general"), ("wildcard", "string"),
                                  ("ip", "string"), ("double_range", "pdouble"),
                                  ("geo_point", "location_rpt"), ("scaled_float", "pdouble"),
                                  ("half_float", "pfloat")):
            fields, _ = translate_opensearch_mapping({"f": {"type": os_type}})
            self.assertEqual(expected, fields["f"]["type"], msg="%s mapped wrongly" % os_type)

    def test_an_unindexed_field_is_declared_unindexed(self):
        # http_logs maps message as a keyword with index false: it is the raw log line the grok
        # pipeline reads, stored and returned but never searched.
        fields, _ = translate_opensearch_mapping({"message": {"type": "keyword", "index": False}})
        self.assertFalse(fields["message"]["indexed"])
        self.assertTrue(fields["message"]["stored"])

    def test_a_text_field_gets_no_doc_values(self):
        # A text field has none, and asking for them is an error.
        fields, _ = translate_opensearch_mapping({"body": {"type": "text"}})
        self.assertNotIn("docValues", fields["body"])

    def test_binary_gets_doc_values_switched_off(self):
        # BinaryFieldMapper is the one that declares docValuesParam(…, false).
        fields, _ = translate_opensearch_mapping({"blob": {"type": "binary"}})
        self.assertIs(False, fields["blob"]["docValues"])


class TestNestedObjectMappings(unittest.TestCase):
    """
    A mapping entry holding nested properties is an object, not a field.

    big5 declares agent, aws, cloud and the rest that way. Treating each as a field put the object's
    name in the schema and none of its leaves, so indexing failed with
    `undefined field: "aws_cloudwatch_log_stream"` — the name the documents and the queries both use.
    """

    def test_an_object_contributes_its_leaves(self):
        fields, _ = translate_opensearch_mapping({
            "agent": {"type": "object", "properties": {
                "id": {"type": "keyword"},
                "name": {"type": "keyword"},
            }},
        })
        self.assertIn("agent_id", fields)
        self.assertIn("agent_name", fields)
        self.assertNotIn("agent", fields, "the object's own name is not a field")

    def test_nesting_goes_all_the_way_down(self):
        # big5 nests two levels: aws -> cloudwatch -> log_stream.
        fields, _ = translate_opensearch_mapping({
            "aws": {"type": "object", "properties": {
                "cloudwatch": {"type": "object", "properties": {
                    "log_stream": {"type": "keyword"},
                }},
            }},
        })
        self.assertIn("aws_cloudwatch_log_stream", fields)

    def test_an_object_with_no_type_is_still_an_object(self):
        # Some mappings omit the type and give only properties.
        fields, _ = translate_opensearch_mapping({
            "metrics": {"properties": {"size": {"type": "integer"}}},
        })
        self.assertIn("metrics_size", fields)

    def test_an_open_object_becomes_a_dynamic_field(self):
        # `"host": {"type": "object"}` accepts any sub-field upstream; Solr needs a declaration, and
        # big5's documents carry host.name, which an operation collapses on.
        fields, _ = translate_opensearch_mapping({"host": {"type": "object"}})
        self.assertIn("host_*", fields)
        self.assertTrue(fields["host_*"].get("dynamic"))

    def test_a_dynamic_field_is_rendered_as_dynamicField(self):
        xml = generate_schema_xml({"host_*": {"type": "string", "dynamic": True}})
        self.assertIn('<dynamicField name="host_*"', xml)
        self.assertNotIn('<field name="host_*"', xml)

    def test_a_plain_field_is_still_rendered_as_field(self):
        xml = generate_schema_xml({"status": {"type": "pint"}})
        self.assertIn('<field name="status"', xml)
        self.assertNotIn("dynamicField name=\"status\"", xml)


class TestTranslateOpenSearchMapping(unittest.TestCase):
    """Test OpenSearch to Solr mapping translation."""

    def test_simple_field_translation(self):
        """Test basic field type translation without multi-fields."""
        properties = {
            "title": {"type": "text"},
            "count": {"type": "integer"},
            "price": {"type": "double"},
        }

        field_defs, copy_fields = translate_opensearch_mapping(properties)

        # Check field definitions
        self.assertEqual("text_general", field_defs["title"]["type"])
        self.assertEqual("pint", field_defs["count"]["type"])
        self.assertEqual("pdouble", field_defs["price"]["type"])

        # No copy fields for simple fields
        self.assertEqual(0, len(copy_fields))

    def test_keyword_field_has_docvalues(self):
        """Test that keyword fields get docValues=True."""
        properties = {
            "country_code": {"type": "keyword"},
        }

        field_defs, _copy_fields = translate_opensearch_mapping(properties)

        self.assertEqual("string", field_defs["country_code"]["type"])
        self.assertTrue(field_defs["country_code"]["docValues"])

    def test_multi_field_with_raw_suffix(self):
        """Test multi-field with .raw sub-field creates separate field and copyField."""
        properties = {
            "country_code": {
                "type": "text",
                "fields": {
                    "raw": {"type": "keyword"}
                }
            }
        }

        field_defs, copy_fields = translate_opensearch_mapping(properties)

        # Main field should be text_general
        self.assertEqual("text_general", field_defs["country_code"]["type"])

        # Sub-field should be created with underscore naming
        self.assertIn("country_code_raw", field_defs)
        self.assertEqual("string", field_defs["country_code_raw"]["type"])
        self.assertTrue(field_defs["country_code_raw"]["docValues"])

        # Should have one copyField directive
        self.assertEqual(1, len(copy_fields))
        self.assertEqual(("country_code", "country_code_raw"), copy_fields[0])

    def test_multi_field_with_keyword_suffix(self):
        """Test multi-field with .keyword sub-field."""
        properties = {
            "name": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            }
        }

        field_defs, copy_fields = translate_opensearch_mapping(properties)

        # Sub-field should be created
        self.assertIn("name_keyword", field_defs)
        self.assertEqual("string", field_defs["name_keyword"]["type"])
        self.assertTrue(field_defs["name_keyword"]["docValues"])

        # Should have copyField directive
        self.assertEqual(1, len(copy_fields))
        self.assertEqual(("name", "name_keyword"), copy_fields[0])

    def test_multi_field_with_multiple_subfields(self):
        """Test field with multiple sub-fields."""
        properties = {
            "title": {
                "type": "text",
                "fields": {
                    "raw": {"type": "keyword"},
                    "sort": {"type": "keyword"}
                }
            }
        }

        field_defs, copy_fields = translate_opensearch_mapping(properties)

        # Main field
        self.assertEqual("text_general", field_defs["title"]["type"])

        # Both sub-fields should be created
        self.assertIn("title_raw", field_defs)
        self.assertIn("title_sort", field_defs)

        # Should have two copyField directives
        self.assertEqual(2, len(copy_fields))
        self.assertIn(("title", "title_raw"), copy_fields)
        self.assertIn(("title", "title_sort"), copy_fields)


class TestGenerateSchemaXML(unittest.TestCase):
    """Test schema.xml generation."""

    def test_simple_schema_generation(self):
        """Test basic schema generation without multi-fields."""
        field_defs = {
            "title": {"type": "text_general", "indexed": True, "stored": True},
            "count": {"type": "pint", "indexed": True, "stored": True},
        }

        schema_xml = generate_schema_xml(field_defs)

        # Check that fields are present
        self.assertIn('<field name="title" type="text_general"', schema_xml)
        self.assertIn('<field name="count" type="pint"', schema_xml)

        # Check required SolrCloud fields
        self.assertIn('<field name="id"', schema_xml)
        self.assertIn('<field name="_version_"', schema_xml)

    def test_schema_with_copyfields(self):
        """Test schema generation with copyField directives."""
        field_defs = {
            "country_code": {"type": "text_general", "indexed": True, "stored": True},
            "country_code_raw": {"type": "string", "indexed": True, "stored": True, "docValues": True},
        }
        copy_fields = [("country_code", "country_code_raw")]

        schema_xml = generate_schema_xml(field_defs, copy_fields=copy_fields)

        # Check that both fields are present
        self.assertIn('<field name="country_code" type="text_general"', schema_xml)
        self.assertIn('<field name="country_code_raw" type="string"', schema_xml)

        # Check that copyField directive is present
        self.assertIn('<copyField source="country_code" dest="country_code_raw"', schema_xml)

    def test_schema_with_multiple_copyfields(self):
        """Test schema generation with multiple copyField directives."""
        field_defs = {
            "title": {"type": "text_general", "indexed": True, "stored": True},
            "title_raw": {"type": "string", "indexed": True, "stored": True, "docValues": True},
            "title_sort": {"type": "string", "indexed": True, "stored": True, "docValues": True},
        }
        copy_fields = [
            ("title", "title_raw"),
            ("title", "title_sort"),
        ]

        schema_xml = generate_schema_xml(field_defs, copy_fields=copy_fields)

        # Check that all copyField directives are present
        self.assertIn('<copyField source="title" dest="title_raw"', schema_xml)
        self.assertIn('<copyField source="title" dest="title_sort"', schema_xml)

    def test_docvalues_attribute_in_schema(self):
        """Test that docValues attribute is properly rendered."""
        field_defs = {
            "name_keyword": {"type": "string", "indexed": True, "stored": True, "docValues": True},
        }

        schema_xml = generate_schema_xml(field_defs)

        self.assertIn('docValues="true"', schema_xml)


if __name__ == "__main__":
    unittest.main()
