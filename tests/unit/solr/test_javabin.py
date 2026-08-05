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
Unit tests for the javabin writer.

These check the encoding against the format JavaBinCodec defines, by decoding what was written. A
test that only asserted "some bytes came out" would have passed while the runner reported success on
an update that indexed nothing, which is how the dropped-body defect got as far as it did.
"""

import datetime
import unittest

from solrorbit.utils import javabin
from solrorbit.utils.javabin import encode_update_request


class _Reader:
    """
    Minimal javabin reader, enough to verify what the writer produced.

    Deliberately independent of the writer's internals: it walks the byte stream by tag, the way
    Solr's decoder does, so a change that keeps the writer self-consistent but diverges from the
    format still fails.
    """

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.extern = []

    def byte(self):
        b = self.data[self.pos]
        self.pos += 1
        return b

    def vint(self):
        shift = 0
        result = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    def size(self, tag_byte):
        size = tag_byte & 0x1F
        if size == 0x1F:
            size += self.vint()
        return size

    def value(self):
        tag = self.byte()
        high = tag & 0xE0
        if high == javabin.STR:
            n = self.size(tag)
            s = self.data[self.pos:self.pos + n].decode("utf-8")
            self.pos += n
            return s
        if high == javabin.EXTERN_STRING:
            idx = self.size(tag)
            if idx:
                return self.extern[idx - 1]
            s = self.value()
            self.extern.append(s)
            return s
        if high == javabin.ARR:
            return [self.value() for _ in range(self.size(tag))]
        if high == javabin.NAMED_LST or high == javabin.ORDERED_MAP:
            return [(self.value(), self.value()) for _ in range(self.size(tag))]
        if tag == javabin.NULL:
            return None
        if tag == javabin.BOOL_TRUE:
            return True
        if tag == javabin.BOOL_FALSE:
            return False
        if tag == javabin.INT:
            v = int.from_bytes(self.data[self.pos:self.pos + 4], "big", signed=True)
            self.pos += 4
            return v
        if tag == javabin.LONG:
            v = int.from_bytes(self.data[self.pos:self.pos + 8], "big", signed=True)
            self.pos += 8
            return v
        if tag == javabin.DATE:
            millis = int.from_bytes(self.data[self.pos:self.pos + 8], "big", signed=True)
            self.pos += 8
            return datetime.datetime.fromtimestamp(millis / 1000, datetime.timezone.utc)
        if tag == javabin.DOUBLE:
            import struct
            v = struct.unpack(">d", self.data[self.pos:self.pos + 8])[0]
            self.pos += 8
            return v
        if tag == javabin.FLOAT:
            import struct
            v = struct.unpack(">f", self.data[self.pos:self.pos + 4])[0]
            self.pos += 4
            return v
        if tag == javabin.MAP:
            return {self.value(): self.value() for _ in range(self.vint())}
        if tag == javabin.BYTEARR:
            n = self.vint()
            v = self.data[self.pos:self.pos + n]
            self.pos += n
            return v
        if tag == javabin.ITERATOR:
            items = []
            while True:
                save = self.pos
                nxt = self.byte()
                if nxt == javabin.END:
                    return items
                self.pos = save
                items.append(self.value())
        if tag == javabin.SOLRINPUTDOC:
            count = self.vint()
            boost = self.value()
            assert boost == 1.0, "document boost must be 1, got %r" % boost
            return {self.value(): self.value() for _ in range(count)}
        raise AssertionError("unexpected tag 0x%02x at %d" % (tag, self.pos - 1))

    def request(self):
        version = self.byte()
        assert version == 2, "version byte %d" % version
        return dict(self.value())


def _decode(data):
    return _Reader(data).request()


class TestJavaBinUpdateRequest(unittest.TestCase):
    def test_version_byte_comes_first(self):
        # The literal 2, not javabin.VERSION: comparing the constant with itself would pass whatever
        # it were set to, and the server rejects anything else with "Invalid version (expected 2)".
        self.assertEqual(2, encode_update_request([{"id": "1"}])[0])

    def test_documents_round_trip(self):
        docs = [{"id": "a", "status": 200, "size": 24736},
                {"id": "b", "status": 404, "size": 7}]
        decoded = _decode(encode_update_request(docs))
        self.assertEqual(docs, decoded["docs"])

    def test_the_three_top_level_entries_are_present_in_order(self):
        # JavabinLoader reads params, delByQ and docs; a different set or order is a different format.
        reader = _Reader(encode_update_request([{"id": "1"}]))
        reader.byte()          # version
        self.assertEqual(["params", "delByQ", "docs"], [name for name, _ in reader.value()])

    def test_value_types_round_trip(self):
        when = datetime.datetime(1998, 4, 30, 19, 30, 17, tzinfo=datetime.timezone.utc)
        doc = {
            "id": "types",
            "an_int": 42,
            "a_long": 2 ** 40,
            "a_float": 1.5,
            "a_true": True,
            "a_false": False,
            "a_null": None,
            "a_list": ["x", "y"],
            "a_date": when,
        }
        decoded = _decode(encode_update_request([doc]))["docs"][0]
        self.assertEqual(42, decoded["an_int"])
        self.assertEqual(2 ** 40, decoded["a_long"])
        self.assertAlmostEqual(1.5, decoded["a_float"])
        self.assertIs(True, decoded["a_true"])
        self.assertIs(False, decoded["a_false"])
        self.assertIsNone(decoded["a_null"])
        self.assertEqual(["x", "y"], decoded["a_list"])
        self.assertEqual(when, decoded["a_date"])

    def test_an_int_beyond_32_bits_becomes_a_long(self):
        # Python has one integer type, so the writer picks the narrowest javabin type that fits.
        for value, tag in ((2 ** 31 - 1, javabin.INT), (2 ** 31, javabin.LONG), (-(2 ** 31), javabin.INT)):
            body = encode_update_request([{"id": "x", "n": value}])
            self.assertEqual(value, _decode(body)["docs"][0]["n"], msg="value %d" % value)
            self.assertIn(bytes([tag]), body, msg="value %d should use tag 0x%02x" % (value, tag))

    def test_an_integer_too_large_for_a_long_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_update_request([{"id": "x", "n": 2 ** 64}])

    def test_an_unencodable_type_is_rejected(self):
        with self.assertRaises(TypeError):
            encode_update_request([{"id": "x", "obj": object()}])

    def test_a_repeated_field_name_is_written_once(self):
        # Extern strings are the format's own compression: the second document refers to the field
        # name by index. If the writer emitted it twice, the reader's table would desynchronise.
        body = encode_update_request([{"id": "a", "clientip": "1.1.1.1"},
                                      {"id": "b", "clientip": "2.2.2.2"}])
        self.assertEqual(1, body.count(b"clientip"))
        decoded = _decode(body)["docs"]
        self.assertEqual(["1.1.1.1", "2.2.2.2"], [d["clientip"] for d in decoded])

    def test_documents_may_be_a_generator(self):
        # A corpus is streamed, so the writer must not require a materialised list.
        docs = ({"id": str(i)} for i in range(3))
        self.assertEqual(["0", "1", "2"], [d["id"] for d in _decode(encode_update_request(docs))["docs"]])

    def test_params_are_carried_as_multi_valued(self):
        decoded = _decode(encode_update_request([{"id": "1"}], params={"commit": "true"}))
        self.assertEqual([("commit", ["true"])], decoded["params"])

    def test_commit_within_is_added_to_params(self):
        decoded = _decode(encode_update_request([{"id": "1"}], commit_within=1000))
        self.assertIn(("commitWithin", [1000]), decoded["params"])

    def test_an_empty_document_set_still_produces_a_valid_request(self):
        decoded = _decode(encode_update_request([]))
        self.assertEqual([], decoded["docs"])


if __name__ == "__main__":
    unittest.main()


class TestJavaBinReader:
    """Decode what the writer wrote, and what a real response looked like."""

    def test_a_round_trip_recovers_every_scalar_type(self):
        from solrorbit.utils.javabin import JavaBinWriter, decode_response
        writer = JavaBinWriter()
        writer._byte(javabin.VERSION)
        writer._value({"i": 7, "l": 2 ** 40, "s": "text", "b": True,
                       "n": None, "d": 1.5, "arr": [1, 2, 3]})
        decoded = decode_response(writer._out.getvalue())
        assert decoded == {"i": 7, "l": 2 ** 40, "s": "text", "b": True,
                           "n": None, "d": 1.5, "arr": [1, 2, 3]}

    def test_a_document_list_decodes_to_its_header_and_docs(self):
        # A response's hit count lives in the SOLRDOCLST header; a reader that skipped it would report
        # zero hits for every binary search.
        from solrorbit.utils.javabin import decode_response
        # version, NAMED_LST of one entry "response" -> SOLRDOCLST(numFound=3, start=0, maxScore=null,
        # docs=[]), written by hand so the test does not depend on the writer emitting a doc list.
        import io
        out = io.BytesIO()
        out.write(bytes([javabin.VERSION]))
        out.write(bytes([javabin.NAMED_LST | 1]))
        out.write(bytes([javabin.EXTERN_STRING | 0]))
        name = "response".encode("utf-8")
        out.write(bytes([javabin.STR | len(name)]))
        out.write(name)
        out.write(bytes([javabin.SOLRDOCLST]))
        out.write(bytes([javabin.ARR | 3]))
        out.write(bytes([javabin.SLONG | 0x03]))   # numFound = 3
        out.write(bytes([javabin.SLONG | 0x00]))   # start = 0
        out.write(bytes([javabin.NULL]))           # maxScore
        out.write(bytes([javabin.ARR | 0]))        # docs
        decoded = decode_response(out.getvalue())
        assert decoded["response"]["numFound"] == 3
        assert decoded["response"]["docs"] == []

    def test_a_packed_int_is_a_value_not_a_length(self):
        # SINT/SLONG pack the value into the same low bits a length uses. Reading them as a length
        # consumed the varint carrying the rest of the value and every field after it decoded as
        # garbage: a live response reported metrics_size 91663 where it was 1855. It only shows when
        # the low nibble is 0x0f, so most values decoded correctly and the failure looked sporadic.
        import io
        from solrorbit.utils.javabin import decode_response
        for value in (0, 1, 14, 15, 16, 31, 1855, 91663, 2 ** 20):
            out = io.BytesIO()
            out.write(bytes([javabin.VERSION]))
            # Write it the way solrj's writeSInt does: four bits in the tag, continuation flag at 0x10.
            if value < 0x10:
                out.write(bytes([javabin.SINT | value]))
            else:
                out.write(bytes([javabin.SINT | 0x10 | (value & 0x0F)]))
                remaining = value >> 4
                while remaining > 0x7F:
                    out.write(bytes([(remaining & 0x7F) | 0x80]))
                    remaining >>= 7
                out.write(bytes([remaining]))
            assert decode_response(out.getvalue()) == value, value

    def test_a_packed_int_does_not_desynchronise_what_follows_it(self):
        # The symptom that mattered: the *next* value was wrong, not this one.
        import io
        from solrorbit.utils.javabin import decode_response
        out = io.BytesIO()
        out.write(bytes([javabin.VERSION]))
        out.write(bytes([javabin.ARR | 2]))
        out.write(bytes([javabin.SINT | 0x10 | 0x0F]))   # 1855 = 0x73F -> low nibble 0xF
        out.write(bytes([0x73]))
        marker = "after".encode("utf-8")
        out.write(bytes([javabin.STR | len(marker)]))
        out.write(marker)
        assert decode_response(out.getvalue()) == [1855, "after"]

    def test_a_repeated_extern_string_resolves_by_its_table_index(self):
        # The second occurrence of a name is a *reference* into the table, not the name again: a
        # response repeats every field name once per document. An off-by-one in the table lookup
        # returns the wrong name, or walks off the end.
        from solrorbit.utils.javabin import JavaBinWriter, decode_response
        writer = JavaBinWriter()
        writer._byte(javabin.VERSION)
        writer._tag(javabin.ARR, 2)
        # Two documents carrying the same two field names, so the second document's names are
        # references rather than literals.
        for value in (1, 2):
            writer._tag(javabin.NAMED_LST, 2)
            writer._extern_str("first")
            writer._value(value)
            writer._extern_str("second")
            writer._value(value)
        decoded = decode_response(writer._out.getvalue())
        assert decoded == [{"first": 1, "second": 1}, {"first": 2, "second": 2}]
