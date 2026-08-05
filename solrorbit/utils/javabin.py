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
Write an update request in Solr's ``javabin`` binary format.

Why this exists: a workload may compare a binary indexing transport against JSON — OpenSearch
Benchmark's http_logs does, through its gRPC operations — and Solr's own binary format is the
comparable thing. ``UpdateRequestHandler`` registers ``application/javabin`` alongside
``application/json``, so the same endpoint accepts either. No Python library writes javabin, and
pysolr is JSON/XML only, hence this writer.

Only the encoder is here, and only the subset an add-documents request needs. It follows
``JavaBinCodec`` in Solr's solrj: a one-byte version, then a NamedList of ``params``, ``delByQ`` and
``docs``, where ``docs`` is an iterator of ``SOLRINPUTDOC`` values.
"""

import datetime
import io

# Tag bytes, from JavaBinCodec. The low tags are plain type markers; the high ones pack a length
# into the same byte, which is why they are shifted by five bits.
NULL = 0
BOOL_TRUE = 1
BOOL_FALSE = 2
BYTE = 3
SHORT = 4
DOUBLE = 5
INT = 6
LONG = 7
FLOAT = 8
DATE = 9
MAP = 10
SOLRDOC = 11
SOLRDOCLST = 12
BYTEARR = 13
ITERATOR = 14
END = 15
SOLRINPUTDOC = 16

STR = 1 << 5
SINT = 2 << 5
SLONG = 3 << 5
ARR = 4 << 5
ORDERED_MAP = 5 << 5
NAMED_LST = 6 << 5
EXTERN_STRING = 7 << 5

VERSION = 2


class JavaBinWriter:
    """
    Encode Python values as javabin.

    An instance keeps the extern-string table for one request, so it is single-use: the reader
    rebuilds that table in the same order and a shared writer would desynchronise it.
    """

    def __init__(self):
        self._out = io.BytesIO()
        # Extern strings are written once and referred to by a 1-based index afterwards. The reader
        # appends to its own list in the same order, so the numbering must match exactly.
        self._extern = {}

    # -- primitives ---------------------------------------------------------

    def _byte(self, b):
        self._out.write(bytes([b & 0xFF]))

    def _vint(self, i):
        while i & ~0x7F:
            self._byte((i & 0x7F) | 0x80)
            i >>= 7
        self._byte(i)

    def _tag(self, tag, size=None):
        if size is None:
            self._byte(tag)
            return
        if tag & 0xE0:
            # A tag that packs its length: values under 0x1f fit in the tag byte itself.
            if size < 0x1F:
                self._byte(tag | size)
            else:
                self._byte(tag | 0x1F)
                self._vint(size - 0x1F)
        else:
            self._byte(tag)
            self._vint(size)

    def _str(self, s):
        encoded = s.encode("utf-8")
        self._tag(STR, len(encoded))
        self._out.write(encoded)

    def _extern_str(self, s):
        idx = self._extern.get(s, 0)
        self._tag(EXTERN_STRING, idx)
        if idx == 0:
            self._str(s)
            self._extern[s] = len(self._extern) + 1

    def _int(self, i):
        self._byte(INT)
        self._out.write(i.to_bytes(4, "big", signed=True))

    def _long(self, i):
        self._byte(LONG)
        self._out.write(i.to_bytes(8, "big", signed=True))

    def _double(self, d):
        import struct
        self._byte(DOUBLE)
        self._out.write(struct.pack(">d", d))

    def _float(self, f):
        import struct
        self._byte(FLOAT)
        self._out.write(struct.pack(">f", f))

    def _date(self, dt):
        # javabin dates are epoch millis in a long, tagged as DATE.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        millis = int(dt.timestamp() * 1000)
        self._byte(DATE)
        self._out.write(millis.to_bytes(8, "big", signed=True))

    # -- composites ---------------------------------------------------------

    def _value(self, val):
        if val is None:
            self._byte(NULL)
        elif isinstance(val, bool):
            self._byte(BOOL_TRUE if val else BOOL_FALSE)
        elif isinstance(val, int):
            # Python has one integer type; pick the narrowest javabin one that fits, as solrj does
            # by static type. A value outside long range cannot be represented.
            if -(2 ** 31) <= val < 2 ** 31:
                self._int(val)
            elif -(2 ** 63) <= val < 2 ** 63:
                self._long(val)
            else:
                raise ValueError("integer too large for javabin: %d" % val)
        elif isinstance(val, float):
            self._double(val)
        elif isinstance(val, str):
            self._str(val)
        elif isinstance(val, (bytes, bytearray)):
            self._tag(BYTEARR, len(val))
            self._out.write(bytes(val))
        elif isinstance(val, datetime.datetime):
            self._date(val)
        elif isinstance(val, (list, tuple)):
            self._tag(ARR, len(val))
            for item in val:
                self._value(item)
        elif isinstance(val, dict):
            self._tag(MAP, len(val))
            for key, item in val.items():
                self._extern_str(str(key))
                self._value(item)
        else:
            raise TypeError("cannot encode %r as javabin" % type(val))

    def _named_list(self, pairs):
        self._tag(NAMED_LST, len(pairs))
        for name, val in pairs:
            self._extern_str(name)
            self._value(val)

    def _document(self, doc):
        # A document is a field count, a boost float that is always 1, then name/value pairs. The
        # server logs a warning for any other boost, so it is not configurable here.
        self._tag(SOLRINPUTDOC, len(doc))
        self._float(1.0)
        for name, val in doc.items():
            self._extern_str(str(name))
            self._value(val)

    # -- the request --------------------------------------------------------

    def update_request(self, docs, params=None, commit_within=None):
        """
        Encode an add-documents update request and return the bytes.

        *docs* is any iterable of field dicts; it is consumed once and streamed, so a generator over
        a large corpus does not have to be materialised.

        ⚠️ *params* go into the body because that is where javabin carries them, but measured against
        a live node, ``JavabinLoader`` does not act on them: a request with ``commit`` set here
        returned 200 and logged ``add=[j1, j2]`` while leaving the documents invisible. Put
        ``commit`` in the query string, which the handler does read.
        """
        request_params = list((params or {}).items())
        if commit_within is not None:
            request_params.append(("commitWithin", commit_within))

        # Three entries, in the order the reader expects: params, delByQ, docs.
        self._byte(VERSION)
        self._tag(NAMED_LST, 3)

        self._extern_str("params")
        self._tag(NAMED_LST, len(request_params))
        for name, val in request_params:
            self._extern_str(name)
            # Request parameters are multi-valued on the wire, as they are in a query string.
            self._value(val if isinstance(val, (list, tuple)) else [val])

        self._extern_str("delByQ")
        self._byte(NULL)

        self._extern_str("docs")
        self._tag(ITERATOR)
        for doc in docs:
            self._document(doc)
        self._tag(END)

        return self._out.getvalue()


def encode_update_request(docs, params=None, commit_within=None):
    """Encode *docs* as a javabin add-documents request. One writer per request."""
    return JavaBinWriter().update_request(docs, params=params, commit_within=commit_within)


class JavaBinReader:
    """
    Decode a javabin response.

    Only what a search response needs: the reader must handle every tag Solr may emit in one, since a
    tag it skips desynchronises everything after it. The extern-string table is rebuilt in the order
    the writer filled it, so an instance decodes exactly one response.
    """

    def __init__(self, data):
        self._in = io.BytesIO(data)
        self._extern = []

    def _byte(self):
        b = self._in.read(1)
        if not b:
            raise EOFError("javabin response ended mid-value")
        return b[0]

    def _vint(self):
        shift = 0
        value = 0
        while True:
            b = self._byte()
            value |= (b & 0x7F) << shift
            if not b & 0x80:
                return value
            shift += 7

    def _size(self, tag_byte, tag):
        size = tag_byte & 0x1F
        if size == 0x1F:
            size += self._vint()
        return size

    def _string(self, size):
        return self._in.read(size).decode("utf-8")

    def read(self):
        """Read the version byte and return the top-level value."""
        self._byte()  # version
        return self._value()

    def _value(self):
        import struct
        tag_byte = self._byte()
        tag = tag_byte & 0xE0
        if tag:
            # SINT and SLONG pack a *value* into the low five bits, not a length: reading them as a
            # length consumed the varint that carried the rest of the value, and every field after it
            # decoded as garbage. It only shows on a value whose low nibble is 0x0f, which is why a
            # response could decode correctly and then break on the next document.
            if tag == SINT or tag == SLONG:
                value = tag_byte & 0x0F
                if tag_byte & 0x10:
                    shift = 4
                    while True:
                        b = self._byte()
                        value |= (b & 0x7F) << shift
                        if not b & 0x80:
                            break
                        shift += 7
                return value
            size = self._size(tag_byte, tag)
            if tag == STR:
                return self._string(size)
            if tag == ARR:
                return [self._value() for _ in range(size)]
            if tag in (ORDERED_MAP, NAMED_LST):
                # A NamedList allows repeated names, so it decodes to a dict only because a search
                # response never repeats one at a level that matters here.
                return {self._extern_string(): self._value() for _ in range(size)}
            if tag == EXTERN_STRING:
                return self._extern_at(size)
            raise ValueError("unhandled javabin tag 0x%02x" % tag_byte)

        if tag_byte == NULL:
            return None
        if tag_byte == BOOL_TRUE:
            return True
        if tag_byte == BOOL_FALSE:
            return False
        if tag_byte == BYTE:
            return int.from_bytes(self._in.read(1), "big", signed=True)
        if tag_byte == SHORT:
            return int.from_bytes(self._in.read(2), "big", signed=True)
        if tag_byte == DOUBLE:
            return struct.unpack(">d", self._in.read(8))[0]
        if tag_byte == INT:
            return int.from_bytes(self._in.read(4), "big", signed=True)
        if tag_byte == LONG:
            return int.from_bytes(self._in.read(8), "big", signed=True)
        if tag_byte == FLOAT:
            return struct.unpack(">f", self._in.read(4))[0]
        if tag_byte == DATE:
            millis = int.from_bytes(self._in.read(8), "big", signed=True)
            return datetime.datetime.fromtimestamp(millis / 1000.0, datetime.timezone.utc)
        if tag_byte == MAP:
            size = self._vint()
            return {self._value(): self._value() for _ in range(size)}
        if tag_byte in (SOLRDOC, SOLRINPUTDOC):
            return self._value()
        if tag_byte == SOLRDOCLST:
            # A document list is a three-element header - numFound, start, maxScore - then the docs.
            header = self._value()
            docs = self._value()
            return {"numFound": header[0], "start": header[1], "maxScore": header[2], "docs": docs}
        if tag_byte == BYTEARR:
            return self._in.read(self._vint())
        if tag_byte == ITERATOR:
            items = []
            while True:
                mark = self._in.tell()
                if self._byte() == END:
                    return items
                self._in.seek(mark)
                items.append(self._value())
        if tag_byte == END:
            return None
        raise ValueError("unhandled javabin tag 0x%02x" % tag_byte)

    def _extern_string(self):
        tag_byte = self._byte()
        if tag_byte & 0xE0 != EXTERN_STRING:
            self._in.seek(-1, io.SEEK_CUR)
            return self._value()
        return self._extern_at(self._size(tag_byte, EXTERN_STRING))

    def _extern_at(self, index):
        if index == 0:
            # A first occurrence carries the string, which then joins the table.
            value = self._value()
            self._extern.append(value)
            return value
        return self._extern[index - 1]


def decode_response(data):
    """Decode a javabin response body. One reader per response."""
    return JavaBinReader(data).read()
