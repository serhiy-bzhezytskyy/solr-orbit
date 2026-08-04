# SPDX-License-Identifier: Apache-2.0
#
# Modifications by Apache Solr contributors; see git log for details.
# Licensed under the Apache License, Version 2.0.
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.
# Licensed to Elasticsearch B.V. under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import asyncio
from unittest import TestCase

import pytest

from solrorbit import client
from solrorbit.utils import net
from tests import run_async


class RequestContextManagerTests(TestCase):
    @pytest.mark.skip(reason="latency is system-dependent")
    @run_async
    async def test_propagates_nested_context(self):
        test_client = client.RequestContextHolder()
        async with test_client.new_request_context() as top_level_ctx:
            test_client.on_request_start()
            await asyncio.sleep(0.1)
            async with test_client.new_request_context() as nested_ctx:
                test_client.on_request_start()
                await asyncio.sleep(0.1)
                test_client.on_request_end()
                nested_duration = nested_ctx.request_end - nested_ctx.request_start
            test_client.on_request_end()
            top_level_duration = top_level_ctx.request_end - top_level_ctx.request_start

        # top level request should cover total duration
        self.assertAlmostEqual(top_level_duration, 0.2, delta=0.05)
        # nested request should only cover nested duration
        self.assertAlmostEqual(nested_duration, 0.1, delta=0.05)

class TlsVerifyTests(TestCase):
    def test_defaults_to_the_resolved_ca_bundle(self):
        self.assertEqual(net.ca_bundle_path(), client._tls_verify())

    def test_ca_certs_names_the_bundle_to_trust(self):
        self.assertEqual("/etc/ssl/private-ca.pem",
                         client._tls_verify(ca_certs="/etc/ssl/private-ca.pem"))

    def test_verify_certs_false_disables_verification(self):
        self.assertFalse(client._tls_verify(verify_certs=False))

    def test_verify_certs_false_wins_over_ca_certs(self):
        self.assertFalse(client._tls_verify(ca_certs="/etc/ssl/private-ca.pem", verify_certs=False))


class ClientTlsOptionsTests(TestCase):
    def test_admin_session_carries_the_ca_bundle(self):
        admin = client.SolrAdminClient(host="localhost", port=8983, tls=True,
                                       ca_certs="/etc/ssl/private-ca.pem")
        session = admin._get_session()

        self.assertEqual("/etc/ssl/private-ca.pem", session.verify)
        # trust_env stays off: it is what makes the session fork-safe on macOS
        self.assertFalse(session.trust_env)

    def test_admin_session_can_disable_verification(self):
        admin = client.SolrAdminClient(host="localhost", port=8983, tls=True, verify_certs=False)

        self.assertFalse(admin._get_session().verify)

    def test_pysolr_session_carries_the_ca_bundle(self):
        sc = client.SolrClient(host="localhost", port=8983, tls=True,
                               ca_certs="/etc/ssl/private-ca.pem")
        solr = sc._get_pysolr("test")

        self.assertEqual("/etc/ssl/private-ca.pem", solr.session.verify)

    def test_factory_passes_the_client_options_through(self):
        factory = client.ClientFactory([{"host": "localhost", "port": 8983}],
                                       {"use_ssl": True, "ca_certs": "/etc/ssl/private-ca.pem"})
        sc = factory.create()

        self.assertEqual("/etc/ssl/private-ca.pem", sc._get_pysolr("test").session.verify)
        self.assertEqual("/etc/ssl/private-ca.pem", sc._admin._get_session().verify)

    def test_factory_defaults_to_verifying(self):
        factory = client.ClientFactory([{"host": "localhost", "port": 8983}], {})
        sc = factory.create()

        self.assertEqual(net.ca_bundle_path(), sc._admin._get_session().verify)
