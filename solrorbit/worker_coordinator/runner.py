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
import contextvars
import json
import logging
import re
import sys
import time
import urllib.parse
import types
from io import BytesIO
from typing import List

import ijson
import pysolr
import requests

from solrorbit import exceptions, workload
from solrorbit.client import RequestContextHolder, CollectionAlreadyExistsError, CollectionNotFoundError
from solrorbit.telemetry import _parse_prometheus_text
from solrorbit.utils.javabin import encode_update_request

__RUNNERS = {}


def register_default_runners():
    # Engine-agnostic operations
    register_runner(workload.OperationType.Sleep, Sleep(), async_runner=True)
    register_runner(workload.OperationType.Composite, Composite(), async_runner=True)
    # Backup and restore. Registered under both spellings: OperationType renders these as
    # "create-backup" and the like, while a workload converted from OpenSearch declares
    # "create-snapshot", and runner_for() is called with the workload's own string.
    _backup_runners = {
        workload.OperationType.CreateBackup: (CreateBackup(), "create-snapshot"),
        workload.OperationType.RestoreBackup: (RestoreBackup(), "restore-snapshot"),
        workload.OperationType.DeleteBackup: (Retry(DeleteBackup()), "delete-snapshot"),
        workload.OperationType.DeleteBackupRepository:
            (Retry(DeleteBackupRepository()), "delete-snapshot-repository"),
        workload.OperationType.CreateBackupRepository:
            (Retry(CreateBackupRepository()), "create-snapshot-repository"),
        workload.OperationType.WaitForBackupCreate:
            (Retry(WaitForBackupCreate()), "wait-for-snapshot-create"),
    }
    for operation_type, (backup_runner, snapshot_alias) in _backup_runners.items():
        register_runner(operation_type, backup_runner, async_runner=True)
        register_runner(snapshot_alias, backup_runner, async_runner=True)
    # Solr-native runners
    register_runner("bulk-index", SolrBulkIndex(), async_runner=True)
    register_runner("search", SolrSearch(), async_runner=True)
    register_runner("commit", SolrCommit(), async_runner=True)
    register_runner("refresh", SolrCommit(), async_runner=True)
    register_runner("optimize", SolrOptimize(), async_runner=True)
    register_runner("wait-for-merges", SolrWaitForMerges(), async_runner=True)
    register_runner("create-collection", SolrCreateCollection(), async_runner=True)
    register_runner("delete-collection", SolrDeleteCollection(), async_runner=True)
    register_runner("create-alias", SolrCreateAlias(), async_runner=True)
    register_runner("delete-alias", SolrDeleteAlias(), async_runner=True)
    # A binary indexing transport, which is what a workload's gRPC operations are measuring. Solr has
    # no gRPC endpoint; its comparable formats are javabin and CBOR on the same /update handler. The
    # converted spelling is kept so an OpenSearch workload loads unchanged.
    _binary_bulk = SolrBinaryBulkIndex()
    register_runner("binary-bulk-index", _binary_bulk, async_runner=True)
    register_runner("proto-bulk", _binary_bulk, async_runner=True)
    register_runner("raw-request", RawRequest(), async_runner=True)
    _paginated_runner = SolrPaginatedSearch()
    register_runner("paginated-search", _paginated_runner, async_runner=True)
    register_runner("scroll-search", _paginated_runner, async_runner=True)

def runner_for(operation_type):
    try:
        return __RUNNERS[operation_type]
    except KeyError:
        raise exceptions.BenchmarkError("No runner available for operation type [%s]" % operation_type)


def enable_assertions(enabled):
    """
    Changes whether assertions are enabled. The status changes for all tasks that are executed after this call.

    :param enabled: ``True`` to enable assertions, ``False`` to disable them.
    """
    AssertingRunner.assertions_enabled = enabled


def register_runner(operation_type, runner, **kwargs):
    logger = logging.getLogger(__name__)
    async_runner = kwargs.get("async_runner", False)
    if isinstance(operation_type, workload.OperationType):
        operation_type = operation_type.to_hyphenated_string()

    if not async_runner:
        raise exceptions.BenchmarkAssertionError(
            "Runner [{}] must be implemented as async runner and registered with async_runner=True.".format(str(runner)))

    if getattr(runner, "multi_cluster", False):
        if "__aenter__" in dir(runner) and "__aexit__" in dir(runner):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Registering runner object [%s] for [%s].", str(runner), str(operation_type))
            cluster_aware_runner = _multi_cluster_runner(runner, str(runner), context_manager_enabled=True)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Registering context-manager capable runner object [%s] for [%s].", str(runner), str(operation_type))
            cluster_aware_runner = _multi_cluster_runner(runner, str(runner))
    # we'd rather use callable() but this will erroneously also classify a class as callable...
    elif isinstance(runner, types.FunctionType):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Registering runner function [%s] for [%s].", str(runner), str(operation_type))
        cluster_aware_runner = _single_cluster_runner(runner, runner.__name__)
    elif "__aenter__" in dir(runner) and "__aexit__" in dir(runner):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Registering context-manager capable runner object [%s] for [%s].", str(runner), str(operation_type))
        cluster_aware_runner = _single_cluster_runner(runner, str(runner), context_manager_enabled=True)
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Registering runner object [%s] for [%s].", str(runner), str(operation_type))
        cluster_aware_runner = _single_cluster_runner(runner, str(runner))

    __RUNNERS[operation_type] = _with_completion(_with_assertions(cluster_aware_runner))

# Only intended for unit-testing!
def remove_runner(operation_type):
    del __RUNNERS[operation_type]


class Runner:
    """
    Base class for all operations against a search cluster.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        return self

    async def __call__(self, client, params):
        """
        Runs the actual method that should be benchmarked.

        :param args: All arguments that are needed to call this method.
        :return: A pair of (int, String). The first component indicates the "weight" of this call. it is typically 1 but for bulk operations
                 it should be the actual bulk size. The second component is the "unit" of weight which should be "ops" (short for
                 "operations") by default. If applicable, the unit should always be in plural form. It is used in metrics records
                 for throughput and results. A value will then be shown as e.g. "111 ops/s".
        """
        raise NotImplementedError("abstract operation")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def _default_kw_params(self, params):
        # map of API kwargs to Solr Orbit config parameters
        kw_dict = {
            "body": "body",
            "headers": "headers",
            "index": "index",
            "opaque_id": "opaque-id",
            "params": "request-params",
            "request_timeout": "request-timeout",
        }
        full_result =  {k: params.get(v) for (k, v) in kw_dict.items()}
        # filter Nones
        return dict(filter(lambda kv: kv[1] is not None, full_result.items()))

    def _transport_request_params(self, params):
        request_params = params.get("request-params", {})
        request_timeout = params.get("request-timeout")
        if request_timeout is not None:
            request_params["request_timeout"] = request_timeout
        headers = params.get("headers") or {}
        opaque_id = params.get("opaque-id")
        if opaque_id is not None:
            headers.update({"x-opaque-id": opaque_id})
        return request_params, headers

request_context_holder = RequestContextHolder()

def time_func(func):
    async def advised(*args, **kwargs):
        request_context_holder.on_client_request_start()
        try:
            response = await func(*args, **kwargs)
            return response
        finally:
            request_context_holder.on_client_request_end()
    return advised


class Delegator:
    """
    Mixin to unify delegate handling
    """
    def __init__(self, delegate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delegate = delegate


def unwrap(runner):
    """
    Unwraps all delegators until the actual runner.

    :param runner: An arbitrarily nested chain of delegators around a runner.
    :return: The innermost runner.
    """
    delegate = getattr(runner, "delegate", None)
    if delegate:
        return unwrap(delegate)
    else:
        return runner


def _single_cluster_runner(runnable, name, context_manager_enabled=False):
    # only pass the default client
    return MultiClientRunner(runnable, name, lambda client: client["default"], context_manager_enabled)


def _multi_cluster_runner(runnable, name, context_manager_enabled=False):
    # pass all clients
    return MultiClientRunner(runnable, name, lambda client: client, context_manager_enabled)


def _with_assertions(delegate):
    return AssertingRunner(delegate)


def _with_completion(delegate):
    unwrapped_runner = unwrap(delegate)
    if hasattr(unwrapped_runner, "completed") and hasattr(unwrapped_runner, "task_progress"):
        return WithCompletion(delegate, unwrapped_runner)
    else:
        return NoCompletion(delegate)


class NoCompletion(Runner, Delegator):
    def __init__(self, delegate):
        super().__init__(delegate=delegate)

    @property
    def completed(self):
        return None

    @property
    def task_progress(self):
        return None

    async def __call__(self, *args):
        return await self.delegate(*args)

    def __repr__(self, *args, **kwargs):
        return repr(self.delegate)

    async def __aenter__(self):
        await self.delegate.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)


class WithCompletion(Runner, Delegator):
    def __init__(self, delegate, progressable):
        super().__init__(delegate=delegate)
        self.progressable = progressable

    @property
    def completed(self):
        return self.progressable.completed

    @property
    def task_progress(self):
        return self.progressable.task_progress

    async def __call__(self, *args):
        return await self.delegate(*args)

    def __repr__(self, *args, **kwargs):
        return repr(self.delegate)

    async def __aenter__(self):
        await self.delegate.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)


class MultiClientRunner(Runner, Delegator):
    def __init__(self, runnable, name, client_extractor, context_manager_enabled=False):
        super().__init__(delegate=runnable)
        self.name = name
        self.client_extractor = client_extractor
        self.context_manager_enabled = context_manager_enabled

    async def __call__(self, *args):
        return await self.delegate(self.client_extractor(args[0]), *args[1:])

    def __repr__(self, *args, **kwargs):
        if self.context_manager_enabled:
            return "user-defined context-manager enabled runner for [%s]" % self.name
        else:
            return "user-defined runner for [%s]" % self.name

    async def __aenter__(self):
        if self.context_manager_enabled:
            await self.delegate.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.context_manager_enabled:
            return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)
        else:
            return False


class AssertingRunner(Runner, Delegator):
    assertions_enabled = False

    def __init__(self, delegate):
        super().__init__(delegate=delegate)
        self.predicates = {
            ">": self.greater_than,
            ">=": self.greater_than_or_equal,
            "<": self.smaller_than,
            "<=": self.smaller_than_or_equal,
            "==": self.equal,
        }

    def greater_than(self, expected, actual):
        return actual > expected

    def greater_than_or_equal(self, expected, actual):
        return actual >= expected

    def smaller_than(self, expected, actual):
        return actual < expected

    def smaller_than_or_equal(self, expected, actual):
        return actual <= expected

    def equal(self, expected, actual):
        return actual == expected

    def check_assertion(self, op_name, assertion, properties):
        path = assertion["property"]
        predicate_name = assertion["condition"]
        expected_value = assertion["value"]
        actual_value = properties
        for k in path.split("."):
            actual_value = actual_value[k]
        predicate = self.predicates[predicate_name]
        success = predicate(expected_value, actual_value)
        if not success:
            if op_name:
                msg = f"Expected [{path}] in [{op_name}] to be {predicate_name} [{expected_value}] but was [{actual_value}]."
            else:
                msg = f"Expected [{path}] to be {predicate_name} [{expected_value}] but was [{actual_value}]."

            raise exceptions.BenchmarkTaskAssertionError(msg)

    async def __call__(self, *args):
        params = args[1]
        return_value = await self.delegate(*args)
        if AssertingRunner.assertions_enabled and "assertions" in params:
            op_name = params.get("name")
            if isinstance(return_value, dict):
                for assertion in params["assertions"]:
                    self.check_assertion(op_name, assertion, return_value)
            else:
                self.logger.debug("Skipping assertion check in [%s] as [%s] does not return a dict.",
                                  op_name, repr(self.delegate))
        return return_value

    def __repr__(self, *args, **kwargs):
        return repr(self.delegate)

    async def __aenter__(self):
        await self.delegate.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)


def mandatory(params, key, op):
    try:
        return params[key]
    except KeyError:
        raise exceptions.DataError(
            f"Parameter source for operation '{str(op)}' did not provide the mandatory parameter '{key}'. "
            f"Add it to your parameter source and try again.")



def escape(v):
    """
    Escapes values so they can be used as query parameters

    :param v: The raw value. May be None.
    :return: The escaped value.
    """
    if v is None:
        return None
    elif isinstance(v, bool):
        return str(v).lower()
    else:
        return str(v)


def parse(text: BytesIO, props: List[str], lists: List[str] = None) -> dict:
    """
    Selectively parse the provided text as JSON extracting only the properties provided in ``props``. If ``lists`` is
    specified, this function determines whether the provided lists are empty (respective value will be ``True``) or
    contain elements (respective key will be ``False``).

    :param text: A text to parse.
    :param props: A mandatory list of property paths (separated by a dot character) for which to extract values.
    :param lists: An optional list of property paths to JSON lists in the provided text.
    :return: A dict containing all properties and lists that have been found in the provided text.
    """
    text.seek(0)
    parser = ijson.parse(text)
    parsed = {}
    parsed_lists = {}
    current_list = None
    expect_end_array = False
    try:
        for prefix, event, value in parser:
            if expect_end_array:
                # True if the list is empty, False otherwise
                parsed_lists[current_list] = event == "end_array"
                expect_end_array = False
            if prefix in props:
                parsed[prefix] = value
            elif lists is not None and prefix in lists and event == "start_array":
                current_list = prefix
                expect_end_array = True
            # found all necessary properties
            if len(parsed) == len(props) and (lists is None or len(parsed_lists) == len(lists)):
                break
    except ijson.IncompleteJSONError:
        # did not find all properties
        pass

    parsed.update(parsed_lists)
    return parsed



class Sleep(Runner):
    """
    Sleeps for the specified duration not issuing any request.
    """
    @time_func
    async def __call__(self, client, params):
        sleep_duration = mandatory(params, "duration", "sleep")
        client.on_request_start()
        try:
            await asyncio.sleep(sleep_duration)
        finally:
            client.on_request_end()

    def __repr__(self, *args, **kwargs):
        return "sleep"


# ---------------------------------------------------------------------------
# Runners: backup and restore
#
# These carry the operation names of OpenSearch's snapshot operations, so a converted workload
# keeps working, and map them onto Solr's backup and restore APIs:
#
#   create-snapshot            POST /api/collections/<collection>/backups/<name>/versions
#   wait-for-snapshot-create   action=LISTBACKUP
#   restore-snapshot           action=RESTORE
#   delete-snapshot            action=DELETEBACKUP
#
# Solr has no API that creates a named repository: repositories are declared in solr.xml with a
# <repository> element and selected per call with the "repository" parameter. The two
# repository operations therefore validate their configuration instead of issuing a request.
# ---------------------------------------------------------------------------

_BACKUP_POLL_INTERVAL = 1.0


def _backup_location(params):
    """Where the backup lives. Solr requires a path the node is allowed to write to."""
    location = params.get("location") or params.get("path")
    if not location:
        raise exceptions.DataError(
            "Operation parameter 'location' is missing. Solr takes the backup destination per "
            "call: give the operation a 'location' the node is allowed to write to, or set "
            "solr.allowPaths. A named repository declared in solr.xml can be selected with "
            "'repository' instead.")
    return location


def _backup_query(params, **extra):
    """Common query parameters for the collections API, dropping the ones left unset."""
    query = {"name": params.get("snapshot") or params.get("name"), **extra}
    if params.get("repository"):
        query["repository"] = params["repository"]
    else:
        query["location"] = _backup_location(params)
    if not query["name"]:
        raise exceptions.DataError("Operation parameter 'snapshot' is missing.")
    return {k: v for k, v in query.items() if v is not None}


async def _await_collections_api(client, request_id, timeout):
    """
    Poll REQUESTSTATUS until an asynchronous collections-API call settles.

    Backup and restore of a real corpus outlast an HTTP request, so they are submitted with async
    and their outcome is collected here. A failure has to be raised rather than returned: the
    collections API answers 200 with a body describing the failure.
    """
    deadline = time.perf_counter() + timeout
    while True:
        response = await _run_in_executor(
            client.raw_request, "GET",
            "/solr/admin/collections?action=REQUESTSTATUS&requestid=%s" % request_id, None, None)
        response.raise_for_status()
        payload = response.json()
        state = (payload.get("status") or {}).get("state")
        if state == "completed":
            return payload
        if state == "failed":
            raise exceptions.BenchmarkError(
                "Collections API request [%s] failed: %s" % (request_id, payload.get("failure")))
        if time.perf_counter() > deadline:
            raise exceptions.BenchmarkError(
                "Collections API request [%s] did not finish within %s seconds; last state [%s]"
                % (request_id, timeout, state))
        await asyncio.sleep(_BACKUP_POLL_INTERVAL)


class CreateBackupRepository(Runner):
    """
    Check that a backup repository is usable.

    Solr has no API for creating one: a repository is a <repository> element in solr.xml, read at
    node startup, and selected per call with the "repository" parameter. An operation naming a
    repository is accepted if the cluster knows it; an operation carrying only a "location" is
    accepted as-is, since Solr takes a destination path per call.
    """
    async def __call__(self, client, params):
        repository = params.get("repository")
        if not repository:
            _backup_location(params)
            return {"weight": 1, "unit": "ops", "success": True}
        response = await _run_in_executor(
            client.raw_request, "GET", "/api/cluster/backups/repositories", None, None)
        if response.status_code == 404:
            # Older Solr does not list repositories; the name is validated on first use instead.
            return {"weight": 1, "unit": "ops", "success": True}
        response.raise_for_status()
        known = [entry.get("name") for entry in (response.json().get("repositories") or [])]
        if known and repository not in known:
            raise exceptions.DataError(
                "Backup repository [%s] is not declared in solr.xml. Solr cannot create one at "
                "run time: add a <repository> element under <backup> and restart the node. "
                "Known repositories: %s" % (repository, ", ".join(known)))
        return {"weight": 1, "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "create-snapshot-repository"


class DeleteBackupRepository(Runner):
    """
    Accept the removal of a backup repository.

    Nothing is deleted: a Solr repository is configuration, not cluster state, so removing it means
    editing solr.xml. Deleting the backups themselves is delete-snapshot.
    """
    async def __call__(self, client, params):
        return {"weight": 1, "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "delete-snapshot-repository"


class CreateBackup(Runner):
    """
    Back a collection up.

    Params:
      - ``collection``, ``snapshot`` (the backup name)
      - ``location`` or ``repository`` — where it goes
      - ``request-timeout`` (default 3600) — how long to wait for the asynchronous call
    """
    async def __call__(self, client, params):
        collection = _get_collection(params)
        name = params.get("snapshot") or params.get("name")
        if not name:
            raise exceptions.DataError("Operation parameter 'snapshot' is missing.")
        body = {"async": "backup-%s" % name}
        if params.get("repository"):
            body["repository"] = params["repository"]
        else:
            body["location"] = _backup_location(params)

        response = await _run_in_executor(
            client.raw_request, "POST",
            "/api/collections/%s/backups/%s/versions" % (collection, name), body, None)
        response.raise_for_status()
        request_id = response.json().get("requestid", body["async"])
        payload = await _await_collections_api(client, request_id,
                                              params.get("request-timeout", 3600))
        shard = next(iter((payload.get("success") or {}).values()), {}).get("response", {})
        return {
            "weight": 1,
            "unit": "ops",
            "success": True,
            "index-file-count": shard.get("indexFileCount"),
            "index-size-mb": shard.get("indexSizeMB"),
        }

    def __repr__(self, *args, **kwargs):
        return "create-snapshot"


class WaitForBackupCreate(Runner):
    """
    Wait until a backup is listed, and report what it holds.

    CreateBackup already waits for its own asynchronous call, so this confirms the result is
    readable rather than repeating the wait.
    """
    async def __call__(self, client, params):
        query = _backup_query(params)
        collection = params.get("collection") or params.get("index")
        if collection:
            query["collection"] = collection
        deadline = time.perf_counter() + params.get("request-timeout", 3600)
        while True:
            response = await _run_in_executor(
                client.raw_request, "GET",
                "/solr/admin/collections?action=LISTBACKUP&" + urllib.parse.urlencode(query),
                None, None)
            if response.status_code == 200:
                backups = response.json().get("backups") or []
                if backups:
                    newest = backups[-1]
                    return {
                        "weight": 1,
                        "unit": "ops",
                        "success": True,
                        "backup-id": newest.get("backupId"),
                        "index-file-count": newest.get("indexFileCount"),
                        "index-size-mb": newest.get("indexSizeMB"),
                    }
            if time.perf_counter() > deadline:
                raise exceptions.BenchmarkError(
                    "Backup [%s] was not listed within the timeout" % query["name"])
            await asyncio.sleep(_BACKUP_POLL_INTERVAL)

    def __repr__(self, *args, **kwargs):
        return "wait-for-snapshot-create"


class DeleteBackup(Runner):
    """
    Delete a backup.

    Params:
      - ``snapshot`` (the backup name), ``location`` or ``repository``
      - ``backup-id`` — delete this backup point; defaults to every one that exists
      - ``max-num-backup-points`` — instead, keep this many and delete the rest

    A backup name holds one or more backup points, each with its own id, and DELETEBACKUP wants to
    be told which: backupId for one, maxNumBackupPoints to keep the newest few, or purgeUnused,
    which only drops index files no remaining backup point references. Passing purgeUnused alone
    therefore reports success while leaving the backup in place, so when no id is given every
    backup point is listed and deleted.
    """
    async def __call__(self, client, params):
        base = _backup_query(params)
        collection = params.get("collection") or params.get("index")
        if collection:
            base["collection"] = collection

        if params.get("max-num-backup-points") is not None:
            targets = [{"maxNumBackupPoints": params["max-num-backup-points"]}]
        elif params.get("backup-id") is not None:
            targets = [{"backupId": params["backup-id"]}]
        else:
            listing = await _run_in_executor(
                client.raw_request, "GET",
                "/solr/admin/collections?action=LISTBACKUP&" + urllib.parse.urlencode(base),
                None, None)
            listing.raise_for_status()
            targets = [{"backupId": entry["backupId"]}
                       for entry in (listing.json().get("backups") or [])]

        backup_ids = index_files = 0
        for target in targets:
            response = await _run_in_executor(
                client.raw_request, "GET",
                "/solr/admin/collections?action=DELETEBACKUP&"
                + urllib.parse.urlencode(dict(base, **target)), None, None)
            response.raise_for_status()
            # "deleted" is an object when deleting by maxNumBackupPoints or purgeUnused, and a list
            # of the removed backup points when deleting by backupId.
            deleted = response.json().get("deleted") or {}
            for entry in (deleted if isinstance(deleted, list) else [deleted]):
                backup_ids += entry.get("numBackupIds") or (1 if "backupId" in entry else 0)
                index_files += entry.get("numIndexFiles") or 0

        return {
            "weight": 1,
            "unit": "ops",
            "success": True,
            "deleted-backup-ids": backup_ids,
            "deleted-index-files": index_files,
        }

    def __repr__(self, *args, **kwargs):
        return "delete-snapshot"


class RestoreBackup(Runner):
    """
    Restore a backup into a collection.

    Params:
      - ``snapshot`` (the backup name), ``collection`` (the collection to restore into)
      - ``location`` or ``repository``
      - ``request-timeout`` (default 3600)

    Solr restores into a collection that does not exist yet, so a workload that restores over its
    own collection has to delete it first.
    """
    async def __call__(self, client, params):
        collection = _get_collection(params)
        query = _backup_query(params, collection=collection,
                              **{"async": "restore-%s" % collection})
        response = await _run_in_executor(
            client.raw_request, "GET",
            "/solr/admin/collections?action=RESTORE&" + urllib.parse.urlencode(query), None, None)
        response.raise_for_status()
        request_id = response.json().get("requestid", query["async"])
        await _await_collections_api(client, request_id, params.get("request-timeout", 3600))
        return {"weight": 1, "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "restore-snapshot"


class CompositeContext:
    ctx = contextvars.ContextVar("composite_context")

    def __init__(self):
        self.token = None

    async def __aenter__(self):
        self.token = CompositeContext.ctx.set({})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        CompositeContext.ctx.reset(self.token)
        return False

    @staticmethod
    def put(key, value):
        CompositeContext._ctx()[key] = value

    @staticmethod
    def get(key):
        try:
            return CompositeContext._ctx()[key]
        except KeyError:
            raise KeyError(f"Unknown property [{key}]. Currently recognized "
                           f"properties are [{', '.join(CompositeContext._ctx().keys())}].") from None

    @staticmethod
    def remove(key):
        try:
            CompositeContext._ctx().pop(key)
        except KeyError:
            raise KeyError(f"Unknown property [{key}]. Currently recognized "
                           f"properties are [{', '.join(CompositeContext._ctx().keys())}].") from None

    @staticmethod
    def _ctx():
        try:
            return CompositeContext.ctx.get()
        except LookupError:
            raise exceptions.BenchmarkAssertionError("This operation is only allowed inside a composite operation.") from None


class Composite(Runner):
    """
    Executes a complex request structure which is measured as one composite operation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_op_types = [
            # Framework-level operations
            "sleep",
            "raw-request",
            # Solr search operations
            "search",
            "paginated-search",
            "scroll-search",
            # Solr data operations
            "bulk-index",
            "commit",
            "refresh",
            "optimize",
            "wait-for-merges",
            # Solr admin operations
            "create-collection",
            "delete-collection",
        ]

    async def run_stream(self, client, stream, connection_limit):
        streams = []
        timings = []
        try:
            for item in stream:
                if "stream" in item:
                    streams.append(asyncio.create_task(self.run_stream(client, item["stream"], connection_limit)))
                elif "operation-type" in item:
                    # consume all prior streams first
                    if streams:
                        streams_timings = await asyncio.gather(*streams)
                        for stream_timings in streams_timings:
                            timings += stream_timings
                        streams = []
                    op_type = item["operation-type"]
                    if op_type not in self.supported_op_types:
                        raise exceptions.BenchmarkAssertionError(
                            f"Unsupported operation-type [{op_type}]. Use one of [{', '.join(self.supported_op_types)}].")
                    runner = RequestTiming(runner_for(op_type))
                    async with connection_limit:
                        async with runner:
                            response = await runner({"default": client}, item)
                            timing = response.get("dependent_timing") if response else None
                            if timing:
                                timings.append(timing)

                else:
                    raise exceptions.BenchmarkAssertionError("Requests structure must contain [stream] or [operation-type].")
        except BaseException:
            # stop all already created tasks in case of exceptions
            for s in streams:
                if not s.done():
                    s.cancel()
            raise

        # complete any outstanding streams
        if streams:
            streams_timings = await asyncio.gather(*streams)
            for stream_timings in streams_timings:
                timings += stream_timings
        return timings

    async def __call__(self, client, params):
        requests = mandatory(params, "requests", self)
        max_connections = params.get("max-connections", sys.maxsize)
        async with CompositeContext():
            response = await self.run_stream(client, requests, asyncio.BoundedSemaphore(max_connections))
        return {
            "weight": 1,
            "unit": "ops",
            "dependent_timing": response
        }

    def __repr__(self, *args, **kwargs):
        return "composite"


class RequestTiming(Runner, Delegator):
    def __init__(self, delegate):
        super().__init__(delegate=delegate)

    async def __aenter__(self):
        await self.delegate.__aenter__()
        return self

    async def __call__(self, client, params):
        absolute_time = time.time()
        async with client["default"].new_request_context() as request_context:
            return_value = await self.delegate(client, params)
            if isinstance(return_value, tuple) and len(return_value) == 2:
                total_ops, total_ops_unit = return_value
                result = {
                    "weight": total_ops,
                    "unit": total_ops_unit,
                    "success": True
                }
            elif isinstance(return_value, dict):
                result = return_value
            else:
                result = {
                    "weight": 1,
                    "unit": "ops",
                    "success": True
                }

            start = request_context.request_start
            end = request_context.request_end
            result["dependent_timing"] = {
                "operation": params.get("name"),
                "operation-type": params.get("operation-type"),
                "absolute_time": absolute_time,
                "request_start": start,
                "request_end": end,
                "service_time": end - start
            }
        return result

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)


# TODO: Allow to use this from (selected) regular runners and add user documentation.
# TODO: It would maybe be interesting to add meta-data on how many retries there were.
class Retry(Runner, Delegator):
    """
    This runner can be used as a wrapper around regular runners to retry operations.

    It defines the following parameters:

    * ``retries`` (optional, default 0): The number of times the operation is retried.
    * ``retry-until-success`` (optional, default False): Retries until the delegate returns a success. This will also
                              forcibly set ``retry-on-error`` to ``True``.
    * ``retry-wait-period`` (optional, default 0.5): The time in seconds to wait after an error.
    * ``retry-on-timeout`` (optional, default True): Whether to retry on connection timeout.
    * ``retry-on-error`` (optional, default False): Whether to retry on failure (i.e. the delegate
                         returns ``success == False``)
    """

    def __init__(self, delegate, retry_until_success=False):
        super().__init__(delegate=delegate)
        self.retry_until_success = retry_until_success

    async def __aenter__(self):
        await self.delegate.__aenter__()
        return self

    async def __call__(self, client, params):
        # pylint: disable=import-outside-toplevel
        import socket
        retry_until_success = params.get("retry-until-success", self.retry_until_success)
        if retry_until_success:
            max_attempts = sys.maxsize
            retry_on_error = True
        else:
            max_attempts = params.get("retries", 0) + 1
            retry_on_error = params.get("retry-on-error", False)
        sleep_time = params.get("retry-wait-period", 0.5)
        retry_on_timeout = params.get("retry-on-timeout", True)

        for attempt in range(max_attempts):
            last_attempt = attempt + 1 == max_attempts
            try:
                return_value = await self.delegate(client, params)
                if last_attempt or not retry_on_error:
                    return return_value
                # we can determine success if and only if the runner returns a dict. Otherwise, we have to assume it was fine.
                elif isinstance(return_value, dict):
                    if return_value.get("success", True):
                        self.logger.debug("%s has returned successfully", repr(self.delegate))
                        return return_value
                    else:
                        self.logger.info("[%s] has returned with an error: %s. Retrying in [%.2f] seconds.",
                                         repr(self.delegate), return_value, sleep_time)
                        await asyncio.sleep(sleep_time)
                else:
                    return return_value
            except Exception as e:
                if isinstance(e, (socket.timeout, exceptions.BenchmarkConnectionError)):
                    if last_attempt or not retry_on_timeout:
                        raise e
                    else:
                        await asyncio.sleep(sleep_time)
                elif isinstance(e, exceptions.BenchmarkTransportError):
                    if last_attempt or not retry_on_timeout:
                        raise e
                    elif e.status_code == 408:
                        self.logger.info("[%s] has timed out. Retrying in [%.2f] seconds.", repr(self.delegate), sleep_time)
                        await asyncio.sleep(sleep_time)
                    else:
                        raise e
                else:
                    raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.delegate.__aexit__(exc_type, exc_val, exc_tb)

    def __repr__(self, *args, **kwargs):
        return "retryable %s" % repr(self.delegate)


# ===========================================================================
# Solr-specific runners
# ===========================================================================

# ---------------------------------------------------------------------------
# Error translation helpers
# ---------------------------------------------------------------------------

def _translate_solr_error(e):
    """Translate a pysolr or requests exception to a BenchmarkTransportError."""
    if isinstance(e, requests.exceptions.ConnectionError):
        return exceptions.BenchmarkConnectionError(str(e), cause=e)
    if isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectTimeout)):
        return exceptions.BenchmarkConnectionTimeout(str(e), cause=e)
    if isinstance(e, requests.exceptions.HTTPError):
        status_code = e.response.status_code if e.response is not None else None
        if status_code == 404:
            return exceptions.BenchmarkNotFoundError(str(e), cause=e)
        return exceptions.BenchmarkTransportError(
            str(e), cause=e, status_code=status_code,
            error=f"HTTP {status_code}", info=str(e))
    if isinstance(e, pysolr.SolrError):
        msg = str(e)
        status_code = None
        for part in msg.split():
            if part.isdigit():
                code = int(part)
                if 100 <= code < 600:
                    status_code = code
                    break
        return exceptions.BenchmarkTransportError(
            msg, cause=e, status_code=status_code, error="SolrError", info=msg)
    return exceptions.BenchmarkTransportError(str(e), cause=e, error=type(e).__name__, info=str(e))


def _solr_runner_decorator(fn):
    """
    Decorator applied to every Solr runner's ``__call__``.

    Translates pysolr and requests exceptions to BenchmarkTransportError, and records the timing
    boundaries every runner has to report.

    It also unwraps the client mapping a composite operation passes down, so that the same runner
    serves both call shapes.

    Two nested pairs are needed, and both are required for a composite operation to be measurable.
    The outer pair brackets the whole call; the inner pair brackets the request itself.
    update_request_start() ignores its value unless a client_request_start has already been seen
    (context.py), so marking only the inner pair records nothing and RequestTiming then computes
    service_time from two Nones.
    """
    import functools

    @functools.wraps(fn)
    async def wrapper(self, client, *args, **kwargs):
        # A composite operation hands its steps the mapping {"default": client}; unwrap it so that
        # every runner sees the client itself and works inside and outside a composite alike.
        if isinstance(client, dict) and "default" in client:
            client = client["default"]
        # There is no request context when a runner is called outside one, e.g. from a test.
        try:
            request_context_holder.request_context.get()
            timed = True
        except LookupError:
            timed = False
        if timed:
            request_context_holder.on_client_request_start()
            request_context_holder.on_request_start()
        try:
            return await fn(self, client, *args, **kwargs)
        except exceptions.BenchmarkTransportError:
            raise
        except (pysolr.SolrError, requests.exceptions.RequestException) as e:
            raise _translate_solr_error(e) from e
        finally:
            if timed:
                request_context_holder.on_request_end()
                request_context_holder.on_client_request_end()
    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_collection(params):
    """Extract and validate the collection name from params."""
    collection = params.get("collection") or params.get("index") or None
    if not collection:
        raise exceptions.DataError(
            "Operation parameter 'collection' is missing. "
            "Make sure your Solr workload specifies a 'collection' name in the operation params or param source. "
            "If you are running an OpenSearch workload, convert it first with 'solr-orbit convert-workload'."
        )
    if collection == "_all":
        raise exceptions.DataError(
            "Operation targets collection '_all' which is an OpenSearch concept. "
            "Solr does not support querying all collections simultaneously. "
            "Remove or replace this operation in your Solr workload."
        )
    return collection


async def _run_in_executor(func, *args, **kwargs):
    """Run a blocking call in the default thread-pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _translate_ndjson_batch(lines):
    """
    Translate NDJSON to a list of Solr document dicts.

    Supports two formats:

    1. OpenSearch bulk format (action/document pairs):
       {"index": {"_id": "1", "_index": "coll"}}
       {"field": "value"}
       → Extracts _id from action line and sets it as "id" field in document

    2. Simple NDJSON (one document per line):
       {"field": "value"}
       {"field2": "value2"}
       → Each line is a document; no stable IDs unless "id" field is present

    Auto-detects format by checking if lines contain bulk action keys
    (index, create, update, delete).
    """
    docs = []
    it = iter(lines)

    first_line = None
    for line in it:
        line = line.strip()
        if line:
            first_line = line
            break

    if not first_line:
        return docs

    try:
        first_obj = json.loads(first_line)
    except json.JSONDecodeError:
        logging.getLogger(__name__).warning("Skipping malformed first line: %s", first_line)
        return docs

    has_action_keys = isinstance(first_obj, dict) and any(
        k in first_obj for k in ("index", "create", "update", "delete")
    )

    if has_action_keys:
        docs = _parse_bulk_pairs(first_line, it)
    else:
        if isinstance(first_obj, dict):
            docs.append(first_obj)
        for line in it:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    docs.append(obj)
            except json.JSONDecodeError as exc:
                logging.getLogger(__name__).warning("Skipping malformed NDJSON line: %s", exc)

    return docs


def _translate_ndjson_stream(lines):
    """
    Stream-translate NDJSON to Solr documents (generator version).

    Yields ``(document, target_collection)`` one pair at a time instead of loading all into memory,
    where *target_collection* is ``None`` unless the bulk action line named one. It has to be
    per-document: a workload can feed several collections from one corpus, and a single collection
    read from the operation params would send everything to whichever it names.

    Supports both OpenSearch bulk format and simple NDJSON.
    """
    _logger = logging.getLogger(__name__)
    it = iter(lines)

    first_line = None
    for line in it:
        line = line.strip()
        if line:
            first_line = line
            break

    if not first_line:
        return

    try:
        first_obj = json.loads(first_line)
    except json.JSONDecodeError:
        _logger.warning("Skipping malformed first line: %s", first_line)
        return

    has_action_keys = isinstance(first_obj, dict) and any(
        k in first_obj for k in ("index", "create", "update", "delete")
    )

    if has_action_keys:
        yield from _stream_bulk_pairs(first_line, it)
    else:
        # Plain NDJSON carries no action line, so there is no per-document target.
        if isinstance(first_obj, dict):
            if "id" not in first_obj:
                first_obj["id"] = str(hash(json.dumps(first_obj, sort_keys=True)))
            yield first_obj, None
        for line in it:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    if "id" not in obj:
                        obj["id"] = str(hash(json.dumps(obj, sort_keys=True)))
                    for key, value in list(obj.items()):
                        if isinstance(value, list) and len(value) == 2:
                            if all(isinstance(v, (int, float)) for v in value):
                                obj[key] = f"{value[1]},{value[0]}"
                    yield obj, None
            except json.JSONDecodeError as exc:
                _logger.warning("Skipping malformed NDJSON line: %s", exc)


def _stream_bulk_pairs(first_action_line, lines_iter):
    """Stream-parse OpenSearch bulk format (generator version)."""
    _logger = logging.getLogger(__name__)
    action_line = first_action_line

    while action_line:
        doc_line = next(lines_iter, None)
        if doc_line is None:
            break
        doc_line = doc_line.strip()
        if not doc_line:
            action_line = next(lines_iter, "").strip()
            continue

        try:
            action = json.loads(action_line)
            doc = json.loads(doc_line)
        except json.JSONDecodeError as exc:
            _logger.warning("Skipping malformed NDJSON pair: %s", exc)
            action_line = next(lines_iter, "").strip()
            continue

        if not isinstance(action, dict) or not isinstance(doc, dict):
            _logger.warning("Skipping non-dict action/doc pair")
            action_line = next(lines_iter, "").strip()
            continue

        id_found = False
        target = None
        for key in ("index", "create", "update", "delete"):
            if key in action:
                meta = action[key]
                if isinstance(meta, dict):
                    if "_id" in meta:
                        doc["id"] = meta["_id"]
                        id_found = True
                    # The action line names the target, which matters when one corpus feeds several
                    # collections: http_logs has a document set per month, each with its own
                    # target-collection, and a single collection taken from the operation params
                    # would send all 247M documents to the first one.
                    target = meta.get("_index") or meta.get("_collection")
                break

        if not id_found:
            doc["id"] = str(abs(hash(json.dumps(doc, sort_keys=True))))

        doc = _flatten_document(doc)

        for key, value in list(doc.items()):
            if isinstance(value, list) and len(value) == 2:
                if all(isinstance(v, (int, float)) for v in value):
                    doc[key] = f"{value[1]},{value[0]}"
            elif isinstance(value, str) and len(value) == 19 and value[10] in (' ', 'T'):
                # A timestamp with no zone. OpenSearch reads it as UTC; Solr's date field requires the
                # zone and rejects the value outright, so every document would fail. pmc writes the
                # space-separated form and noaa the T-separated one; both mean the same instant.
                if value[4] == '-' and value[7] == '-' and value[13] == ':' and value[16] == ':':
                    doc[key] = value.replace(' ', 'T') + 'Z'

        yield doc, target
        action_line = next(lines_iter, "").strip()


def _flatten_document(doc, prefix="", separator="_"):
    """
    Flatten a nested document, joining the path with an underscore.

    Solr documents are flat, so a corpus that nests — noaa carries an eight-field ``station`` object
    with a ``location`` object inside it — cannot be sent as it stands. The name a nested value takes
    is its path joined by underscores, ``station_location_lat``, which is what field-name
    normalisation produces on the query side for ``station.location.lat``, so the two agree.

    A geographic point (an object of exactly ``lat`` and ``lon``) is also emitted as one
    ``"lat,lon"`` string under the parent's own name, since that is what a Solr spatial field takes.
    The components are kept as well: an RPT field cannot expose them as a ValueSource.

    A list of objects is left alone. Solr's answer to that is a child document, which is a different
    shape than flattening, and no workload ported so far has one.
    """
    if not isinstance(doc, dict):
        return doc

    out = {}
    for key, value in doc.items():
        # A key may itself contain dots. big5 writes a literal "aws.cloudwatch" key holding an object,
        # so the path is normalised the way the query side normalises a field name — every dot becomes
        # an underscore — or the documents would carry aws.cloudwatch_log_stream while the operations
        # ask for aws_cloudwatch_log_stream.
        key = str(key).replace(".", separator)
        name = "%s%s%s" % (prefix, separator, key) if prefix else key
        if isinstance(value, dict):
            keys = set(value)
            if keys == {"lat", "lon"}:
                out[name] = "%s,%s" % (value["lat"], value["lon"])
            for sub_name, sub_value in _flatten_document(value, name, separator).items():
                out[sub_name] = sub_value
        else:
            out[name] = value
    return out


def _parse_bulk_pairs(first_action_line, lines_iter):
    """Parse OpenSearch bulk format (alternating action/doc pairs)."""
    _logger = logging.getLogger(__name__)
    docs = []
    action_line = first_action_line

    while action_line:
        doc_line = next(lines_iter, None)
        if doc_line is None:
            break
        doc_line = doc_line.strip()
        if not doc_line:
            action_line = next(lines_iter, "").strip()
            continue

        try:
            action = json.loads(action_line)
            doc = json.loads(doc_line)
        except json.JSONDecodeError as exc:
            _logger.warning("Skipping malformed NDJSON pair: %s", exc)
            action_line = next(lines_iter, "").strip()
            continue

        if not isinstance(action, dict) or not isinstance(doc, dict):
            _logger.warning("Skipping non-dict action/doc pair")
            action_line = next(lines_iter, "").strip()
            continue

        meta = {}
        for key in ("index", "create", "update", "delete"):
            if key in action:
                meta = action[key]
                break

        doc_id = meta.get("_id")
        if doc_id is not None:
            doc["id"] = doc_id

        routing_collection = meta.get("_index")
        if routing_collection:
            _logger.debug("NDJSON _index='%s' (routing only, not stored)", routing_collection)

        docs.append(doc)
        action_line = next(lines_iter, "").strip()

    return docs


# ---------------------------------------------------------------------------
# Base runner with automatic error translation
# ---------------------------------------------------------------------------

class SolrRunner(Runner):
    """Base class for all Solr runners.

    Wraps ``__call__`` so that pysolr and requests exceptions are automatically
    translated to ``BenchmarkTransportError`` subclasses before they reach the
    worker_coordinator framework.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            cls.__call__ = _solr_runner_decorator(cls.__call__)


# ---------------------------------------------------------------------------
# Runner: bulk-index
# ---------------------------------------------------------------------------

class SolrBulkIndex(SolrRunner):
    """
    Index documents from an NDJSON corpus into Solr.

    Params:
      - ``collection``, ``bulk-size`` (default 500), ``commit`` (default False)
      - ``body`` or ``corpus`` — NDJSON line pairs (action + document) or bytes blob
    """

    async def __call__(self, client, params):
        body = params.get("body", params.get("corpus", []))
        if isinstance(body, bytes):
            corpus_lines = [line.decode("utf-8") for line in body.split(b"\n") if line]
        elif isinstance(body, str):
            corpus_lines = [line for line in body.split("\n") if line]
        else:
            corpus_lines = body

        batch_size = params.get("bulk-size", 500)
        do_commit = params.get("commit", False)
        # A corpus whose document sets name their own target-collection supplies it per document, so
        # the operation need not carry one; the error is raised only if neither source has it.
        collection = params.get("collection") or params.get("index")
        # An operation may route its documents through a named update chain — what an OpenSearch
        # workload calls an ingest pipeline. The two spellings are both accepted so a converted
        # workload loads unchanged.
        chain = params.get("update-chain") or params.get("pipeline")
        sc = client

        doc_stream = _translate_ndjson_stream(corpus_lines)
        total_docs = 0
        errors = 0

        start = time.perf_counter()

        # Batched per target: a corpus may name a collection per document set, as http_logs does with
        # one per month, and sending those to a single collection would put every document in the
        # first one. A document with no target named falls back to the operation's collection.
        batches = {}

        async def flush(target, batch, final=False):
            nonlocal total_docs, errors
            try:
                if chain:
                    # Not through pysolr: it builds the path as handler + "/" + "?commit=…", so a
                    # handler carrying a query string comes out as "update?update.chain=x/?commit=true"
                    # and Solr reports 'unknown UpdateRequestProcessorChain: x/'.
                    path = "/solr/%s/update?%s" % (
                        urllib.parse.quote(target),
                        urllib.parse.urlencode({"update.chain": chain, "commitWithin": 1000}))
                    resp = await _run_in_executor(
                        sc.raw_request, "POST", path, json.dumps(batch),
                        {"Content-type": "application/json"})
                    if resp.status_code >= 400:
                        raise exceptions.BenchmarkError(
                            "Update through chain '%s' returned %d: %s"
                            % (chain, resp.status_code, resp.text[:200]))
                else:
                    await _run_in_executor(sc.add, target, batch, commit=False, commitWithin=1000)
                total_docs += len(batch)
            except (pysolr.SolrError, exceptions.BenchmarkError) as exc:
                logging.getLogger(__name__).error(
                    "Bulk index error on %sbatch for '%s': %s", "final " if final else "", target, exc)
                errors += len(batch)

        for doc, target in doc_stream:
            target = target or collection
            if not target:
                raise exceptions.DataError(
                    "Neither the operation nor the corpus names a collection to index into. Give the "
                    "operation a 'collection' param, or a 'target-collection' to each document set."
                )
            batch = batches.setdefault(target, [])
            batch.append(doc)
            if len(batch) >= batch_size:
                await flush(target, batch)
                batches[target] = []

        for target, batch in batches.items():
            if batch:
                await flush(target, batch, final=True)

        if do_commit:
            for target in (batches or {collection: None}):
                await _run_in_executor(sc.commit, target)

        elapsed = time.perf_counter() - start
        weight = total_docs - errors

        return {
            "weight": weight,
            "unit": "docs",
            "bulk-size": total_docs,
            "success": errors == 0,
            "error-count": errors,
            "took": elapsed,
        }

    def __str__(self):
        return "solr-bulk-index"


# ---------------------------------------------------------------------------
# Runner: binary-bulk-index (javabin / CBOR)
# ---------------------------------------------------------------------------

# Solr's UpdateRequestHandler registers a loader per content type, so the same /update endpoint takes
# a binary body. These are the two binary formats it accepts.
_BINARY_UPDATE_FORMATS = ("javabin", "cbor")


class SolrBinaryBulkIndex(SolrRunner):
    """
    Index documents through Solr's binary update formats rather than JSON.

    A workload may compare a binary indexing transport against JSON — http_logs does, via its gRPC
    operations. Solr has no gRPC endpoint, but ``UpdateRequestHandler`` registers
    ``application/javabin`` and ``application/cbor`` next to ``application/json``, so the comparable
    measurement is the same documents over the same endpoint in a binary encoding.

    Params:
      - ``collection``, ``bulk-size`` (default 500), ``commit`` (default False)
      - ``document-format`` — ``javabin`` (default) or ``cbor``
      - ``body`` or ``corpus`` — NDJSON line pairs, as for ``bulk-index``

    ⚠️ ``commit`` goes in the query string, not the body: measured against a live node, a javabin
    request carrying ``commit`` in its params returned 200 and logged the adds while leaving the
    documents invisible.
    """

    async def __call__(self, client, params):
        body = params.get("body", params.get("corpus", []))
        if isinstance(body, bytes):
            corpus_lines = [line.decode("utf-8") for line in body.split(b"\n") if line]
        elif isinstance(body, str):
            corpus_lines = [line for line in body.split("\n") if line]
        else:
            corpus_lines = body

        fmt = params.get("document-format", "javabin").lower()
        if fmt not in _BINARY_UPDATE_FORMATS:
            raise exceptions.DataError(
                "Unknown document-format '%s' for a binary update; Solr accepts %s."
                % (fmt, " and ".join(_BINARY_UPDATE_FORMATS))
            )
        if fmt == "cbor":
            try:
                import cbor2
            except ImportError as exc:
                raise exceptions.SystemSetupError(
                    "The cbor2 package is needed to index in CBOR format; install it or use "
                    "document-format=javabin."
                ) from exc

        batch_size = params.get("bulk-size", 500)
        do_commit = params.get("commit", False)
        collection = params.get("collection") or params.get("index")
        chain = params.get("update-chain") or params.get("pipeline")
        sc = client

        total_docs = 0
        errors = 0
        start = time.perf_counter()

        def encode(docs):
            if fmt == "cbor":
                import cbor2
                return cbor2.dumps(docs), "application/cbor"
            return encode_update_request(docs), "application/javabin"

        async def flush(target, docs, final=False):
            nonlocal total_docs, errors
            payload, content_type = encode(docs)
            if not payload:
                # An empty body is not a successful update. A dropped body once produced 200 with
                # nothing indexed, and the runner reported success on it.
                raise exceptions.BenchmarkError(
                    "Encoding %d documents as %s produced an empty body" % (len(docs), fmt))
            path = "/solr/%s/update" % urllib.parse.quote(target)
            if chain:
                path += "?" + urllib.parse.urlencode({"update.chain": chain})
            try:
                resp = await _run_in_executor(
                    sc.raw_request, "POST", path, payload, {"Content-type": content_type})
                if resp.status_code >= 400:
                    raise exceptions.BenchmarkError(
                        "Binary update returned %d: %s" % (resp.status_code, resp.text[:200]))
                total_docs += len(docs)
            except Exception as exc:
                logging.getLogger(__name__).error(
                    "Binary (%s) index error on %sbatch for '%s': %s",
                    fmt, "final " if final else "", target, exc)
                errors += len(docs)

        # Batched per target, as bulk-index does: a corpus may name a collection per document set.
        batches = {}
        for doc, target in _translate_ndjson_stream(corpus_lines):
            target = target or collection
            if not target:
                raise exceptions.DataError(
                    "Neither the operation nor the corpus names a collection to index into. Give the "
                    "operation a 'collection' param, or a 'target-collection' to each document set."
                )
            batch = batches.setdefault(target, [])
            batch.append(doc)
            if len(batch) >= batch_size:
                await flush(target, batch)
                batches[target] = []

        for target, batch in batches.items():
            if batch:
                await flush(target, batch, final=True)

        if do_commit:
            for target in (batches or {collection: None}):
                await _run_in_executor(sc.commit, target)

        elapsed = time.perf_counter() - start
        return {
            "weight": total_docs if total_docs else 1,
            "unit": "docs",
            "bulk-size": total_docs,
            "success": errors == 0,
            "error-count": errors,
            "took": elapsed,
        }

    def __str__(self):
        return "solr-binary-bulk-index"


# ---------------------------------------------------------------------------
# Runner: search
# ---------------------------------------------------------------------------

# Solr's terms query parser splits its value on the separator with a plain str.split() and has no
# escaping, so the separator must be a character no unique key can contain. \x01 is measured to work
# where \x1f, tab and newline return no results.
_PUBLISHED_ID_SEPARATOR = "\x01"
_PUBLISHED_IDS_PATTERN = re.compile(r"\$\{published-ids:([^}]+)\}")


class SolrSearch(SolrRunner):
    """
    Execute a Solr search query.

    - Classic Solr params: ``q``, ``fl``, ``rows``, ``fq``, ``sort``, ``request-params``
    - Solr JSON Query body: when ``body`` is present, POSTs it to ``/solr/{collection}/query``
      using the `Solr JSON Request API <https://solr.apache.org/guide/solr/latest/query-guide/json-request-api.html>`_.
    - ``publish-ids``: inside a composite operation, store the returned documents' unique keys in
      the composite context under this name, so that a later step can query the same documents.
      Requires a ``rows``/``limit`` bound and the unique key in the returned fields.
    - ``${published-ids:<name>}`` in any string of the operation expands to a terms query over the
      ids a preceding step published. The ``$`` form is deliberate: workload files are Jinja
      templates, so ``{{...}}`` would be consumed before the runner ever sees it.
    """

    async def __call__(self, client, params):
        collection = _get_collection(params)
        sc = client
        publish_ids = params.get("publish-ids")
        id_field = params.get("id-field", "id")

        params = self._substitute_published_ids(params, id_field)

        start = time.perf_counter()

        docs = None
        body = params.get("body")
        if body is not None:
            resp = await _run_in_executor(
                sc.raw_request, "POST", f"/solr/{collection}/query", body,
                {"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            response = resp.json().get("response", {})
            num_hits = response.get("numFound", 0)
            docs = response.get("docs")
        else:
            q = params.get("q", "*:*")
            kwargs = {}
            for key in ("fl", "rows", "fq", "sort"):
                if key in params:
                    kwargs[key] = params[key]
            kwargs.update(params.get("request-params", {}))

            results = await _run_in_executor(sc.search, collection, q, **kwargs)
            num_hits = results.hits
            docs = results.docs

        elapsed = time.perf_counter() - start

        if publish_ids:
            CompositeContext.put(publish_ids, [doc[id_field] for doc in (docs or []) if id_field in doc])

        return {
            "weight": 1,
            "unit": "ops",
            "hits": num_hits,
            "hits-total": num_hits,
            "took": elapsed,
        }

    @staticmethod
    def _substitute_published_ids(params, id_field):
        """
        Replace occurrences of ``${published-ids:<name>}`` with the ids a preceding step of the
        same composite operation published under ``<name>``, joined for a Solr terms query.

        The ids are read from the composite context, so this is only meaningful inside a composite
        operation. A separator that no unique key can contain is used, since Solr's terms parser
        splits on its separator with no escaping.
        """
        def render(value):
            if isinstance(value, str):
                def replace(match):
                    ids = CompositeContext.get(match.group(1))
                    return "{!terms f=%s separator=%s}%s" % (
                        id_field, _PUBLISHED_ID_SEPARATOR,
                        _PUBLISHED_ID_SEPARATOR.join(str(i) for i in ids))
                return _PUBLISHED_IDS_PATTERN.sub(replace, value)
            if isinstance(value, dict):
                return {k: render(v) for k, v in value.items()}
            if isinstance(value, list):
                return [render(v) for v in value]
            return value

        return render(params)

    def __str__(self):
        return "solr-search"


# ---------------------------------------------------------------------------
# Runner: paginated search (cursorMark deep pagination)
# ---------------------------------------------------------------------------

class SolrPaginatedSearch(SolrRunner):
    """
    Execute a cursor-paginated Solr search using cursorMark.

    Fetches pages from a result set using Solr's deep pagination API, stopping either when
    the result set is exhausted or when ``pages`` pages have been fetched.
    Params:
      - ``collection``, ``q`` (default ``*:*``), ``sort`` (must include a uniqueKey field,
        defaults to ``id asc``)
      - ``pages`` — maximum number of pages to fetch; unbounded when absent
      - ``results-per-page`` or ``rows`` — page size, default 100
      - ``fl``, ``fq``
      - ``request-params`` — additional Solr query params passed through
    Returns weight = total docs fetched across all pages.
    """

    async def __call__(self, client, params):
        collection = _get_collection(params)
        sc = client
        q = params.get("q", "*:*")
        max_pages = params.get("pages")
        rows = params.get("results-per-page", params.get("rows", 100))
        sort = params.get("sort", "id asc")
        kwargs = {"rows": rows, "sort": sort}
        for key in ("fl", "fq"):
            if key in params:
                kwargs[key] = params[key]
        kwargs.update(params.get("request-params", {}))

        cursor_mark = "*"
        total_docs = 0
        pages = 0
        start = time.perf_counter()

        while True:
            kwargs["cursorMark"] = cursor_mark
            results = await _run_in_executor(sc.search, collection, q, **kwargs)
            next_cursor = getattr(results, "nextCursorMark", None)
            total_docs += len(results.docs)
            pages += 1
            if max_pages is not None and pages >= max_pages:
                break
            if next_cursor is None or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor

        elapsed = time.perf_counter() - start
        return {
            "weight": total_docs,
            "unit": "docs",
            "hits": total_docs,
            "pages": pages,
            "took": elapsed,
        }

    def __str__(self):
        return "solr-paginated-search"


# ---------------------------------------------------------------------------
# Runner: commit
# ---------------------------------------------------------------------------

class SolrCommit(SolrRunner):
    """
    Commit pending changes in Solr.

    Params:
      - ``collection``, ``soft-commit`` (bool, default False)
    """

    async def __call__(self, client, params):
        collection = _get_collection(params)
        sc = client
        soft = params.get("soft-commit", False)

        start = time.perf_counter()
        if soft:
            await _run_in_executor(sc.commit, collection, softCommit=True)
        else:
            await _run_in_executor(sc.commit, collection)
        elapsed = time.perf_counter() - start

        return {"weight": 1, "unit": "ops", "took": elapsed}

    def __str__(self):
        return "solr-commit"


# ---------------------------------------------------------------------------
# Runner: optimize
# ---------------------------------------------------------------------------

class SolrOptimize(SolrRunner):
    """
    Force-merge Solr segments (optimize).

    Params:
      - ``collection``, ``max-segments`` (int, default 1)
    """

    async def __call__(self, client, params):
        collection = _get_collection(params)
        sc = client
        max_segments = params.get("max-segments", 1)

        start = time.perf_counter()
        await _run_in_executor(sc.optimize, collection, maxSegments=max_segments)
        elapsed = time.perf_counter() - start

        return {"weight": 1, "unit": "ops", "took": elapsed}

    def __str__(self):
        return "solr-optimize"


# ---------------------------------------------------------------------------
# Runner: wait-for-merges
# ---------------------------------------------------------------------------

class SolrWaitForMerges(SolrRunner):
    """
    Poll Solr node metrics until no active merge operations remain across any core.

    Params:
      - ``retry-wait-period`` (default 2.0s), ``max-wait-seconds`` (default 3600s)
    """

    async def __call__(self, client, params):
        sc = client
        retry_wait = float(params.get("retry-wait-period", 2.0))
        max_wait = float(params.get("max-wait-seconds", 3600))
        start = time.perf_counter()

        while True:
            raw = await _run_in_executor(sc.get_node_metrics)
            total_running = 0

            if isinstance(raw, str):
                m = _parse_prometheus_text(raw)
                for key, val in m.items():
                    if "merge" in key and "running" in key:
                        total_running += int(val)
            elif isinstance(raw, dict):
                for core_metrics in raw.get("metrics", {}).values():
                    for key in ("INDEX.merge.major.running",
                                "INDEX.merge.minor.running"):
                        val = core_metrics.get(key, 0)
                        if isinstance(val, dict):
                            val = val.get("value", 0)
                        total_running += int(val)

            elapsed = time.perf_counter() - start
            if total_running == 0 or elapsed >= max_wait:
                break
            await asyncio.sleep(retry_wait)

        return {
            "weight": 1,
            "unit": "ops",
            "took": time.perf_counter() - start,
            "success": total_running == 0,
        }

    def __str__(self):
        return "solr-wait-for-merges"


# ---------------------------------------------------------------------------
# Runner: create-collection
# ---------------------------------------------------------------------------

class SolrCreateCollection(SolrRunner):
    """
    Collection creation — optionally with configset upload.

    Params:
      - ``collection``, ``configset`` (default: collection name),
        ``configset-path`` (local dir; omit to use existing server configset),
        ``num-shards`` (default 1), ``replication-factor`` (default 1),
        ``tlog-replicas`` (default 0), ``pull-replicas`` (default 0),
        ``delete-configset-on-error`` (default True)
    """

    async def __call__(self, client, params):
        sc = client
        collection = params["collection"]
        configset = params.get("configset", collection)
        configset_path = params.get("configset-path")
        num_shards = params.get("num-shards", 1)
        replication_factor = params.get("replication-factor", 1)
        tlog_replicas = params.get("tlog-replicas", 0)
        pull_replicas = params.get("pull-replicas", 0)

        start = time.perf_counter()

        if configset_path:
            await _run_in_executor(sc.upload_configset, configset, configset_path)
            logging.getLogger(__name__).info("Uploaded configset '%s' from '%s'", configset, configset_path)

        try:
            await _run_in_executor(
                sc.create_collection,
                collection,
                configset,
                num_shards,
                replication_factor,
                tlog_replicas,
                pull_replicas,
            )
        except CollectionAlreadyExistsError:
            logging.getLogger(__name__).warning("Collection '%s' already exists, skipping creation.", collection)
        except Exception:
            if configset_path and params.get("delete-configset-on-error", True):
                try:
                    await _run_in_executor(sc.delete_configset, configset)
                except Exception as cleanup_exc:
                    logging.getLogger(__name__).warning("Failed to clean up configset '%s': %s", configset, cleanup_exc)
            raise

        elapsed = time.perf_counter() - start
        return {"weight": 1, "unit": "ops", "took": elapsed}

    def __str__(self):
        return "solr-create-collection"


# ---------------------------------------------------------------------------
# Runner: delete-collection
# ---------------------------------------------------------------------------

class SolrDeleteCollection(SolrRunner):
    """
    Delete a Solr collection, optionally deleting its configset too.

    Params:
      - ``collection``, ``configset`` (default: collection name),
        ``delete-configset`` (bool, default True),
        ``ignore-missing`` (bool, default True)
    """

    async def __call__(self, client, params):
        sc = client
        collection = params["collection"]
        configset = params.get("configset", collection)
        ignore_missing = params.get("ignore-missing", True)
        delete_configset = params.get("delete-configset", True)

        start = time.perf_counter()
        try:
            await _run_in_executor(sc.delete_collection, collection)
        except CollectionNotFoundError:
            if not ignore_missing:
                raise
            logging.getLogger(__name__).info("Collection '%s' not found, skipping delete.", collection)

        if delete_configset:
            try:
                await _run_in_executor(sc.delete_configset, configset)
            except Exception as exc:
                logging.getLogger(__name__).warning("Could not delete configset '%s': %s", configset, exc)

        elapsed = time.perf_counter() - start
        return {"weight": 1, "unit": "ops", "took": elapsed}

    def __str__(self):
        return "solr-delete-collection"


# ---------------------------------------------------------------------------
# Runner: create-alias / delete-alias
# ---------------------------------------------------------------------------

class SolrCreateAlias(SolrRunner):
    """
    Point one name at several collections, so a workload written against an index pattern can run.

    A workload that searches ``logs-*`` has no direct Solr equivalent — there is no wildcard
    collection name, and asking for one is a 404. A standard alias over the matching collections
    searches all their shards as one whole, which is the behaviour those operations are after.

    Params:
      - ``alias``, ``collections`` (list or comma-separated string)
    """

    async def __call__(self, client, params):
        sc = client
        alias = params.get("alias") or params["collection"]
        collections = params.get("collections")
        if not collections:
            raise exceptions.DataError(
                "create-alias needs a 'collections' parameter naming the collections the alias spans."
            )

        start = time.perf_counter()
        await _run_in_executor(sc.create_alias, alias, collections)
        elapsed = time.perf_counter() - start
        return {"weight": 1, "unit": "ops", "took": elapsed}

    def __str__(self):
        return "solr-create-alias"


class SolrDeleteAlias(SolrRunner):
    """
    Remove an alias.

    Params:
      - ``alias``, ``ignore-missing`` (bool, default True)
    """

    async def __call__(self, client, params):
        sc = client
        alias = params.get("alias") or params["collection"]
        ignore_missing = params.get("ignore-missing", True)

        start = time.perf_counter()
        await _run_in_executor(sc.delete_alias, alias, ignore_missing=ignore_missing)
        elapsed = time.perf_counter() - start
        return {"weight": 1, "unit": "ops", "took": elapsed}

    def __str__(self):
        return "solr-delete-alias"


# ---------------------------------------------------------------------------
# Runner: raw-request
# ---------------------------------------------------------------------------

class RawRequest(Runner):
    """
    Send an arbitrary HTTP request to any Solr endpoint.

    Params:
      - ``method`` (default "GET"), ``path``, ``body``, ``headers``
      - ``form`` — send the body form-encoded rather than as JSON

    ⚠️ Some Solr handlers read their input from request parameters, not from a JSON body: ``/sql``
    answers "stmt parameter cannot be null" to a JSON body carrying ``stmt`` and works when the same
    body is form-encoded. Setting ``form`` says which, since the runner cannot tell from the path.
    """

    async def __call__(self, client, params):
        sc = client
        method = params.get("method", "GET")
        path = params["path"]
        body = params.get("body")
        headers = dict(params.get("headers", {}))

        if params.get("form") and isinstance(body, dict):
            body = urllib.parse.urlencode(body)
            headers.setdefault("Content-type", "application/x-www-form-urlencoded")

        start = time.perf_counter()
        resp = await _run_in_executor(sc.raw_request, method, path, body, headers)
        elapsed = time.perf_counter() - start

        # A handler can answer 200 and report the failure inside the payload. Solr's SQL handler puts
        # an EXCEPTION entry in its result-set, so a status check alone reports success on a query
        # that did not run.
        error = None
        if resp.status_code < 400:
            try:
                payload = resp.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                docs = (payload.get("result-set") or {}).get("docs") or []
                for doc in docs:
                    if isinstance(doc, dict) and "EXCEPTION" in doc:
                        error = str(doc["EXCEPTION"])[:200]
                        break

        if error:
            logging.getLogger(__name__).error("Request to %s reported: %s", path, error)

        return {
            "weight": 1,
            "unit": "ops",
            "http-status": resp.status_code,
            "success": resp.status_code < 400 and error is None,
            "error-count": 0 if error is None and resp.status_code < 400 else 1,
            "took": elapsed,
        }

    def __str__(self):
        return "raw-request"
