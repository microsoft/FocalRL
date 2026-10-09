import asyncio
import json

import pytest

from adaptive_branching.src.swe import harbor_cleanup as cleanup

A, B, C, D = (letter * 64 for letter in "abcd")
HOST = "unix:///tmp/test-docker.sock"


class Docker:
    def __init__(self, entries=None):
        self.entries = dict(entries or {})
        self.calls = []
        self.failures = {}
        self.resources = {"network": {}, "volume": {}}

    async def __call__(self, host, *args, timeout=15):
        assert host == HOST and timeout > 0
        self.calls.append(args)
        failure = self.failures.get(args[:2])
        if failure:
            raise failure
        if args[0] == "ps":
            if args[-1] == '{{.Label "com.docker.compose.project"}}':
                return "\n".join(owner for owner, _ in self.entries.values())
            return "\n".join(f"{cid}\t{owner}\t{state}" for cid, (owner, state) in self.entries.items())
        if args[0] == "kill":
            owner, state = self.entries[args[1]]
            self.entries[args[1]] = owner, "exited"
            return args[1]
        if args[0] == "rm":
            assert self.entries[args[1]][1] in cleanup.STOPPED
            del self.entries[args[1]]
            return args[1]
        kind, operation = args[:2]
        if operation == "ls":
            if args[-1] == '{{.Label "com.docker.compose.project"}}':
                return "\n".join(self.resources[kind].values())
            owner = args[-1].split("=", 2)[-1]
            return "\n".join(name for name, label in self.resources[kind].items() if label == owner)
        assert operation == "rm"
        del self.resources[kind][args[2]]
        return args[2]


def queue(tmp_path, fake):
    result = cleanup.CleanupQueue(tmp_path, HOST, command=fake)
    if isinstance(fake, Docker):
        for project in {owner for owner, _ in fake.entries.values()} | {
            owner for resources in fake.resources.values() for owner in resources.values()
        }:
            result.register_project(project)
    return result


@pytest.mark.parametrize(
    "entries", [{}, {A: ("trial", "running")}, {A: ("trial", "running"), B: ("trial", "running")}]
)
def test_stop_then_delete_and_persist_success(tmp_path, entries):
    fake = Docker(entries)
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    stopped = asyncio.run(q.reconcile("trial"))
    assert stopped["phase"] == ("disk_pending" if entries else "done")
    assert not any(call[0] == "rm" for call in fake.calls)
    assert all(state == "exited" for _, state in fake.entries.values())
    completed = asyncio.run(queue(tmp_path, fake).reconcile("trial", disk=True))
    assert completed["phase"] == "done" and not fake.entries
    assert queue(tmp_path, fake).read(q.path("trial"))["phase"] == "done"


def test_one_stop_failure_does_not_skip_sidecar_or_delete_anything(tmp_path):
    fake = Docker({A: ("trial", "running"), B: ("trial", "running")})
    fake.failures[("kill", A)] = cleanup.DockerFailure("main stop failed")
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    result = asyncio.run(q.reconcile("trial", disk=True))
    assert ("kill", B) in fake.calls and fake.entries[B][1] == "exited"
    assert result["phase"] == "stop_pending" and result["errors"]
    assert not any(call[0] == "rm" for call in fake.calls)
    fake.failures.clear()
    assert asyncio.run(queue(tmp_path, fake).reconcile("trial", disk=True))["phase"] == "done"


@pytest.mark.parametrize("kind", ["network", "volume"])
def test_resource_discovery_failure_still_stops_every_container(tmp_path, kind):
    fake = Docker({A: ("trial", "running"), B: ("trial", "running")})
    fake.failures[(kind, "ls")] = cleanup.DockerFailure("resource listing unavailable")
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    result = asyncio.run(q.reconcile("trial", disk=True))
    assert all(state == "exited" for _, state in fake.entries.values())
    assert result["phase"] == "disk_pending" and result["errors"]
    assert result["containers"] == [A, B]
    assert not any(call[0] == "rm" for call in fake.calls)
    fake.failures.clear()
    assert asyncio.run(queue(tmp_path, fake).reconcile("trial", disk=True))["phase"] == "done"


def test_busy_main_does_not_leave_live_sidecar_and_retries_after_restart(tmp_path, monkeypatch):
    fake = Docker({A: ("trial", "running"), B: ("trial", "running")})
    fake.failures[("rm", A)] = cleanup.DockerFailure("zfs dataset is busy")
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    result = asyncio.run(q.reconcile("trial", disk=True))
    assert result["phase"] == "disk_pending" and "busy" in result["errors"][0]
    assert fake.entries == {A: ("trial", "exited")}
    assert max(fake.calls.index(("kill", cid)) for cid in [A, B]) < fake.calls.index(("rm", A))
    fake.failures.clear()
    monkeypatch.setattr(cleanup.time, "time", lambda: result["next_retry"] + 1)
    restarted = queue(tmp_path, fake)
    asyncio.run(restarted.drain_once())
    assert restarted.read(restarted.path("trial"))["phase"] == "done"


@pytest.mark.parametrize("failure", [cleanup.DockerFailure("timeout", uncertain=True), asyncio.CancelledError()])
def test_uncertain_delete_is_not_reissued_on_restart(tmp_path, failure):
    fake = Docker({A: ("trial", "running"), B: ("trial", "running")})
    fake.failures[("rm", A)] = failure
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(q.reconcile("trial", disk=True))
    else:
        asyncio.run(q.reconcile("trial", disk=True))
    assert q.read(q.path("trial"))["phase"] == "needs_attention"
    assert ("rm", B) not in fake.calls
    calls = list(fake.calls)
    asyncio.run(queue(tmp_path, fake).drain_once())
    assert fake.calls == calls


def test_hard_worker_exit_during_delete_is_durable(tmp_path):
    fake = Docker({A: ("trial", "running")})
    fake.failures[("rm", A)] = SystemExit(9)
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    with pytest.raises(SystemExit):
        asyncio.run(q.reconcile("trial", disk=True))
    assert q.read(q.path("trial"))["phase"] == "delete_inflight"
    calls = list(fake.calls)
    result = asyncio.run(queue(tmp_path, fake).reconcile("trial", disk=True))
    assert result["phase"] == "needs_attention" and fake.calls == calls


def test_exact_ownership_and_nested_verifier_resources(tmp_path):
    fake = Docker({A: ("trial", "running"), B: ("trial__verifier__x", "running"), C: ("trial-other", "running")})
    fake.resources["network"] = {"owned": "trial", "nested": "trial__verifier__x", "other": "trial-other"}
    fake.resources["volume"] = {"owned-vol": "trial__verifier__x", "other-vol": "trial-other"}
    q = queue(tmp_path, fake)
    q.enqueue("trial", include_children=True)
    assert asyncio.run(q.reconcile("trial", disk=True, include_children=True))["phase"] == "done"
    assert fake.entries == {C: ("trial-other", "running")}
    assert fake.resources == {"network": {"other": "trial-other"}, "volume": {"other-vol": "trial-other"}}


def test_keep_containers_stops_without_deleting(tmp_path):
    fake = Docker({A: ("trial", "running")})
    q = queue(tmp_path, fake)
    q.enqueue("trial", remove=False)
    assert asyncio.run(q.reconcile("trial", disk=True))["phase"] == "done"
    assert fake.entries == {A: ("trial", "exited")}
    with pytest.raises(ValueError, match="conflicting"):
        q.enqueue("trial", remove=True)


def test_overlapping_workers_do_not_duplicate_mutations(tmp_path):
    fake = Docker({A: ("trial", "running")})
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    other = queue(tmp_path, fake)
    with q.locked("trial") as acquired:
        assert acquired
        assert asyncio.run(other.reconcile("trial", disk=True)) is None
        assert other.enqueue("trial")["phase"] == "stop_pending"
        assert not fake.calls


@pytest.mark.parametrize("project", ["", "../bad", "UPPER", "a" * 256, None, 1])
def test_reject_bad_project(tmp_path, project):
    with pytest.raises(ValueError):
        queue(tmp_path, Docker()).enqueue(project)
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize(
    "directory,host", [("relative", HOST), ("/tmp/unused", "tcp://remote:1234"), ("/tmp/unused", "")]
)
def test_reject_bad_configuration(directory, host):
    with pytest.raises(ValueError):
        cleanup.CleanupQueue(directory, host)


@pytest.mark.parametrize(
    "mutation",
    [{"phase": "corrupt"}, {"host": "unix:///other"}, {"containers": ["bad"]}, {"resource_projects": ["unowned"]}],
)
def test_corrupt_record_fails_without_mutation(tmp_path, mutation):
    fake = Docker()
    q = queue(tmp_path, fake)
    record = q.enqueue("trial")
    q.path("trial").write_text(json.dumps({**record, **mutation}))
    with pytest.raises(ValueError):
        asyncio.run(q.drain_once())
    assert not any(call[0] == "rm" or call[1] == "rm" for call in fake.calls)


def test_retry_backoff_and_incomplete_temp_file(tmp_path):
    fake = Docker()
    q = queue(tmp_path, fake)
    row = q.enqueue("trial")
    row["next_retry"] = cleanup.time.time() + 100
    q.write(row)
    (tmp_path / "unfinished.tmp").write_text('{"broken":')
    asyncio.run(q.drain_once())
    assert not any(call[0] == "rm" or call[1] == "rm" for call in fake.calls)


@pytest.mark.parametrize("text", ["bad", f"{A}\ttrial\tunknown", f"{A}\t../bad\trunning"])
def test_malformed_docker_listing_fails_closed(tmp_path, text):
    async def command(*args, **kwargs):
        return text

    q = queue(tmp_path, command)
    q.enqueue("trial")
    with pytest.raises(ValueError):
        asyncio.run(q.reconcile("trial"))


def test_configured_finish_reports_stop_failure_and_preserves_queue(tmp_path, monkeypatch):
    fake = Docker({A: ("trial", "running")})
    fake.failures[("kill", A)] = cleanup.DockerFailure("failed")
    q = queue(tmp_path, fake)
    monkeypatch.setattr(cleanup, "configured_queue", lambda: q)
    with pytest.raises(RuntimeError, match="stop not verified"):
        asyncio.run(cleanup.finish_project("trial"))
    assert q.read(q.path("trial"))["phase"] == "stop_pending"


def test_cli_timeout_is_explicit_and_reaps_client(monkeypatch):
    class Process:
        returncode = None
        killed = False
        waited = False

        async def communicate(self):
            await asyncio.sleep(10)

        def kill(self):
            self.killed = True

        async def wait(self):
            self.waited = True

    proc = Process()

    async def create(*args, **kwargs):
        return proc

    monkeypatch.setattr(cleanup.asyncio, "create_subprocess_exec", create)
    with pytest.raises(cleanup.DockerFailure) as exc:
        asyncio.run(cleanup.docker(HOST, "rm", A, timeout=0.001))
    assert exc.value.uncertain and proc.killed and proc.waited


def test_delete_false_preserves_volumes_and_removes_networks(tmp_path):
    fake = Docker({A: ("trial", "running")})
    fake.resources = {"network": {"net": "trial"}, "volume": {"vol": "trial"}}
    q = queue(tmp_path, fake)
    q.enqueue("trial", remove_volumes=False)
    assert asyncio.run(q.reconcile("trial", disk=True))["phase"] == "done"
    assert fake.resources == {"network": {}, "volume": {"vol": "trial"}}


def test_network_busy_is_persisted_and_retried(tmp_path):
    fake = Docker()
    fake.resources["network"] = {"net": "trial"}
    fake.failures[("network", "rm")] = cleanup.DockerFailure("network busy")
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    assert asyncio.run(q.reconcile("trial", disk=True))["phase"] == "disk_pending"
    fake.failures.clear()
    assert asyncio.run(queue(tmp_path, fake).reconcile("trial", disk=True))["phase"] == "done"


def test_live_removal_is_not_reported_as_stopped(tmp_path):
    fake = Docker({A: ("trial", "removing")})
    fake.failures[("kill", A)] = cleanup.DockerFailure("removal already in progress")
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    assert asyncio.run(q.reconcile("trial", disk=True))["phase"] == "stop_pending"
    assert not any(call[0] == "rm" for call in fake.calls)


def test_maximum_project_name_and_settings_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", HOST)
    monkeypatch.delenv("MILES_SWE_DOCKER_SOCKET", raising=False)
    q = queue(tmp_path / "queue", Docker())
    q.enqueue("a" * 255)
    assert q.read(q.path("a" * 255))["phase"] == "stop_pending"
    monkeypatch.setattr(cleanup, "__file__", str(tmp_path / "durable_cleanup.py"))
    path = tmp_path / "cleanup_settings.json"
    path.write_text(json.dumps({"directory": str(tmp_path / "queue"), "host": HOST}))
    assert cleanup.configured_queue().host == HOST
    path.write_text("{}")
    with pytest.raises(ValueError):
        cleanup.configured_queue()


def test_runtime_daemon_overrides_shared_prepared_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "__file__", str(tmp_path / "durable_cleanup.py"))
    (tmp_path / "cleanup_settings.json").write_text(
        json.dumps({"directory": str(tmp_path / "queue"), "host": "unix:///run/ruc01/docker.sock"})
    )
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/ruc02/docker.sock")
    monkeypatch.setenv("MILES_SWE_DOCKER_SOCKET", "/run/ruc02/docker.sock")
    q = cleanup.configured_queue()
    assert q.host == "unix:///run/ruc02/docker.sock"
    assert cleanup.configured_queue().directory == q.directory
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/ruc01/docker.sock")
    monkeypatch.setenv("MILES_SWE_DOCKER_SOCKET", "/run/ruc01/docker.sock")
    assert cleanup.configured_queue().directory != q.directory


@pytest.mark.parametrize("host,sock", [(None, "/run/a"), ("", "/run/a"), ("unix:///", "/"),
                                        ("tcp://localhost:2375", "/run/a"), ("unix:///run/a", "/run/b")])
def test_cleanup_rejects_invalid_or_mismatched_runtime(tmp_path, monkeypatch, host, sock):
    monkeypatch.setattr(cleanup, "__file__", str(tmp_path / "durable_cleanup.py"))
    (tmp_path / "cleanup_settings.json").write_text(
        json.dumps({"directory": str(tmp_path / "queue"), "host": HOST})
    )
    if host is None:
        monkeypatch.delenv("DOCKER_HOST", raising=False)
    else:
        monkeypatch.setenv("DOCKER_HOST", host)
    monkeypatch.setenv("MILES_SWE_DOCKER_SOCKET", sock)
    with pytest.raises(ValueError):
        cleanup.configured_queue()


def test_background_loop_processes_restart_queue_and_reports_failure(tmp_path, monkeypatch, caplog):
    q = queue(tmp_path, Docker({A: ("trial", "running")}))
    q.enqueue("trial")
    monkeypatch.setattr(cleanup, "configured_queue", lambda: q)

    async def cancel_sleep(seconds):
        assert seconds == 5
        raise asyncio.CancelledError()

    monkeypatch.setattr(cleanup.asyncio, "sleep", cancel_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cleanup.cleanup_loop())
    assert q.read(q.path("trial"))["phase"] == "done"

    async def failed():
        raise ValueError("invalid journal")

    async def exercise():
        task = asyncio.create_task(failed())
        with pytest.raises(ValueError):
            await task
        cleanup.report_worker_exit(task)

    asyncio.run(exercise())
    assert "persistent cleanup worker exited" in caplog.text


def test_main_teardown_does_not_stop_active_verifier_but_terminal_parent_does(tmp_path, monkeypatch):
    fake = Docker({A: ("trial", "running"), B: ("trial__verifier__x", "running")})
    q = queue(tmp_path, fake)
    q.enqueue("trial")
    assert asyncio.run(q.reconcile("trial", disk=True))["phase"] == "done"
    assert fake.entries == {B: ("trial__verifier__x", "running")}
    monkeypatch.setattr(cleanup, "configured_queue", lambda: q)
    assert asyncio.run(cleanup.finish_trial("trial"))["phase"] == "disk_pending"
    assert fake.entries == {B: ("trial__verifier__x", "exited")}
    assert q.path("trial") != q.path("trial", include_children=True)
    assert asyncio.run(q.reconcile("trial", disk=True, include_children=True))["phase"] == "done"


def test_journal_fsync_does_not_block_other_http_tasks(tmp_path, monkeypatch):
    import time

    q = queue(tmp_path, Docker({A: ("trial", "running")}))
    original = q.write

    def slow_write(row):
        time.sleep(0.05)
        original(row)

    monkeypatch.setattr(q, "write", slow_write)

    async def exercise():
        task = asyncio.create_task(q.enqueue_async("trial"))
        await asyncio.sleep(0.01)
        assert not task.done()
        await task
        assert q.read(q.path("trial"))["phase"] == "stop_pending"

    asyncio.run(exercise())


def test_cancel_during_enqueue_still_finishes_durable_write(tmp_path, monkeypatch):
    import time

    q = queue(tmp_path, Docker())
    original = q.write

    def slow_write(row):
        time.sleep(0.05)
        original(row)

    monkeypatch.setattr(q, "write", slow_write)

    async def exercise():
        task = asyncio.create_task(q.enqueue_async("trial", include_children=True))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert q.read(q.path("trial", include_children=True))["phase"] == "stop_pending"

    asyncio.run(exercise())


def test_parent_honors_real_harbor_main_session_retention(tmp_path, monkeypatch):
    fake = Docker({A: ("trial__env", "running")})
    q = queue(tmp_path, fake)
    q.enqueue("trial__env", remove=False, remove_volumes=False)
    assert asyncio.run(q.reconcile("trial__env"))["phase"] == "done"
    monkeypatch.setattr(cleanup, "configured_queue", lambda: q)
    result = asyncio.run(cleanup.finish_trial("trial"))
    assert result["phase"] == "done" and result["remove"] is False
    assert fake.entries == {A: ("trial__env", "exited")}
    assert not any(call[0] == "rm" for call in fake.calls)
