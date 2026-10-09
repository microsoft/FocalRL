"""Durable cleanup for terminal Harbor projects, separate from task rewards.

Copied into agent_server by prepare_harbor_cleanup. Only explicitly enqueued
terminal projects are eligible; this module does not discover abandoned trials.
"""

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import socket
import time
from contextlib import ExitStack, contextmanager
from itertools import islice
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT = re.compile(r"[a-z0-9][a-z0-9_-]{0,254}\Z")
CONTAINER = re.compile(r"[0-9a-f]{64}\Z")
STOPPED = {"created", "exited", "dead"}
PHASES = {"stop_pending", "disk_pending", "delete_inflight", "done", "needs_attention"}


class DockerFailure(RuntimeError):
    def __init__(self, message, *, uncertain=False):
        if not isinstance(message, str) or not message or type(uncertain) is not bool:
            raise ValueError("DockerFailure requires a message and boolean uncertainty")
        super().__init__(message)
        self.uncertain = uncertain


async def docker(host, *args, timeout=15):
    if not isinstance(host, str) or not host.startswith("unix:///") or timeout <= 0 or not args:
        raise ValueError("cleanup requires an explicit Unix Docker socket, command and positive timeout")
    proc = await asyncio.create_subprocess_exec(
        "docker", "--host", host, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise DockerFailure(
            f"Docker command timed out: {args!r}; daemon operation may still be running",
            uncertain=args[0] in {"kill", "rm"} or args[:2] in {("network", "rm"), ("volume", "rm")},
        ) from exc
    if proc.returncode:
        raise DockerFailure(
            f"Docker command {args!r} failed ({proc.returncode}): {err.decode(errors='replace')[-2000:]}"
        )
    return out.decode()


class CleanupQueue:
    def __init__(self, directory, host, *, command=docker, concurrency=4):
        self.directory = Path(directory)
        if not self.directory.is_absolute() or not isinstance(host, str) or not host.startswith("unix:///"):
            raise ValueError("cleanup queue and Docker socket must be explicit absolute paths")
        if not callable(command):
            raise TypeError("command must be callable")
        if type(concurrency) is not int or not 1 <= concurrency <= 16:
            raise ValueError("cleanup concurrency must be an integer between 1 and 16")
        self.concurrency = concurrency
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending = self.directory / "pending"
        self.pending.mkdir(exist_ok=True)
        self.ownership = self.directory / "ownership"
        self.ownership.mkdir(exist_ok=True)
        self.host, self.command = host, command
        self._journal_scan = iter(())
        self._indexed = False

    def path(self, project, *, include_children=False):
        if type(include_children) is not bool:
            raise TypeError("include_children must be boolean")
        if not isinstance(project, str) or not PROJECT.fullmatch(project):
            raise ValueError(f"invalid Compose project: {project!r}")
        key = hashlib.sha256(f"{self.host}\0{project}\0{include_children}".encode()).hexdigest()
        return self.directory / f"{key}.json"

    @contextmanager
    def locked(self, project, *, include_children=False):
        with self.path(project, include_children=include_children).with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextmanager
    def locked_resources(self, projects):
        if not isinstance(projects, list) or not projects or len(projects) != len(set(projects)):
            raise ValueError("resource locks require a nonempty list of distinct projects")
        paths = sorted(self.path(project).with_suffix(".resource.lock") for project in projects)
        with ExitStack() as stack:
            for path in paths:
                lock = stack.enter_context(path.open("a"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    yield False
                    return
            yield True  # Closing every descriptor releases all acquired locks.

    def read(self, path):
        row = json.loads(Path(path).read_text())
        if (
            row.get("schema") != 1
            or row.get("host") != self.host
            or row.get("phase") not in PHASES
            or type(row.get("include_children")) is not bool
            or type(row.get("remove")) is not bool
            or type(row.get("remove_volumes")) is not bool
            or type(row.get("attempts")) is not int
            or row["attempts"] < 0
            or not isinstance(row.get("next_retry"), (float, int))
            or not isinstance(row.get("updated_at"), (float, int))
            or not isinstance(row.get("containers"), list)
            or any(not isinstance(cid, str) or not CONTAINER.fullmatch(cid) for cid in row["containers"])
            or len(row["containers"]) != len(set(row["containers"]))
            or not isinstance(row.get("errors"), list)
            or any(not isinstance(error, str) for error in row["errors"])
            or any(
                not isinstance(owner, str)
                or not PROJECT.fullmatch(owner)
                or not (owner == row.get("project") or owner.startswith(str(row.get("project")) + "__"))
                for owner in row.get("resource_projects", [])
            )
            or self.path(row.get("project"), include_children=row["include_children"]) != Path(path)
        ):
            raise ValueError(f"invalid cleanup record: {path}")
        return row

    def unresolved_resource_records(self, projects, *, exclude):
        if not isinstance(projects, list) or not projects or not isinstance(exclude, Path):
            raise ValueError("resource uncertainty lookup requires projects and the current record path")
        if exclude.parent != self.directory:
            raise ValueError(f"record outside cleanup queue: {exclude}")
        paths = set()
        for project in projects:
            paths.add(self.path(project))
            parts = project.split("__")
            paths.update(self.path("__".join(parts[:end]), include_children=True) for end in range(1, len(parts) + 1))
        blocked = []
        for path in sorted(paths - {exclude}):
            if path.exists() and self.read(path)["phase"] in {"delete_inflight", "needs_attention"}:
                blocked.append(path.name)
        return blocked

    def write(self, row):
        if not isinstance(row, dict) or row.get("phase") not in PHASES:
            raise ValueError("invalid cleanup state to write")
        path = self.path(row["project"], include_children=row["include_children"])
        row["updated_at"] = time.time()
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("w") as stream:
            json.dump(row, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.index_record(path, row)

    def index_record(self, path, row, *, sync=True):
        if (
            type(sync) is not bool
            or self.path(row["project"], include_children=row["include_children"]) != path
            or row["phase"] not in PHASES
        ):
            raise ValueError(f"invalid cleanup index record: {path}")
        marker = self.pending / path.name
        if row["phase"] in {"done", "needs_attention"}:
            marker.unlink(missing_ok=True)
        else:
            marker.touch(exist_ok=True)
        if not sync:
            return
        fd = os.open(self.pending, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def index_journal(self):
        """Upgrade legacy journals once; completed receipts stay out of the hot queue."""
        ready = self.pending / ".indexed"
        if self._indexed:
            return
        with (self.pending / ".index.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for path in self.directory.glob("*.json"):
                initial = self.read(path)
                with self.locked(initial["project"], include_children=initial["include_children"]) as acquired:
                    if not acquired:
                        # Its writer updates the index before releasing the lock.
                        continue
                    self.index_record(path, self.read(path), sync=False)
            ready.touch()
            fd = os.open(self.pending, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            self._indexed = True

    async def persist(self, row):
        """Keep fsync off the HTTP loop; finish the durable write before cancelling."""
        if not isinstance(row, dict) or row.get("phase") not in PHASES:
            raise ValueError("invalid cleanup state to persist")
        task = asyncio.create_task(asyncio.to_thread(self.write, row))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def enqueue_async(self, project, **policy):
        self.path(project, include_children=policy.get("include_children", False))
        task = asyncio.create_task(asyncio.to_thread(self.enqueue, project, **policy))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def enqueue(self, project, *, remove=True, remove_volumes=True, include_children=False):
        if type(remove) is not bool or type(remove_volumes) is not bool:
            raise TypeError("remove must be a boolean")
        with self.locked(project, include_children=include_children) as acquired:
            path = self.path(project, include_children=include_children)
            if not acquired:
                row = self.read(path)
                if row["remove"] != remove or row["remove_volumes"] != remove_volumes:
                    raise ValueError(f"conflicting keep/delete policy: {project}")
                return row
            if path.exists():
                row = self.read(path)
                if row["remove"] != remove or row["remove_volumes"] != remove_volumes:
                    raise ValueError(f"conflicting keep/delete policy: {project}")
                return row
            row = dict(
                schema=1,
                host=self.host,
                project=project,
                include_children=include_children,
                remove=remove,
                remove_volumes=remove_volumes,
                phase="stop_pending",
                attempts=0,
                next_retry=0,
                containers=[],
                errors=[],
                created_at=time.time(),
            )
            self.write(row)
            return row

    async def containers(self, project, *, include_children=False):
        self.path(project, include_children=include_children)
        if include_children:
            found, owners = {}, set()
            for child in await asyncio.to_thread(self.registered_projects, project):
                containers, labels = await self.containers(child)
                if found.keys() & containers.keys():
                    raise ValueError(f"duplicate container ownership for {project}")
                found.update(containers)
                owners.update(labels)
            return found, owners
        text = await self.command(
            self.host,
            "ps",
            "-a",
            "--no-trunc",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            '{{.ID}}\t{{.Label "com.docker.compose.project"}}\t{{.State}}',
        )
        found, owners = {}, set()
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) != 3 or not CONTAINER.fullmatch(parts[0]) or not PROJECT.fullmatch(parts[1]):
                raise ValueError(f"unexpected Docker project listing: {line!r}")
            cid, owner, state = parts
            if state not in STOPPED | {"running", "paused", "restarting", "removing"} or cid in found:
                raise ValueError(f"unexpected Docker state/duplicate: {line!r}")
            if owner == project or (include_children and owner.startswith(project + "__")):
                found[cid] = state
                owners.add(owner)
        return found, owners

    def register_project(self, project):
        """Record ownership before Compose can allocate even a partial environment."""
        self.path(project)
        parts = project.split("__")
        for end in range(1, len(parts) + 1):
            parent = "__".join(parts[:end])
            folder = self.ownership / self.path(parent).stem
            folder.mkdir(exist_ok=True)
            target = folder / self.path(project).name
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("w") as stream:
                json.dump({"host": self.host, "project": project}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            fd = os.open(folder, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fd = os.open(self.ownership, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def registered_projects(self, project):
        self.path(project)
        projects = {project}
        for path in (self.ownership / self.path(project).stem).glob("*.json"):
            row = json.loads(path.read_text())
            child = row.get("project")
            if (
                row.get("host") != self.host
                or not isinstance(child, str)
                or not PROJECT.fullmatch(child)
                or not (child == project or child.startswith(project + "__"))
                or path.name != self.path(child).name
            ):
                raise ValueError(f"invalid registered project: {path}")
            projects.add(child)
        return sorted(projects)

    async def resources(self, projects, *, remove_volumes):
        if not isinstance(projects, list) or not projects or type(remove_volumes) is not bool:
            raise ValueError("resource lookup requires projects and a boolean volume policy")
        result = {"network": [], "volume": []}
        for owner in projects:
            self.path(owner)
            for kind in result:
                if kind == "volume" and not remove_volumes:
                    continue
                listing = await self.command(
                    self.host, kind, "ls", "-q", "--filter", f"label=com.docker.compose.project={owner}"
                )
                for resource in listing.splitlines():
                    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", resource):
                        raise ValueError(f"invalid {kind} resource: {resource!r}")
                    result[kind].append(resource)
        return result

    def terminal_policy(self, project):
        self.path(project)
        # Harbor's main session is <trial_name>__env; the parent owns trial_name.
        for owner in (project, project + "__env"):
            if not PROJECT.fullmatch(owner):
                continue
            path = self.path(owner)
            if path.exists():
                row = self.read(path)
                return {"remove": row["remove"], "remove_volumes": row["remove_volumes"]}
        return {"remove": True, "remove_volumes": True}

    async def drain_once(self):
        await asyncio.to_thread(self.index_journal)
        batch = await asyncio.to_thread(lambda: list(islice(self._journal_scan, 256)))
        if not batch:
            self._journal_scan = self.pending.glob("*.json")
            batch = await asyncio.to_thread(lambda: list(islice(self._journal_scan, 256)))
        semaphore = asyncio.Semaphore(self.concurrency)

        async def drain_marker(marker):
            if not isinstance(marker, Path) or marker.parent != self.pending:
                raise ValueError(f"invalid pending cleanup marker: {marker}")
            async with semaphore:
                path = self.directory / marker.name
                row = await asyncio.to_thread(self.read, path)
                if row["phase"] in {"done", "needs_attention"}:
                    await asyncio.to_thread(self.index_record, path, row)
                elif row["next_retry"] <= time.time():
                    await self.reconcile(row["project"], disk=True, include_children=row["include_children"])

        results = await asyncio.gather(*(drain_marker(marker) for marker in batch), return_exceptions=True)
        # Reap every operation before reporting failure; no detached delete tasks.
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def reconcile(self, project, *, disk=False, include_children=False):
        if type(disk) is not bool:
            raise TypeError("disk must be boolean")
        with self.locked(project, include_children=include_children) as acquired, ExitStack() as resource_locks:
            if not acquired:
                return None
            row = self.read(self.path(project, include_children=include_children))
            if row["phase"] == "delete_inflight":
                row["phase"] = "needs_attention"
                row["errors"].append("previous worker exited during deletion; inspect daemon before retry")
                await self.persist(row)
            if row["phase"] in {"done", "needs_attention"}:
                return row
            projects = await asyncio.to_thread(self.registered_projects, project) if include_children else [project]
            if not resource_locks.enter_context(self.locked_resources(projects)):
                return None
            row["attempts"] += 1
            row["errors"] = []
            try:
                blocked = await asyncio.to_thread(
                    self.unresolved_resource_records,
                    projects,
                    exclude=self.path(project, include_children=include_children),
                )
                if blocked:
                    raise DockerFailure(f"owned resources have unresolved cleanup records: {blocked}")
                found, owners = await self.containers(project, include_children=include_children)
                if include_children:
                    owners.update(await asyncio.to_thread(self.registered_projects, project))
                row["resource_projects"] = sorted(set(row.get("resource_projects", [])) | owners | {project})
                row["containers"] = sorted(set(row["containers"]) | set(found))
                await self.persist(row)  # Persist ownership before issuing any stop/delete.
                # Every container gets a stop attempt, even if another call fails.
                for cid, state in found.items():
                    if state in STOPPED:
                        continue
                    try:
                        await self.command(self.host, "kill", cid)
                    except DockerFailure as exc:
                        row["errors"].append(str(exc))
                remaining = found
                if any(state not in STOPPED for state in found.values()):
                    remaining, _ = await self.containers(project, include_children=include_children)
                live = {cid: state for cid, state in remaining.items() if state not in STOPPED}
                if live:
                    row["phase"] = "stop_pending"
                    raise DockerFailure(f"project still has live containers: {live}")
                row["phase"] = "disk_pending" if row["remove"] else "done"
                # Resource discovery may fail independently of container stops.
                # Stop every live container first; persist resources before deletion.
                row["resources"] = (
                    await self.resources(row["resource_projects"], remove_volumes=row["remove_volumes"])
                    if row["remove"]
                    else {}
                )
                if not remaining and not any(row["resources"].values()):
                    row["phase"] = "done"
                await self.persist(row)
                if not disk or row["phase"] == "done":
                    logger.info("environment cleanup state: %s", json.dumps(row))
                    return row
                # No force deletion of live containers. A removal timeout is
                # uncertain server-side work, so never blindly enqueue it again.
                for cid in remaining:
                    row["phase"] = "delete_inflight"
                    await self.persist(row)
                    try:
                        await self.command(self.host, "rm", cid, timeout=120)
                    except DockerFailure as exc:
                        row["errors"].append(str(exc))
                        if exc.uncertain:
                            row["phase"] = "needs_attention"
                            break
                    row["phase"] = "disk_pending"
                    await self.persist(row)
                if row["phase"] != "needs_attention":
                    remaining, _ = await self.containers(project, include_children=include_children)
                    if not remaining:
                        # Only exact project labels; never global prune/images.
                        for kind, resources in row["resources"].items():
                            for resource in resources:
                                row["phase"] = "delete_inflight"
                                await self.persist(row)
                                try:
                                    await self.command(self.host, kind, "rm", resource, timeout=120)
                                except DockerFailure:
                                    row["phase"] = "disk_pending"
                                    raise
                                row["phase"] = "disk_pending"
                                await self.persist(row)
                        row["phase"] = "done"
            except DockerFailure as exc:
                row["errors"].append(str(exc))
                if disk and exc.uncertain:
                    row["phase"] = "needs_attention"
            except asyncio.CancelledError:
                # Persist a conservative state if a daemon mutation may outlive
                # this process. Restart must not multiply unresolved deletes.
                if disk:
                    row["phase"] = "needs_attention"
                row["errors"].append("cleanup interrupted; inspect outstanding Docker operations before retry")
                await self.persist(row)
                raise
            finally:
                row["next_retry"] = time.time() + min(300, 30 * row["attempts"])
                await self.persist(row)
            if row["errors"]:
                logger.error("environment cleanup %s: %s", project, json.dumps(row))
            return row


def configured_queue():
    """Use the runtime Docker daemon, with a host-scoped durable journal.

    Prepared Harbor sources can be shared by multiple machines. Their baked
    settings must never override the daemon used by the running controller.
    """
    path = Path(__file__).with_name("cleanup_settings.json")
    settings = json.loads(path.read_text())
    if not isinstance(settings, dict) or set(settings) != {"directory", "host"}:
        raise ValueError(f"invalid cleanup settings: {path}")
    host = os.environ.get("DOCKER_HOST")
    if not isinstance(host, str) or not host.startswith("unix:///") or Path(host[7:]) == Path("/"):
        raise ValueError(f"cleanup requires an absolute Unix DOCKER_HOST: {host!r}")
    mounted_socket = os.environ.get("MILES_SWE_DOCKER_SOCKET")
    if mounted_socket is not None and "unix://" + mounted_socket != host:
        raise ValueError(f"Docker runtime/socket mismatch: {host!r} vs {mounted_socket!r}")
    directory = Path(settings["directory"])
    if not directory.is_absolute():
        raise ValueError(f"cleanup directory must be absolute: {directory}")
    identity = hashlib.sha256(f"{socket.gethostname()}\0{host}".encode()).hexdigest()[:16]
    return CleanupQueue(directory / ("runtime-" + identity), host)


async def register_project(project):
    queue = configured_queue()
    queue.path(project)
    task = asyncio.create_task(asyncio.to_thread(queue.register_project, project))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class CleanupPending(RuntimeError):
    """Stopping failed after cleanup ownership was durably enqueued."""


async def finish_project(project, *, remove=True, remove_volumes=True, include_children=False):
    queue = configured_queue()
    await queue.enqueue_async(project, remove=remove, remove_volumes=remove_volumes, include_children=include_children)
    row = await queue.reconcile(project, include_children=include_children)
    if row is None:
        row = queue.read(queue.path(project, include_children=include_children))
    if row["phase"] == "stop_pending":
        raise CleanupPending(f"container stop not verified; durable cleanup pending: {project}")
    return row


async def cleanup_loop():
    queue = configured_queue()
    while True:
        await queue.drain_once()
        await asyncio.sleep(5)


async def finish_trial(project):
    """Parent-process fallback after success, cancellation, or worker failure."""
    queue = configured_queue()
    previous = queue.terminal_policy(project)
    return await finish_project(
        project, remove=previous["remove"], remove_volumes=previous["remove_volumes"], include_children=True
    )


def report_worker_exit(task):
    if not isinstance(task, asyncio.Task):
        raise TypeError("cleanup worker callback requires an asyncio Task")
    if not task.cancelled():
        error = task.exception()
        logger.critical("persistent cleanup worker exited; queued records retained: %r", error)
