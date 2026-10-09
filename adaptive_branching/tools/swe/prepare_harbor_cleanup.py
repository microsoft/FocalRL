"""Prepare an isolated Harbor tree with durable terminal-project cleanup."""

import argparse
import ast
import json
import shutil
from pathlib import Path

from adaptive_branching.src.swe import harbor_cleanup

MARKER = "# Durable terminal-project cleanup v1"


def replace_once(source, old, new):
    if not isinstance(source, str) or not old or source.count(old) != 1:
        raise ValueError(f"unexpected Harbor source anchor: {old[:100]!r}")
    return source.replace(old, new, 1)


def patch_source(name, source):
    if not isinstance(source, str) or not source or MARKER in source:
        raise ValueError("empty or already patched Harbor source")
    if name == "src/harbor/environments/docker/docker-compose-egress-control.yaml":
        source = replace_once(
            source, "      interval: 1s\n", "      interval: 30s\n      start_interval: 1s\n      start_period: 30s\n"
        )
        return source + "\n" + MARKER + "\n"
    tree = ast.parse(source)
    if name == "src/harbor/environments/docker/docker.py":
        starts = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "start"]
        if len(starts) != 1 or ast.unparse(starts[0].args) != "self, force_build: bool":
            raise ValueError("unexpected Docker.start signature")
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "stop"]
        if len(matches) != 1 or ast.unparse(matches[0].args) != "self, delete: bool":
            raise ValueError("unexpected Docker.stop signature")
        node = matches[0]
        old = "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
        if 'self.logger.warning(f"Docker compose down failed: {e}")' not in old:
            raise ValueError("unexpected Docker.stop body")
        new = """    async def stop(self, delete: bool):
        from agent_server.durable_cleanup import CleanupPending, finish_project

        if type(delete) is not bool:
            raise TypeError("delete must be boolean")
        # Bound the existing best-effort log ownership fix. Always stop every
        # service, even when this ancillary exec fails or gets cancelled.
        try:
            try:
                await asyncio.wait_for(self.prepare_logs_for_host(), timeout=10)
            except Exception:
                self.logger.exception("log ownership preparation failed; continuing container stop")
        finally:
            try:
                await finish_project(
                    _sanitize_docker_compose_project_name(self.session_id),
                    remove=not self._keep_containers,
                    remove_volumes=delete,
                )
            except CleanupPending:
                self.logger.exception("environment cleanup durably pending; preserving trial outcome")
            finally:
                self._cleanup_mounts_compose_file()
                self._cleanup_resources_compose_file()
                self._cleanup_env_compose_file()
                self._cleanup_egress_control_services_compose_file()
                self._cleanup_egress_control_base_compose_file()
"""
        source = replace_once(source, old, new)
        source = replace_once(
            source,
            "    async def start(self, force_build: bool):\n",
            """    async def start(self, force_build: bool):
        from agent_server.durable_cleanup import register_project
        await register_project(_sanitize_docker_compose_project_name(self.session_id))
""",
        )
    elif name == "agent_server/trial_runner.py":
        source = replace_once(
            source,
            "        proc.kill()\n        await proc.wait()\n",
            """        try:
            if project is not None:
                from agent_server.durable_cleanup import configured_queue
                queue = configured_queue()
                previous = queue.terminal_policy(project)
                await queue.enqueue_async(project, remove=previous["remove"], remove_volumes=previous["remove_volumes"], include_children=True)
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
""",
        )
        old = """            cleanup = await _kill_trial_projects({project})
            if cleanup["errors"]:
                logger.error("trial cancellation cleanup failed: %s", cleanup["errors"])
"""
        source = replace_once(
            source,
            old,
            """            from agent_server.durable_cleanup import CleanupPending, finish_trial
            try:
                await finish_trial(project)
            except CleanupPending:
                logger.exception("terminal project cleanup pending after cancellation")
""",
        )
        source = replace_once(
            source,
            '    for line in reversed(out.decode(errors="replace").splitlines()):\n',
            """    if project is not None:
        from agent_server.durable_cleanup import CleanupPending, finish_trial
        try:
            cleanup_status = await finish_trial(project)
        except CleanupPending:
            # Only a durably queued stop failure can be deferred. Configuration,
            # journal corruption and persistence failures still fail explicitly.
            logger.exception("terminal project cleanup pending; preserving trial result")
        else:
            logger.info("terminal project cleanup status: %s", json.dumps(cleanup_status))
    for line in reversed(out.decode(errors="replace").splitlines()):
""",
        )
    elif name == "miles_agent_server.py":
        source = replace_once(
            source,
            "    checker_task = asyncio.create_task(_health_checker_loop())\n",
            """    from agent_server.durable_cleanup import cleanup_loop, configured_queue, report_worker_exit
    configured_queue()  # Fail startup on invalid/missing persistent configuration.
    cleanup_task = asyncio.create_task(cleanup_loop())
    cleanup_task.add_done_callback(report_worker_exit)
    checker_task = asyncio.create_task(_health_checker_loop())
""",
        )
        source = replace_once(
            source,
            "        checker_task.cancel()\n",
            """        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        checker_task.cancel()
""",
        )
    else:
        raise ValueError(f"unsupported Harbor file: {name}")
    source += "\n" + MARKER + "\n"
    compile(source, name, "exec")
    return source


def prepare(source, destination, *, queue_dir, docker_host):
    source, destination, queue_dir = Path(source).resolve(), Path(destination).resolve(), Path(queue_dir)
    if not source.is_dir() or destination.exists() or source == destination or source in destination.parents:
        raise ValueError("requires existing source and new external destination")
    if not queue_dir.is_absolute() or queue_dir == destination or destination in queue_dir.parents:
        raise ValueError("queue_dir must be absolute and persist outside the prepared source")
    if not isinstance(docker_host, str) or not docker_host.startswith("unix:///"):
        raise ValueError("an explicit absolute Unix Docker socket is required")
    names = (
        "src/harbor/environments/docker/docker.py",
        "src/harbor/environments/docker/docker-compose-egress-control.yaml",
        "agent_server/trial_runner.py",
        "miles_agent_server.py",
    )
    patched = {name: patch_source(name, (source / name).read_text()) for name in names}
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    for name, content in patched.items():
        (destination / name).write_text(content)
    shutil.copyfile(harbor_cleanup.__file__, destination / "agent_server/durable_cleanup.py")
    (destination / "agent_server/cleanup_settings.json").write_text(
        json.dumps({"directory": str(queue_dir), "host": docker_host}, indent=2) + "\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "queue-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--docker-host", required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.destination, queue_dir=args.queue_dir, docker_host=args.docker_host))


if __name__ == "__main__":
    main()
