"""Patch an isolated Harbor server to support reconnectable /run requests."""

import argparse
from pathlib import Path

from adaptive_branching.tools.swe.prepare_harbor_cleanup import replace_once

MARKER = "# Reconnectable Harbor runs v1"


def patch_source(source):
    if not isinstance(source, str) or not source or MARKER in source:
        raise ValueError("requires nonempty, unpatched Harbor server source")
    source = replace_once(
        source,
        "from agent_server.state import _state\n",
        "from agent_server.state import _state\n"
        "from adaptive_branching.src.swe.harbor_durable_run import DurableRuns, HEADER, PROTOCOL\n"
        "_durable_runs = None\n",
    )
    source = replace_once(
        source,
        "    _state.semaphore = asyncio.Semaphore(max_concurrent)\n",
        "    _state.semaphore = asyncio.Semaphore(max_concurrent)\n"
        "    global _durable_runs\n"
        "    if _state.trials_dir is None:\n"
        "        raise RuntimeError('durable runs require configured trials_dir')\n"
        "    _durable_runs = DurableRuns(Path(_state.trials_dir).parent / 'durable-runs.sqlite')\n",
    )
    source = replace_once(
        source,
        '@app.post("/run")\nasync def run_instance(',
        'async def _run_instance_once(',
    )
    source = replace_once(
        source,
        '@app.get("/health")\nasync def health():\n    return {"status": "ok"}\n',
        '''@app.post("/run")
async def run_instance(request: RunRequest, raw_request: Request) -> RunResponse:
    key = raw_request.headers.get(HEADER)
    if key is None:
        return await _run_instance_once(request, raw_request)
    _require_admin_secret(raw_request.headers.get("authorization"))
    if _durable_runs is None:
        raise HTTPException(503, "durable runs not initialized")
    return await _durable_runs.run(key, request.model_dump(), lambda: _run_instance_once(request, raw_request))


@app.get("/health")
async def health():
    return {"status": "ok", PROTOCOL: _durable_runs is not None}
''',
    )
    source += "\n" + MARKER + "\n"
    compile(source, "miles_agent_server.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.server_file.is_file():
        raise ValueError(f"missing server: {args.server_file}")
    patched = patch_source(args.server_file.read_text())
    args.server_file.write_text(patched)


if __name__ == "__main__":
    main()
