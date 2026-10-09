"""Serve checkpoint evaluation sessions through an existing SGLang endpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4


def build_session_args(checkpoint: Path, host: str, port: int, timeout: float) -> SimpleNamespace:
    if not isinstance(checkpoint, Path) or not checkpoint.is_dir():
        raise ValueError("checkpoint must be an existing directory")
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(checkpoint / filename)
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be nonempty")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in [1, 65535]")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    roles = ["tool", "user"]
    template = Path(__file__).resolve().parents[3] / "miles/utils/chat_template_utils/templates/qwen3.5_fixed.jinja"
    kwargs = {"clear_thinking": False}
    if not template.is_file():
        raise ValueError("Qwen3.5 evaluation requires the bundled append-safe chat template")
    return SimpleNamespace(
        hf_checkpoint=str(checkpoint.resolve()),
        session_server_ip=host,
        session_server_port=port,
        session_server_instance_id=uuid4().hex,
        tito_model="qwen35",
        tito_allowed_append_roles=roles,
        chat_template_path=str(template),
        apply_chat_template_kwargs=kwargs,
        miles_router_timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend-url", required=True, help="SGLang server URL, e.g. http://127.0.0.1:30000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    url = urlsplit(args.backend_url)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in ("", "/")):
        raise ValueError("backend-url must be an HTTP(S) server origin without credentials")
    session_args = build_session_args(args.checkpoint.expanduser(), args.host, args.port, args.timeout)
    from miles.rollout.session.session_server import run_session_server

    run_session_server(session_args, args.backend_url.rstrip("/"))


if __name__ == "__main__":
    main()
