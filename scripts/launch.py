"""FocalRL training launcher. Preview by default; --launch uses an existing Ray cluster."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def model_args(size: str) -> list[str]:
    if size not in {"4B", "9B"}:
        raise ValueError("model size must be 4B or 9B")
    path = ROOT / f"scripts/models/qwen3.5-{size}.sh"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text().strip()
    if not text.startswith("MODEL_ARGS=(") or not text.endswith(")"):
        raise ValueError(f"unexpected model argument file: {path}")
    return shlex.split(text[len("MODEL_ARGS=("):-1], comments=True)


def recipe(domain: str) -> dict:
    if domain not in {"swe", "deep_research"}:
        raise ValueError("domain must be swe or deep_research")
    return dict(domain=domain, full_groups=16, local_groups=32 if domain == "swe" else 48,
                samples_per_group=8, local_horizon=20 if domain == "swe" else 5,
                steps=40 if domain == "swe" else 80, max_seq_len=262144, learning_rate=1e-6)


def required_env(name: str, *, url: bool = False) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"required environment variable {name} is missing")
    if url:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f"{name} must be an HTTP(S) URL without embedded credentials")
    return value


def build(args) -> tuple[list[str], dict[str, str]]:
    cfg = recipe(args.domain)
    for name in ("actor_gpus", "rollout_gpus", "rollout_tp", "context_parallel", "gpus_per_node"):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.actor_gpus > args.gpus_per_node or args.actor_gpus % args.context_parallel:
        raise ValueError("actor GPUs must fit one node and divide evenly by context parallelism")
    if args.rollout_gpus % args.rollout_tp or args.gpus_per_node % args.rollout_tp or args.actor_gpus % args.rollout_tp:
        raise ValueError("rollout TP must divide rollout GPUs, GPUs per node, and actor GPU offset")
    for name in ("hf_checkpoint", "ref_load"):
        value = getattr(args, name)
        if value is None or not value.is_dir():
            raise ValueError(f"{name} must be an existing checkpoint directory")
    if not (args.hf_checkpoint / "config.json").is_file():
        raise FileNotFoundError("HF checkpoint requires config.json")
    if args.prompt_data is None or not args.prompt_data.is_file() or args.prompt_data.stat().st_size == 0:
        raise ValueError("prompt_data must be a nonempty JSONL file")
    if args.save is None or args.save.exists():
        raise ValueError("save must be a new output directory; explicit resume is not supported by this launcher")
    # Validate every record before submitting GPU work; a gold patch stays controller-only.
    for index, line in enumerate(args.prompt_data.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{args.prompt_data}:{index}: invalid JSON") from exc
        key = "prompt" if args.domain == "swe" else "question"
        if not isinstance(row, dict) or not isinstance(row.get(key), (str, list)) or not row[key]:
            raise ValueError(f"{args.prompt_data}:{index}: missing nonempty {key}")
        if args.domain == "swe":
            metadata = row.get("metadata")
            if not isinstance(metadata, dict) or not all(isinstance(metadata.get(k), str) and metadata[k].strip()
                                                       for k in ("instance_id", "ab_swe_golden_patch")):
                raise ValueError(f"{args.prompt_data}:{index}: requires instance_id and ab_swe_golden_patch")
        elif not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError(f"{args.prompt_data}:{index}: requires answer")
    groups = cfg["full_groups"] + cfg["local_groups"]
    swe = args.domain == "swe"
    package = "adaptive_branching.src."
    agent = package + ("swe.lightning_local.run" if swe else "deep_research.agent.run")
    reward = package + ("swe.lightning_local.reward_func" if swe else "deep_research.reward_function.reward_func")
    generate = package + "swe.lightning_generate.generate" if swe else "miles.rollout.generate_hub.agentic_tool_call.generate"
    command = [sys.executable, str(ROOT / "train_async.py"), *model_args(args.model_size)]
    options = {
        "hf-checkpoint": args.hf_checkpoint.resolve(), "ref-load": args.ref_load.resolve(),
        "save": args.save.resolve(), "save-interval": 20,
        "prompt-data": args.prompt_data.resolve(), "input-key": "prompt" if swe else "question",
        "metadata-key": "metadata",
        "num-rollout": cfg["steps"], "rollout-batch-size": groups,
        "n-samples-per-prompt": cfg["samples_per_group"], "global-batch-size": groups * cfg["samples_per_group"],
        "max-seq-len": cfg["max_seq_len"], "rollout-max-response-len": 12288 if swe else 22000,
        "rollout-temperature": 1.0, "rollout-seed": 42,
        "rollout-function-path": package + "deep_research.fully_async_value_cliff_collector.generate_rollout_fully_async",
        "custom-generate-function-path": generate, "custom-agent-function-path": agent,
        "custom-rm-path": reward, "reward-key": "score",
        "custom-config-path": ROOT / "examples/train_infer_mismatch_helper/mis.yaml",
        "custom-tis-function-path": "examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp",
        "dynamic-sampling-filter-path": "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted",
        "pause-generation-mode": "retract", "tito-model": "qwen35",
        "advantage-estimator": "grpo", "kl-loss-coef": 0.0, "kl-loss-type": "low_var_kl",
        "entropy-coef": 0.0, "eps-clip": 0.2, "eps-clip-high": 0.28,
        "optimizer": "adam", "lr": cfg["learning_rate"], "lr-decay-style": "constant",
        "weight-decay": 0.1, "adam-beta1": 0.9, "adam-beta2": 0.98,
        "tensor-model-parallel-size": 1, "pipeline-model-parallel-size": 1,
        "context-parallel-size": args.context_parallel, "expert-model-parallel-size": 1,
        "expert-tensor-parallel-size": 1, "recompute-granularity": "full",
        "recompute-method": "uniform", "recompute-num-layers": 1,
        "max-tokens-per-gpu": 32768, "log-probs-chunk-size": 256,
        "actor-num-nodes": 1, "actor-num-gpus-per-node": args.actor_gpus,
        "rollout-num-gpus": args.rollout_gpus, "num-gpus-per-node": args.gpus_per_node,
        "rollout-num-gpus-per-engine": args.rollout_tp,
        "sglang-context-length": cfg["max_seq_len"], "sglang-mem-fraction-static": 0.85,
        "sglang-server-concurrency": 64, "sglang-max-running-requests": 64,
        "sglang-chunked-prefill-size": 65536, "sglang-max-mamba-cache-size": 1024,
        "sglang-tool-call-parser": "qwen3_coder", "sglang-reasoning-parser": "qwen3",
        "apply-chat-template-kwargs": '{"enable_thinking":true}',
        "attention-dropout": 0.0, "hidden-dropout": 0.0,
    }
    if not swe:
        options["label-key"] = "answer"
    for key, value in options.items():
        command += ["--" + key, str(value)]
    command += ["--" + flag for flag in (
        "group-rm", "rollout-shuffle", "balance-data", "use-miles-router", "use-session-server",
        "partial-rollout", "calculate-per-token-loss", "disable-grpo-std-normalization", "use-kl-loss",
        "sequence-parallel", "recompute-loss-function", "use-dynamic-batch-size",
        "accumulate-allreduce-grads-in-fp32", "attention-softmax-in-fp32",
        "sglang-disable-overlap-schedule", "sglang-disable-custom-all-reduce")]
    command += ["--tito-allowed-append-roles", "tool", "user"]
    env = {
        "AB_LOCAL_ROLLOUT_ENABLE": "1", "AB_BRANCH_SELECTION": "value_cliff",
        "AB_LOCAL_REWARD_MODE": "v6_prm", "AB_FULL_TRAIN_GROUPS_PER_BATCH": str(cfg["full_groups"]),
        "AB_LOCAL_TRAIN_GROUPS_PER_BATCH": str(cfg["local_groups"]),
        "AB_EVENT_HIDDEN_MAX_TURNS": str(cfg["local_horizon"]), "AB_LOCAL_MAX_GROUPS_PER_FULL_GROUP": "8",
        "AB_BALANCED_GENERATION": "0", "AGENT_MAX_TURNS": "200", "AGENT_CONTEXT_RESERVE_TOKENS": "32768",
        "SWE_LIGHTNING_BUDGET": "large", "SWE_PRM_BEHAVIOR_VETO": "0",
        "FORCE_EXCLUDE": "NONE" if swe else "Wrong", "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_JUDGE_CONFIG": str(ROOT / "adaptive_branching/config/judge.yaml"),
        "AGENT_TOOLS_CONFIG": str(ROOT / "adaptive_branching/config/microsoft_gaia_tools.yaml"),
        "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    return command, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("deep_research", "swe"))
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--model-size", choices=("4B", "9B"), default="4B")
    for name in ("hf-checkpoint", "ref-load", "prompt-data", "save"):
        parser.add_argument("--" + name, type=Path)
    for name, default in (("actor-gpus", 4), ("rollout-gpus", 12), ("rollout-tp", 4),
                          ("context-parallel", 4), ("gpus-per-node", 8)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    if args.describe:
        if args.launch:
            parser.error("--describe and --launch are mutually exclusive")
        print(json.dumps(recipe(args.domain), indent=2))
        return
    command, env = build(args)
    if not args.launch:
        print(json.dumps({"command": command, "public_environment": env}, indent=2))
        return
    names = {"LLM_JUDGE_KEY": False, "LLM_JUDGE_URL": True}
    if args.domain == "deep_research":
        names.update(MS_API_KEYS=False, LLM_API_KEY=False, SEARCH_ENDPOINT=True,
                     BROWSE_ENDPOINT=True, BROWSER_LLM_URL=True)
    else:
        names.update(HARBOR_SERVER_URL=True, MILES_ROUTER_EXTERNAL_HOST=False)
    for name, is_url in names.items():
        env[name] = required_env(name, url=is_url)
    for name in ("HARBOR_ADMIN_SECRET", "MILES_ROUTER_EXTERNAL_PORT", "AGENT_MODEL_NAME", "JINA_BASE_URL",
                 "SGLANG_USE_AITER", "SGLANG_USE_AITER_AR", "USE_ROCM_AITER_ROPE_BACKEND",
                 "AITER_USE_SYSTEM_TRITON", "SGLANG_USE_ROCM700A", "TORCHDYNAMO_DISABLE"):
        if name in os.environ:
            env[name] = os.environ[name]
    sys.path.insert(0, str(ROOT))
    import ray
    # Propagate keys in runtime_env, never through argv or preview output.
    ray.init(address="auto", runtime_env={"env_vars": env})
    os.environ.update(env)
    sys.argv = command[1:]
    runpy.run_path(str(ROOT / "train_async.py"), run_name="__main__")


if __name__ == "__main__":
    main()
