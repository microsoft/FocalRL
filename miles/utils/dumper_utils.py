from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import os
import shutil
from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_sglang_dumper_module: Any | None = None
DumperConfig: Any | None = None


def _load_sglang_dumper() -> Any:
    global DumperConfig, _sglang_dumper_module
    if _sglang_dumper_module is None:
        os.environ.setdefault("SGLANG_USE_AITER", "0")
        os.environ.setdefault("SGLANG_USE_AITER_AR", "0")
        os.environ.setdefault("USE_ROCM_AITER_ROPE_BACKEND", "0")
        from sglang.srt.debug_utils import dumper as loaded

        _sglang_dumper_module = loaded
        DumperConfig = getattr(loaded, "DumperConfig", None) or getattr(loaded, "_DumperConfig")
    return _sglang_dumper_module


def _get_dumper_config_cls() -> Any:
    _load_sglang_dumper()
    return DumperConfig


def _get_rank() -> int:
    return _load_sglang_dumper()._get_rank()


class _LazyDumperProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(_load_sglang_dumper().dumper, name)


dumper = _LazyDumperProxy()


class DumperPhase(enum.Enum):
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


# ------------------------------- SGLang -------------------------------------


def get_sglang_env(args: Namespace) -> dict[str, str]:
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return {}

    env: dict[str, str] = {"DUMPER_SERVER_PORT": "reuse"}
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)

    # SGLang registers non-intrusive hooks while loading the model. Configs that
    # affect hook registration must be present in the actor environment; the
    # later HTTP configure call only controls active dumping/output location.
    if non_intrusive_mode := overrides.get("non_intrusive_mode"):
        env["DUMPER_NON_INTRUSIVE_MODE"] = str(non_intrusive_mode)

    if source_patcher_config := args.dumper_source_patcher_config_inference:
        env["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config
    elif source_patcher_config := overrides.get("source_patcher_config"):
        env["DUMPER_SOURCE_PATCHER_CONFIG"] = str(source_patcher_config)

    return env


async def configure_sglang(args: Namespace) -> None:
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return

    from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
    from miles.utils.http_utils import post

    worker_urls = await get_worker_urls(args)
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)

    engines_dir: Path = _get_dir(args) / "engines"
    _cleanup_dump_dir(engines_dir)

    coros = []
    for i, url in enumerate(worker_urls):
        body = {
            "enable": True,
            "dir": str(_get_dir(args)),
            "exp_name": f"engines/engine_{i}",
            **overrides,
        }
        coros.append(post(f"{url}/dumper/configure", body))

    await asyncio.gather(*coros)
    logger.info("Configured dumper on %d SGLang engines", len(worker_urls))


# ------------------------------- Megatron -------------------------------------


class DumperMegatronUtil:
    def __init__(self, args: Namespace, model: Sequence[torch.nn.Module], phase: DumperPhase) -> None:
        self.phase = phase
        self.overrides = _get_phase_override_configs(args, phase)
        self.enabled = self._configure(args, phase, self.overrides)
        if self.enabled:
            dumper.register_non_intrusive_dumper(self._extract_model(model))

    def wrap_forward_step(self, forward_step_func: Callable) -> Callable:
        if not self.enabled:
            return forward_step_func

        return _wrap_forward_step_with_stepping(forward_step_func)

    def finalize(self, model: Sequence[torch.nn.Module]) -> None:
        if not self.enabled:
            return

        extracted_model = self._extract_model(model)
        if self.phase is DumperPhase.FWD_BWD and self.overrides.get("enable_model_grad"):
            _log_model_grad_coverage(extracted_model)

        dumper.dump_model(extracted_model)
        dumper.step()
        dumper.configure(enable=False)

    @staticmethod
    def _extract_model(model: Sequence[torch.nn.Module]) -> torch.nn.Module:
        assert (
            len(model) == 1
        ), f"Dumper does not yet support virtual pipeline parallelism (got {len(model)} model chunks)"
        return model[0]

    @staticmethod
    def _configure(args: Namespace, phase: DumperPhase, overrides: dict[str, Any] | None = None) -> bool:
        if overrides is None:
            overrides = _get_phase_override_configs(args, phase)
        if not overrides.get("enable"):
            return False

        merged = {
            "dir": str(_get_dir(args)),
            "exp_name": phase.value,
            **overrides,
        }

        full_config = _get_dumper_config_cls()(**merged)
        dumper.reset()
        _cleanup_dump_dir(Path(merged["dir"]) / merged["exp_name"])
        dumper.configure(**dataclasses.asdict(full_config))
        return True


def _log_model_grad_coverage(model: torch.nn.Module) -> None:
    missing: list[str] = []
    with_grad = 0
    total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        total += 1
        grad = param.grad if param.grad is not None else getattr(param, "main_grad", None)
        if grad is None:
            missing.append(name)
        else:
            with_grad += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else _get_rank()
    logger.info(
        "Dumper fwd_bwd model grad coverage rank=%s with_grad=%d total=%d missing=%d missing_names=%s",
        rank,
        with_grad,
        total,
        len(missing),
        missing[:20],
    )


def _wrap_forward_step_with_stepping(forward_step_func: Callable) -> Callable:
    is_first_call = True

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal is_first_call
        if not is_first_call:
            dumper.step()
        is_first_call = False
        return forward_step_func(*args, **kwargs)

    return _wrapped


# ------------------------------- Common -------------------------------------


def _cleanup_dump_dir(dump_dir: Path) -> None:
    if _get_rank() == 0 and dump_dir.is_dir():
        shutil.rmtree(dump_dir)
    if dist.is_initialized():
        dist.barrier()


def _get_phase_override_configs(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    raw = getattr(args, f"dumper_{phase.value}")
    return {"enable": args.dumper_enable, **_dumper_kv_pairs_to_dict(raw)}


def _is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    return _get_phase_override_configs(args, phase).get("enable", False)


def _get_dir(args: Namespace) -> Path:
    return Path(args.dumper_dir)


def _dumper_kv_pairs_to_dict(raw: Sequence[str] | None) -> dict[str, Any]:
    if raw is None:
        return {}

    upstream_parser = getattr(DumperConfig, "_kv_pairs_to_dict", None) if DumperConfig is not None else None
    if upstream_parser is not None:
        return upstream_parser(raw)

    fields = getattr(DumperConfig, "__dataclass_fields__", {}) if DumperConfig is not None else {}
    parsed: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Dumper config entry must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        default = fields[key].default if key in fields else None
        parsed[key] = _parse_dumper_value(value, default)
    return parsed


def _parse_dumper_value(value: str, default: Any) -> Any:
    if isinstance(default, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value
