import logging
import os

import torch.distributed as dist
import torch.nn as nn

from miles_plugins.models.hf_attention import HuggingfaceAttention

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def detect_and_setup_hybrid_cp(model: nn.Module, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> int:
    """Scan for GatedDeltaNet modules and configure them for native fla CP."""
    if _env_flag("MILES_DISABLE_GDN_HYBRID_CP"):
        logger.info("Skipping GDN hybrid CP setup because MILES_DISABLE_GDN_HYBRID_CP=1")
        return 0

    count = 0
    for module in model.modules():
        if isinstance(module, HuggingfaceAttention):
            linear_attn = getattr(module, "linear_attn", None)
            if linear_attn is not None:
                linear_attn.cp_group = cp_group
                linear_attn.cp_rank = cp_rank
                linear_attn.cp_world_size = cp_world_size
                module.hybrid_cp = True
                count += 1

    if count > 0:
        logger.info(f"Configured hybrid CP on {count} GDN modules (fla native state passing)")
    return count
