import socket
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import init_process_group

from .mixin import DistBucketedWeightUpdateMixin


class UpdateWeightFromDistributed(DistBucketedWeightUpdateMixin):
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self._engine_group_names = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        pp_rank = get_parallel_state().pp.rank
        if self._is_source:
            self._group_name = f"miles-pp_{pp_rank}"

        if self._is_source:
            if (g := self._model_update_groups) is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._engine_group_names, g, self.rollout_engines
                )
            self._model_update_groups, self._engine_group_names = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines, engine_gpu_counts=engine_gpu_counts
            )

    @property
    def _is_source(self):
        """If it's the source gpu that broadcasting weights to rollout side"""
        return get_parallel_state().intra_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Lock → broadcast → clear → unlock. Lock prevents NCCL deadlock."""
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        refs = update_weights_from_distributed(
            self._engine_group_names,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
        )
        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        if pbar:
            pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> tuple[list[dist.ProcessGroup], list[str]]:
    """
    Create one NCCL group per rollout engine: training rank 0 + that engine's GPUs.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()

    socks = [socket.socket() for _ in rollout_engines]
    for sock in socks:
        sock.bind(("", 0))
    master_ports = [sock.getsockname()[1] for sock in socks]
    group_names = [f"{group_name}_eng_{i}" for i in range(len(rollout_engines))]

    refs = []
    for engine, engine_gpu_count, master_port, engine_group_name in zip(
        rollout_engines, engine_gpu_counts, master_ports, group_names, strict=True
    ):
        refs.append(
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                1,
                engine_gpu_count + 1,
                engine_group_name,
                backend="nccl",
            )
        )
    for sock in socks:
        sock.close()

    model_update_groups = []
    for engine_gpu_count, master_port, engine_group_name in zip(
        engine_gpu_counts, master_ports, group_names, strict=True
    ):
        model_update_groups.append(
            init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=engine_gpu_count + 1,
                rank=0,
                group_name=engine_group_name,
            )
        )
    ray.get(refs)
    return model_update_groups, group_names


def disconnect_rollout_engines_from_distributed(args, group_names, model_update_groups, rollout_engines):
    """
    Destroy NCCL groups on training and engines.
    """
    assert len(group_names) == len(model_update_groups) == len(rollout_engines), (
        f"length mismatch: {len(group_names)=} {len(model_update_groups)=} {len(rollout_engines)=}"
    )
    refs = [
        engine.destroy_weights_update_group.remote(group_name)
        for engine, group_name in zip(rollout_engines, group_names, strict=True)
    ]
    for group in model_update_groups:
        dist.destroy_process_group(group)
    ray.get(refs)


def update_weights_from_distributed(
    group_names: Sequence[str],
    groups: Sequence[dist.ProcessGroup],
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    assert len(group_names) == len(groups) == len(rollout_engines), (
        f"length mismatch: {len(group_names)=} {len(groups)=} {len(rollout_engines)=}"
    )
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine, group_name in zip(rollout_engines, group_names, strict=True)
    ]

    handles = []
    for group in groups:
        for _, param in converted_named_tensors:
            handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
