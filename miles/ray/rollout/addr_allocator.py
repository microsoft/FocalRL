import logging
from dataclasses import dataclass

import ray

logger = logging.getLogger(__name__)


@dataclass
class PortCursors:
    _values: dict[int, int]

    @staticmethod
    def empty() -> "PortCursors":
        return PortCursors(_values={})

    def assign(self, other: "PortCursors"):
        self._values = other._values.copy()

    def next_base_port(self) -> int:
        return max(self._values.values()) if self._values else 15000


# NOTE: May re-implement this in a potentially easier way if needed
def allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    """Allocate on each actor's actual node, including partially occupied nodes."""
    gpus_per_engine = args.rollout_num_gpus_per_engine if num_gpus_per_engine is None else num_gpus_per_engine
    for name, value in (
        ("gpus_per_engine", gpus_per_engine),
        ("num_gpus_per_node", args.num_gpus_per_node),
        ("sglang_dp_size", args.sglang_dp_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if type(rank_offset) is not int or rank_offset < 0:
        raise ValueError(f"Invalid rank_offset: {rank_offset!r}")
    if type(base_port) is not int or not 1 <= base_port <= 65535:
        raise ValueError(f"Invalid base_port: {base_port!r}")
    if max(gpus_per_engine, args.num_gpus_per_node) % min(gpus_per_engine, args.num_gpus_per_node):
        raise ValueError("Engine and node GPU counts must divide one another")
    engines = dict(rollout_engines)
    if len(engines) != len(rollout_engines):
        raise ValueError("Duplicate rollout engine ranks")
    if any(type(rank) is not int or rank < rank_offset for rank in engines):
        raise ValueError(f"Invalid engine ranks {list(engines)} for offset {rank_offset}")
    nodes_per_engine = max(1, gpus_per_engine // args.num_gpus_per_node)
    if nodes_per_engine > 1:
        for rank in engines:
            leader = rank - (rank - rank_offset) % nodes_per_engine
            if not set(range(leader, leader + nodes_per_engine)) <= engines.keys():
                raise ValueError(f"Incomplete multi-node engine at rank {leader}: {sorted(engines)}")

    addr_and_ports = {}
    node_indices = {}
    node_port_cursor = {}
    for rank, engine in sorted(engines.items()):
        host, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"Engine {rank} returned invalid host {host!r}")
        node_index = node_indices.setdefault(host, len(node_indices))

        def get_port(consecutive=1):
            assert type(consecutive) is int and consecutive > 0, consecutive
            start = node_port_cursor.get(node_index, base_port)
            actual_host, port = ray.get(
                engine._get_current_node_ip_and_free_port.remote(start_port=start, consecutive=consecutive)
            )
            if actual_host != host:
                raise RuntimeError(f"Engine {rank} moved from {host} to {actual_host} during port allocation")
            if type(port) is not int or not start <= port <= 65536 - consecutive:
                raise ValueError(f"Engine {rank} on {host} returned invalid port {port!r}, start={start}")
            node_port_cursor[node_index] = port + consecutive
            return port

        entry = dict(host=host, port=get_port(), nccl_port=get_port(), engine_info_bootstrap_port=get_port())
        if worker_type == "prefill":
            entry["disaggregation_bootstrap_port"] = get_port()
        if (rank - rank_offset) % nodes_per_engine == 0:
            entry["dist_init_addr"] = f"{host}:{get_port(30 + args.sglang_dp_size)}"
        else:
            leader = rank - (rank - rank_offset) % nodes_per_engine
            entry["dist_init_addr"] = addr_and_ports[leader]["dist_init_addr"]
        addr_and_ports[rank] = entry
        logger.info(f"Ports for engine {rank}: {entry}")

    assert addr_and_ports.keys() == engines.keys(), "Allocated unexpected engine ranks"
    return addr_and_ports, PortCursors(_values=node_port_cursor)


def allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports
