"""SWE entry point for the shared DeepResearch full/local collector."""

from adaptive_branching.src.deep_research.fully_async_value_cliff_collector import generate_rollout_fully_async

__all__ = ["generate_rollout_fully_async"]
