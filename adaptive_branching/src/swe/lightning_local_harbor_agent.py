"""Opt-in Lightning bridge for audited Full/Local training."""

import json

from adaptive_branching.src.swe.agent import Episode, run_episode, training_config
from adaptive_branching.src.swe.lightning_harbor_agent import LightningSweAgent
from adaptive_branching.src.swe.lightning_replay import SCHEMA, restore_replay, validate_prefix


class LocalLightningSweAgent(LightningSweAgent):
    def __init__(self, *args, replay=False, max_turns=100, **kwargs):
        if type(replay) is not bool or type(max_turns) is not int or max_turns <= 0:
            raise ValueError("replay must be bool and max_turns a positive integer")
        super().__init__(*args, **kwargs)
        self.replay = replay
        self.request_max_turns = max_turns
        self.prefix_calls = 0
        if not replay and max_turns != self.config.max_turns:
            raise ValueError(f"Full Lightning training requires {self.config.max_turns} turns")
        if replay and not 10 <= max_turns < self.config.max_turns + 30:
            raise ValueError("local replay request exceeds the full prefix budget plus H30")

    async def run_loop(self, instruction, query, execute, environment):
        if not callable(query) or not callable(execute) or not callable(getattr(environment, "exec", None)):
            raise TypeError("local loop requires executable environment and model/shell callables")
        if self.replay:
            path = self.logs_dir / "replay.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            state = json.loads(path.read_text())
            if not isinstance(state, dict):
                raise ValueError(f"replay artifact must be an object: {path}")
            if state.get("source_max_turns", 100) != self.config.max_turns:
                raise ValueError("replay source budget differs from the active Full budget")
            validate_prefix(
                state.get("messages"), state.get("n_calls"), problem=instruction, max_turns=self.config.max_turns
            )
            horizon = self.request_max_turns - state["n_calls"]
            if horizon not in (10, 20, 30):
                raise ValueError(f"Local Lightning requires H=10/20/30, got {horizon}")
            prefix = await restore_replay(state, execute)
            self.prefix_calls = state["n_calls"]
            self.episode = Episode(messages=prefix, turns=self.prefix_calls)
            await run_episode(
                instruction,
                query,
                execute,
                config=self.config,
                episode=self.episode,
                resume=True,
                local_horizon=horizon,
            )
        else:
            # Return-code replay needs only the command ledger. Do not launch
            # filesystem readers before model calls on the Full source path.
            await run_episode(instruction, query, execute, config=self.config, episode=self.episode)

    def trajectory(self, metrics):
        result = super().trajectory(metrics)
        result.update(
            replay_schema=SCHEMA,
            command_results=self.episode.command_results,
            prefix_calls=self.prefix_calls,
        )
        if self.config.max_turns != 100:
            result["source_max_turns"] = self.config.max_turns
        return result


class LargeLocalLightningSweAgent(LocalLightningSweAgent):
    """Same H10/H20/H30 continuation contract with a 200-turn/256K source budget."""

    CONFIG = training_config("large")
