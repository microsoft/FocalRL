"""Preserve complete SWE trajectories within the agent's model context."""

from copy import copy
from dataclasses import replace

from adaptive_branching.src.swe.agent import training_config
from adaptive_branching.src.swe.behavior_guard import audit_generated_format


def constrain_sample(sample, *, evaluation: bool):
    if type(evaluation) is not bool or not isinstance(sample.metadata, dict):
        raise ValueError("evaluation must be bool and metadata an object")
    if type(sample.response_length) is not int or not 0 <= sample.response_length <= len(sample.tokens):
        raise ValueError("response_length outside token sequence")
    model_context = training_config().model_context
    if len(sample.tokens) > model_context:
        raise ValueError(
            f"sample index={getattr(sample, 'index', None)} has {len(sample.tokens)} tokens "
            f"exceeding model_context={model_context}; refusing to truncate a rewarded trajectory"
        )
    for name in ("loss_mask", "rollout_log_probs"):
        values = getattr(sample, name, None)
        if values is not None and len(values) != sample.response_length:
            raise ValueError(f"{name} length does not match response_length")
    sample.metadata["swe_lightning_evaluation"] = evaluation
    if sample.response_length == 0 or (sample.loss_mask is not None and not any(sample.loss_mask)):
        sample.remove_sample = True
        sample.metadata["agent_excluded_from_training"] = True
    return sample


async def generate(input):
    from miles.rollout.generate_hub.agentic_tool_call import generate as trace_generate

    if input.args.generate_multi_samples:
        raise ValueError("Lightning preset requires merged trajectories")
    model_context = training_config().model_context
    if type(input.args.max_seq_len) is not int or input.args.max_seq_len != model_context:
        raise ValueError(f"training max_seq_len must equal model_context={model_context}")
    # The generic tracer truncates before merging. Disable that operation only
    # for this call; shared rollout state/args must remain unchanged.
    trace_state = copy(input.state)
    trace_state.args = copy(input.args)
    trace_state.args.max_seq_len = None
    output = await trace_generate(replace(input, state=trace_state))
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    if not samples:
        raise ValueError("tracer returned an empty sample list")
    for sample in samples:
        constrain_sample(sample, evaluation=input.evaluation)
        if not sample.remove_sample:
            audit_generated_format(sample, input.state.tokenizer)
    return output


def _add_arguments(parser):
    from miles.rollout.generate_hub.agentic_tool_call import _add_arguments as add_trace_arguments

    if parser is None:
        raise ValueError("parser required")
    add_trace_arguments(parser)


generate.add_arguments = _add_arguments
