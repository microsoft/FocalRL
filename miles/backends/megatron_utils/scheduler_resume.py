"""Preserve loaded scheduler state when continuing a Megatron checkpoint."""


def advance_unrestored_scheduler(scheduler, *, iteration, global_batch_size, no_load_optim):
    if type(iteration) is not int or iteration < 0:
        raise ValueError(f"invalid checkpoint iteration: {iteration!r}")
    if type(global_batch_size) is not int or global_batch_size <= 0:
        raise ValueError(f"invalid global batch size: {global_batch_size!r}")
    # Megatron's store_true argument defaults to None (not set).
    if no_load_optim is not None and type(no_load_optim) is not bool:
        raise ValueError("no_load_optim must be boolean or None")
    if scheduler is None:
        return None
    steps = scheduler.num_steps
    if type(steps) is not int or steps < 0:
        raise ValueError(f"invalid scheduler num_steps: {steps!r}")
    # Megatron loads the scheduler together with the optimizer. Advancing it
    # again would double-count the checkpoint's completed training history.
    if no_load_optim:
        if steps != 0:
            raise ValueError(f"unrestored scheduler must start at zero, got {steps}")
        if iteration:
            scheduler.step(increment=iteration * global_batch_size)
    return scheduler.num_steps
