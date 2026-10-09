"""Failures for which a remote agent may still own a model session."""


class RemoteTrialUnresolved(RuntimeError):
    """Stop the rollout without deleting the session of an unconfirmed trial."""
