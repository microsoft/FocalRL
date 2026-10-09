"""Agent Lightning protocol adapted for thinking mode; see manifest.json and LICENSE."""

import re

_HIDDEN_GIT_DIR = "/opt/agl_tmp"

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


_GIT_INVOKE_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:(?:if|elif|while|until|then|do|!)\s+)*"
    r"(?:\w+=\S+\s+)*(?:[\w./-]*/)?git(?:-[a-z]+)?(?=\s|$|;|&|\|)",
    re.I,
)


_GIT_ACCESS_RE = re.compile(r'--git-dir|--work-tree|(?:^|[\s=:"\'/])\.git(?:/|\b)', re.I)


_NET_FETCH_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?"
    r"(?:curl|wget|httpie|http|https|aria2c|scp|sftp|rsync|nc|ncat|netcat|telnet)"
    r"(?=\s|$|;|&|\|)",
    re.I,
)


_PKG_INSTALL_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:"
    r"(?:[\w./-]*/)?(?:pip|pip3|conda|mamba|easy_install|uv)\b[^\n;&|]*?\binstall\b"
    r"|(?:[\w./-]*/)?python[0-9.]*\s+-m\s+pip\b[^\n;&|]*?\binstall\b)",
    re.I,
)


_PY_NET_RE = re.compile(
    r"urllib\.request|\burlopen\b|\brequests\.(?:get|post|put|head|Session)\b|"
    r"\bhttpx\.|\bsocket\.(?:socket|create_connection)\b|\burllib3\b",
    re.I,
)


_TEST_TAMPER_RE = re.compile(
    r'(?:>>?|\btee\b(?:\s+-a)?\s+)\s*[\'"]?[^\s\'"|;&<>]*'
    r"(?:conftest\.py|pytest\.ini|tox\.ini|sitecustomize\.py|usercustomize\.py|"
    r'setup\.cfg|pyproject\.toml|\.pth)(?=[\'"\s;&|]|$)',
    re.I,
)


_CMD_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


_OVERLONG_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can use head, tail or sed to view a smaller "
    "number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, use a more "
    "selective search pattern.\n"
    "If you really need the full output, redirect it to a file and search within it."
)


def is_submission(output: str) -> bool:
    """True when command output begins with the submit marker."""
    stripped = output.lstrip()
    if not stripped:
        return False
    return stripped.split("\n", 1)[0].strip() == SUBMIT_MARKER


def render_observation(returncode: int, output: str, obs_cap: int) -> str:
    """Format a command result the way mini-swe-agent does.

    Output shorter than ``obs_cap`` is shown verbatim; longer output is elided to
    a head+tail window with a warning so one noisy command cannot blow up context.
    """
    if len(output) < obs_cap:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>"
    half = obs_cap // 2
    elided = len(output) - obs_cap
    return (
        f"<returncode>{returncode}</returncode>\n"
        f"<warning>\n{_OVERLONG_WARNING}\n</warning>\n"
        f"<output_head>\n{output[:half]}\n</output_head>\n"
        f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-half:]}\n</output_tail>"
    )


def _forbidden_action(action: str) -> str | None:
    """Return a rejection reason if the action cheats instead of solving the bug.

    Blocks four reward-hacking routes so the agent must fix the bug from the
    local /testbed source only:
      - git (relocate_git() moved the repo; history holds the fix + deleted tests)
      - network fetches (curl/wget/python-http) that download the upstream code
      - package installs (pip/conda) that pull the target package's correct source
      - writing test-harness files (conftest/pytest.ini/sitecustomize) to force PASS
    Returns None for allowed actions. Network/install/tamper blocks are a code
    backstop; the authoritative fix is a default-deny egress NetworkPolicy.
    """
    if _GIT_INVOKE_RE.search(action):
        return (
            "git is disabled in this environment; do not use it for any "
            "purpose. Inspect and edit the source files under /testbed "
            "directly (cat, grep, sed, python) to fix the bug."
        )
    if _GIT_ACCESS_RE.search(action) or _HIDDEN_GIT_DIR in action:
        return (
            "Accessing the git metadata directory is not allowed. Work only "
            "with the source files under /testbed; do not read .git."
        )
    if _NET_FETCH_RE.search(action) or _PY_NET_RE.search(action):
        return (
            "Network access is disabled. Do not fetch code from the internet "
            "(curl, wget, urllib, requests, etc.); all dependencies are "
            "already installed. Solve the bug using only the source files "
            "already present under /testbed."
        )
    if _PKG_INSTALL_RE.search(action):
        return (
            "Installing packages is not allowed. Everything needed to run the "
            "code and its tests is already installed. Fix the bug by editing "
            "the source under /testbed; do not install anything."
        )
    if _TEST_TAMPER_RE.search(action):
        return (
            "Modifying test-harness or config files (conftest.py, pytest.ini, "
            "tox.ini, setup.cfg, pyproject.toml, sitecustomize.py, .pth) is "
            "not allowed. Fix the bug in the source under /testbed instead."
        )
    return None


def length_penalized_reward(
    reward: float, n_turns: int, max_turns: int, *, t0: int, lam: float, is_train: bool
) -> float:
    """Apply the long-turn penalty (plan A) to a SOLVED *training* trajectory's reward.

    The penalty applies **only** when ``is_train`` is True and ``reward >= 1.0``
    (solved). This keeps VALIDATION reward the true, unshaped metric (validation
    drives checkpoint selection, so it must never be reshaped), and leaves
    unsolved rollouts full exploration room. The penalty is a linear soft ramp
    over the turn budget::

        reward = 1.0 - lam * clip((n_turns - t0) / (max_turns - t0), 0, 1)

    A train solve within ``t0`` turns keeps reward ``1.0``; one that runs to the
    cap gets ``1 - lam``. Validation rewards and non-solved rewards are returned
    unchanged.
    """
    if not is_train or reward < 1.0 or n_turns <= t0:
        return reward
    span = max(1, max_turns - t0)
    frac = min((n_turns - t0) / span, 1.0)
    return 1.0 - lam * frac


def prompt_length_penalty(
    reward: float,
    max_prompt_tokens: int,
    *,
    soft_start: int,
    hard_cap: int,
    max_pen: float,
    is_train: bool,
    solved: bool,
) -> float:
    """Penalizes context bloat: the longest single-turn prompt of a rollout is its
    true upper bound on context pressure (each turn's prompt embeds all prior
    history). Gated identically to the turn penalty — only ``is_train`` and
    ``solved`` — so VALIDATION reward stays the unshaped metric and unsolved
    rollouts are untouched. ``solved`` is the *raw* solved status (not the running
    reward), so this composes with the turn penalty even after it lowered the
    reward below 1.0. Soft linear ramp over the context band::

        penalty = max_pen * clip((max_prompt_tokens - soft_start) /
                                 (hard_cap - soft_start), 0, 1)
        reward -= penalty

    A solve whose longest prompt is <= ``soft_start`` keeps its reward; at or
    above ``hard_cap`` it loses the full ``max_pen``. Stacks with (subtracts on
    top of) the turn penalty.
    """
    if not is_train or not solved or max_prompt_tokens <= soft_start:
        return reward
    if max_prompt_tokens >= hard_cap:
        return reward - max_pen
    span = max(1, hard_cap - soft_start)
    frac = min((max_prompt_tokens - soft_start) / span, 1.0)
    return reward - max_pen * frac
