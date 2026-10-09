"""Verified adapters using the RL loop, with explicit small/large budgets."""

from adaptive_branching.src.swe.agent import AgentConfig
from adaptive_branching.src.swe.lightning_harbor_agent import LightningSweAgent

# Only runs in a disposable task sandbox, before the model sees the repository.
# Preserve the image's exact working tree as the patch baseline, without exposing
# upstream history. The existing isolated official verifier consumes that diff.
PREPARE_BASELINE = """set -euo pipefail
cd /testbed
test -d .git
git rev-parse --verify refs/harbor/image-baseline^{commit} >/dev/null
git diff --quiet
git diff --cached --quiet
test -z "$(git ls-files --others --exclude-standard)"
command -v bash
command -v timeout
test ! -e /tmp/verified-baseline-files
git ls-files -z > /tmp/verified-baseline-files
test -s /tmp/verified-baseline-files
rm -rf -- .git
git init -q
git config core.fileMode false
# No background Git writer may outlive setup: run() relocates .git immediately.
git config gc.auto 0
git config gc.autoDetach false
git config maintenance.auto false
test "$(git config --get gc.auto)" = 0
test "$(git config --get maintenance.auto)" = false
xargs -0 git add -f -- < /tmp/verified-baseline-files
rm /tmp/verified-baseline-files
GIT_AUTHOR_DATE=2000-01-01T00:00:00Z GIT_COMMITTER_DATE=2000-01-01T00:00:00Z \
git -c user.name=Harbor -c user.email=harbor@invalid -c commit.gpgsign=false \
    -c core.hooksPath=/dev/null commit -q --allow-empty --no-verify -m 'Evaluation baseline'
git update-ref refs/harbor/image-baseline HEAD
test "$(git rev-list --all --count)" -eq 1
test -z "$(git remote)"
"""


class VerifiedSmallAgent(LightningSweAgent):
    async def setup(self, environment):
        if not callable(getattr(environment, "exec", None)):
            raise TypeError("Verified setup requires an executable sandbox")
        result = await environment.exec(PREPARE_BASELINE, cwd="/testbed", timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(f"Verified baseline preparation failed: {result.stdout}\n{result.stderr}")


class VerifiedLargeAgent(VerifiedSmallAgent):
    CONFIG = AgentConfig(max_turns=250, max_tokens=32768, model_context=262144)
