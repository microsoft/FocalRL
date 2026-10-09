# Software engineering

The SWE agent follows a ReAct interaction loop and executes tools in Harbor-managed containers. Harbor provides an isolated task environment and a separate test verifier, while Miles records generated policy tokens for training. Local continuations restore the conversation and replay preceding commands in a fresh container, checking command exit codes for consistency. Full trajectories are scored by the test verifier; local continuations are scored against repair rubrics.

## Harbor source

Apply the integration patch to Harbor commit `53a6e92a2793633cc23d79992b52457ef85459de`:

```bash
git clone --depth 1 https://github.com/harbor-framework/harbor.git /tmp/harbor-base
git -C /tmp/harbor-base fetch --depth 1 origin 53a6e92a2793633cc23d79992b52457ef85459de
git -C /tmp/harbor-base checkout --detach 53a6e92a2793633cc23d79992b52457ef85459de
git -C /tmp/harbor-base apply --check "$PWD/adaptive_branching/patches/harbor-miles-v0.20.0.patch"
git -C /tmp/harbor-base apply "$PWD/adaptive_branching/patches/harbor-miles-v0.20.0.patch"
python -m adaptive_branching.tools.swe.prepare_lightning_local_harbor \
  --source /tmp/harbor-base --destination /tmp/harbor-focalrl
```

Use a new destination directory for the prepared Harbor controller. The preparation enables environment replay and connects the FocalRL agent to Harbor.

Install the prepared Harbor source and its SWE adapter in a separate Python 3.12 Linux environment, following the Harbor dependencies. Both Harbor and FocalRL must be importable in the controller environment. Prepare Docker and the task container images.

## Tasks and data

For R2E data, use `adaptive_branching.tools.swe.prepare_r2e_tasks` to build task directories, test verifier files, and prompt JSONL from your dataset and container images. Inspect `--help` for dataset and image arguments. Then use `adaptive_branching.tools.swe.prepare_lightning_tasks` to create tasks with network access disabled in the agent environment, preserving the verifier's separate configuration. Choose a new destination directory and set it as the Harbor task root.

Each training row has this structure:

```json
{"prompt": "Issue description", "metadata": {"instance_id": "task-directory-name", "ab_swe_golden_patch": "reference source patch"}}
```

Replace the example values with your issue, prepared task identifier, and reference patch. The task identifier must resolve to a directory under the Harbor task root. The training controller uses the reference patch to localize errors; the policy receives the issue and interaction history.

Use `adaptive_branching.tools.swe.prepare_swebench_prompts` to prepare SWE-bench prompts.

## Start the Harbor controller

From the prepared Harbor directory, start `miles_agent_server.py` with `--host`, `--port`, `--max-concurrent`, `--agent-timeout`, `--trials-dir`, and `--dashboard-port` appropriate to your allocated machine. Configure:

- `HARBOR_TASKS_DIR`: prepared task root.
- `HARBOR_DELETE_CONTAINERS=true`: delete containers after each trial.
- `HARBOR_TIMEOUT_MULTIPLIER=1.0`: use the configured task timeouts.
- `PYTHONPATH`: both the prepared Harbor source/`src` and the FocalRL checkout.

Use the upstream Docker socket configuration appropriate to your machine. Keep containers isolated from host secrets and keep grading in the separate verifier environment. Do not expose an unauthenticated controller to the public internet.

Keep the controller running during training. Startup duration cannot be estimated reliably because it depends on container images and Docker initialization. Redirect logs to a dedicated file, save the process ID, and use `tail -f` to monitor startup. Check the controller's health endpoint and run a task to confirm readiness. Stop the controller by its process ID and retain the trial directory for diagnostics. Before restarting, inspect outstanding trials and containers.

## Training-side environment

Set `HARBOR_SERVER_URL`, `MILES_ROUTER_EXTERNAL_HOST`, `LLM_JUDGE_URL`, and `LLM_JUDGE_KEY`. If authentication is enabled, supply `HARBOR_ADMIN_SECRET`. `MILES_ROUTER_EXTERNAL_PORT` can override the externally reachable router port. The controller must be able to reach Miles' session endpoint, and training workers must reach the controller.

Each training batch contains 16 groups of full trajectories and 32 groups of local continuations, with eight samples per group. Full rollouts use a 200-turn interaction limit and a 256K-token context window; local continuations allow up to 20 turns. Local rewards evaluate whether the continuation avoids the identified error and makes the specified correction, assigning scores of 0, 0.5, or 1 as described in the [training guide](training.md#training-configuration).

Before a full run, run a small task and inspect its returned messages, verifier result, replay prefix and local continuation.
