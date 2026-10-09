# Checkpoint evaluation

The evaluation runners use the same agent implementations as training. Deep Research is scored against reference answers by an LLM judge; SWE is scored by the isolated benchmark test verifier. Run commands from the repository root with Python 3.11 or newer. Use the [training guide](training.md) to install the GPU serving environment and the [SWE guide](swe.md) to prepare Harbor.

## Serve a checkpoint

Set `CHECKPOINT` to your local Hugging Face checkpoint directory, `MODEL` to the model name advertised by the server, and `TP_SIZE` to the number of GPUs assigned to serving. The checkpoint directory must include its tokenizer files. For the Qwen3.5-based FocalRL checkpoints:

```bash
mkdir -p outputs
sglang serve \
  --model-path "$CHECKPOINT" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port 30000 --tp-size "$TP_SIZE" \
  --context-length 262144 \
  --chat-template "$PWD/miles/utils/chat_template_utils/templates/qwen3.5_fixed.jinja" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  > outputs/model.log 2>&1 &
echo $! > outputs/model.pid
```

Startup time depends on checkpoint size, storage bandwidth, and GPU capacity and cannot be estimated reliably. Monitor with `tail -f outputs/model.log`; readiness is a successful `curl -fsS http://127.0.0.1:30000/v1/models` listing `MODEL`. Stop this server with `kill "$(cat outputs/model.pid)"`. Retain the log when diagnosing startup failures; restarting uses the same checkpoint files.

## Deep Research

### Install and prepare data

```bash
python -m pip install -r requirements-eval.txt
```

Installation time depends on the package cache and network. Pip displays progress; exit code zero indicates success. Interrupt with Ctrl-C and rerun to reuse cached downloads.

Prepare labeled BrowseComp or GAIA-text data as JSONL or Parquet. Each task needs a unique identifier, a question, and a reference answer:

```json
{"task_id": "example-1", "question": "Which city is described by the clues?", "answer": "Reference city"}
```

The runner also accepts `problem` / `task_question` for the question, `ground_truth` for the answer, and `id` for the identifier. GAIA columns `Question`, `Final answer`, and `task_id` are supported directly. Select GAIA tasks whose `file_name` is empty to evaluate GAIA-text. Every selected row must include its reference answer. For encrypted BrowseComp exports, decrypt the questions and answers using the benchmark's official preparation procedure before evaluation.

### Configure services

The evaluation configuration is [`eval_tools.yaml`](../adaptive_branching/config/eval_tools.yaml). It uses [Serper](https://serper.dev/) for search, the Jina reader for page fetching, and an OpenAI-compatible model for page summaries. Set these environment variables:

| Variable | Purpose |
| --- | --- |
| `SERPER_API_KEY` | Serper search credential |
| `BROWSER_LLM_URL`, `LLM_API_KEY` | Page summarization endpoint and credential |
| `BROWSER_LLM_MODEL` | Page summarization model; defaults to the model in `eval_tools.yaml` |
| `LLM_JUDGE_URL`, `LLM_JUDGE_KEY` | Answer judge endpoint and credential |
| `AGENT_CHAT_API_KEY` | Policy server credential, when authentication is enabled |

Set the answer judge model and reasoning settings in [`judge.yaml`](../adaptive_branching/config/judge.yaml). `AGENT_TOOLS_CONFIG` and `AGENT_JUDGE_CONFIG` select alternate YAML files. `JINA_BASE_URL` selects another compatible reader service, and `SERPER_ENDPOINT` selects the Serper search endpoint. The evaluation configuration fetches pages through Jina without requiring Microsoft browse-service credentials.

### Run BrowseComp or GAIA-text

Set `DATA` to your prepared data file and choose a separate `OUT` directory for each checkpoint and benchmark:

```bash
export BASE_URL=http://127.0.0.1:30000/v1
export DATA="$PWD/data/browsecomp.jsonl"
export OUT="$PWD/outputs/browsecomp"
mkdir -p outputs
bash adaptive_branching/shells/eval/run_browsecomp_eval.sh \
  > outputs/browsecomp.log 2>&1 &
echo $! > outputs/browsecomp.pid
```

For GAIA-text, use `DATA="$PWD/data/gaia-text.jsonl"` and a distinct output and log path. The same runner evaluates both benchmarks.

The defaults are one sample per task, 200 assistant turns, a 262144-token context window, a 32768-token context reserve, and 22000 output tokens per model call. Sampling uses temperature 1.0 and top-p 1.0. The agent retains the five most recent tool results in its history. Override `CONCURRENCY`, `MAX_TURNS`, `MAX_TOKENS`, `MAX_SEQ_LEN`, `CONTEXT_RESERVE_TOKENS`, or `KEEP_TOOL_RESULTS` through environment variables; additional Python options can be passed to the shell script. Inspect them with `bash adaptive_branching/shells/eval/run_browsecomp_eval.sh --help`.

Runtime depends on dataset size, trajectory lengths, policy throughput, and search and judge service limits and cannot be estimated reliably. Monitor with `tail -f outputs/browsecomp.log`. Completion writes `$OUT/summary.json` and prints `summary -> ...`; for a complete run, `actual_valid_records` equals `expected_valid_records` in every summary entry. Accuracy is the fraction of completed samples judged correct. The per-task JSONL contains answers, interaction histories, agent metrics, and judge verdicts.

Stop the client with `kill "$(cat outputs/browsecomp.pid)"`. Resume using the same settings and output directory by adding `--resume` to the command. Completed attempts are retained; generation or judge failures can be retried. Changed model, data, sampling, history, or service settings are rejected. Keep the partial JSONL for resuming and diagnostics. The default failure policy stops on errors; `--keep-going` records per-task failures and continues.

## Software engineering

### Prepare services and benchmark tasks

Install the client and session-server dependencies:

```bash
python -m pip install -r requirements-swe-eval.txt
```

Installation time depends on the package cache and network. Pip displays progress; exit code zero indicates success. Interrupt with Ctrl-C and rerun to reuse cached downloads.

Prepare the patched Harbor controller, Docker environments, and isolated verifier as described in the [SWE guide](swe.md). Use the patched SWE-bench adapter to prepare Verified tasks:

```bash
python -m swebench_adapter.main --all --task-dir "$HARBOR_TASKS_DIR"
python -m adaptive_branching.tools.swe.prepare_swebench_prompts \
  --tasks-dir "$HARBOR_TASKS_DIR" --output data/swe-verified.jsonl
```

Task preparation time depends on dataset and container downloads and cannot be estimated reliably. Run it in the foreground to view progress. The prompt command prints `WROTE_SWEBENCH_PROMPTS` on success. Verified requires all 500 unique tasks; Lite requires all 300. On failure, stop with Ctrl-C and retain existing task directories for inspection; rerun prompt preparation to a new output file. Container preparation runs separately from policy inference.

For Lite, use prepared Lite task directories with the `SWE-bench_Lite`, `SWE-bench/SWE-bench_Lite`, and `test` dataset markers, and pass `--dataset lite` to the prompt preparation command. The runner checks the official verifier version: 4.0.3 for Verified and 5.0.2 for Lite.

Start a session server in the SGLang serving environment, in front of the running model server:

```bash
python -m adaptive_branching.tools.swe.start_eval_session \
  --checkpoint "$CHECKPOINT" --backend-url http://127.0.0.1:30000 \
  --host 127.0.0.1 --port 30001 \
  > outputs/session.log 2>&1 &
echo $! > outputs/session.pid
```

Session startup loads the tokenizer; duration depends on storage and cannot be estimated reliably. Monitor with `tail -f outputs/session.log`; readiness is `curl -fsS http://127.0.0.1:30001/health` returning a `session_server_instance_id`. Stop with `kill "$(cat outputs/session.pid)"` and restart with the same checkpoint after inspecting the log. Set `--host` to an address reachable from the Harbor controller when services run on separate machines.

### Run Verified or Lite

Set `HARBOR_SERVER_URL` and `SESSION_SERVER_URL` to the services reachable from both the client and controller:

```bash
python -m adaptive_branching.tools.swe.run_lightning_verified \
  --benchmark verified --prompts data/swe-verified.jsonl \
  --output outputs/swe-verified --model "$MODEL" --budget large \
  --harbor "$HARBOR_SERVER_URL" --session "$SESSION_SERVER_URL" \
  --concurrency 32 \
  > outputs/swe-verified.log 2>&1 &
echo $! > outputs/swe-verified.pid
```

Use `--benchmark lite` with the Lite prompt file and a separate output directory to evaluate Lite. The original evaluation runner's `large` setting allows 250 assistant turns, 32768 output tokens per call, and a 262144-token context window, at temperature 1.0. The `rl` setting uses 100 turns, 12288 output tokens per call, and an 81920-token context window. Choose concurrency according to the capacity of the policy server and task and verifier containers.

Each task receives one attempt. Results preserve the official verifier report. `score` treats a final generation truncated by its output limit as a failure; `verifier_accuracy` reports the official verifier's raw resolved rate. Both use the full benchmark denominator. An evidenced official test timeout is counted as an unresolved task; infrastructure errors remain recorded separately.

Runtime depends on task execution and verifier tests, container capacity, and policy throughput and cannot be estimated reliably. Monitor with `tail -f outputs/swe-verified.log`; each completed task prints `PROGRESS`. Completion writes `outputs/swe-verified/DONE.json` and prints `EVALUATION_DONE`, with `completed` equal to `total` and `missing` equal to zero. Inspect the error count alongside the score.

Stop the runner with `kill "$(cat outputs/swe-verified.pid)"`. Keep `config.json`, `started.jsonl`, `results.jsonl`, and the Harbor trial directories. Restart the same command to resume; durable start records prevent a second trajectory for an already-dispatched task. An interrupted attempt without a result is recorded as an infrastructure failure. The runner stops on errors by default. `--keep-going` explicitly enables continuing with recorded failures; after consecutive errors it writes `PAUSE.json`, pauses new dispatches, and drains active work. Diagnose the services before removing `PAUSE.json` to resume dispatch.
