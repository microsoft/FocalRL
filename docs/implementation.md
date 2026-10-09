# Implementation

FocalRL's task implementations are in `adaptive_branching/src/deep_research` and `adaptive_branching/src/swe`. The included Miles framework handles rollout generation and policy optimization, including asynchronous execution.

## Components

- Deep Research and SWE task runtimes, consequential error localization, and local repair rewards.
- Search/browse adapters, SWE task preparation utilities, and Harbor integration patches.
- Qwen3.5 model configurations and training settings in `scripts/launch.py`.
- CPU regression tests under `adaptive_branching/tests` and `tests`.

## Configuration

Service endpoints and credentials are configured through environment variables.

Local repair rewards evaluate whether the continuation avoids the identified error and makes the specified correction. See the [training guide](training.md), [Deep Research guide](deep_research.md), and [SWE guide](swe.md) for configuration and launch instructions.
