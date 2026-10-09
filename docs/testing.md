# Testing

Run the CPU regression suite from the repository root:

```bash
python -m pytest -q
```

## Test coverage

- Full/local rollout collection, error localization, repair rubrics, and local rewards.
- Agent interaction limits, environment replay, termination, and cleanup.
- Group credit assignment and training data conversion.
- Training configurations, argument/data validation, and launch dispatch with a mocked Ray connection.

External model and environment services are mocked in the CPU suite. The Harbor process-group timeout test runs on Linux and is skipped on other platforms.

CPU test dependencies are listed in [`requirements-cpu.txt`](../requirements-cpu.txt). See the [training guide](training.md) for the GPU environment and the [SWE guide](swe.md) for Harbor setup.
