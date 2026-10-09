# Deep Research

The agent uses `search` and `fetch_url` to gather information. Search results and page summaries become observations for subsequent actions. Reference answers and repair rubrics are used by the training controller for error localization and reward evaluation, while the policy receives the question and interaction history.

## Data

Each JSONL row has a nonempty `question` and `answer`:

```json
{"question": "Which city is described by the supplied clues?", "answer": "Reference city"}
```

Prepare an SFT checkpoint for Deep Research training.

## Tool and judge configuration

In our experiments, we used [WebIQ](https://webiq.microsoft.ai/) for web search during training and [Serper](https://serper.dev/) during evaluation.

Configure tool adapters in [`microsoft_gaia_tools.yaml`](../adaptive_branching/config/microsoft_gaia_tools.yaml) and the judge in [`judge.yaml`](../adaptive_branching/config/judge.yaml). Supply service addresses and credentials through the environment variables below.

Set the following environment variables:

| Environment variable | Purpose |
| --- | --- |
| `LLM_JUDGE_URL`, `LLM_JUDGE_KEY` | OpenAI-compatible judge endpoint and credential |
| `SEARCH_ENDPOINT`, `MS_API_KEYS` | Search/grounding service endpoint and key(s) |
| `BROWSE_ENDPOINT` | Browse/grounding fallback endpoint |
| `BROWSER_LLM_URL`, `LLM_API_KEY` | Browser summarization model endpoint and key |
| `AGENT_SEARCH_PROVIDER` | Search backend: `microsoft`, `serper`, `serpent`, or `prismcrawl` |
| `SERPER_API_KEY`, `SERPER_ENDPOINT` | Serper credential and search API endpoint |

Specify the judge model in `judge.yaml` and the page summarization model in `microsoft_gaia_tools.yaml`. Use model identifiers supported by your endpoints and preserve the structured JSON output required for error localization and reward evaluation.

The default tools use the request and response formats defined in `MicrosoftSearchTool` and `MicrosoftBrowseTool`. Page fetching first uses the Jina reader; set `JINA_BASE_URL` to use another compatible reader endpoint. Search adapters for Serper, Serpent, and PrismCrawl are also available, each with its own configuration and credentials.

To use Serper for evaluation, set `AGENT_SEARCH_PROVIDER=serper`, provide `SERPER_API_KEY`, and set `SERPER_ENDPOINT` to your Serper search API endpoint.

Set the endpoint variables to your running services before launch. Offline tests use loopback endpoints, dummy credentials, and mocked external calls. Caches are written under the Git-ignored `.cache/` directory.

## Training procedure

Sample full trajectories and evaluate their final answers. Within groups containing both successful and failed trajectories, use a successful trajectory and the reference answer to identify a consequential error in a failed trajectory and construct a repair rubric.

Resume from the conversation immediately before the selected error and sample continuations of at most five assistant turns. Score each continuation against the corresponding repair rubric. Full trajectories and local continuations form separate reward groups and jointly contribute to the policy update.
