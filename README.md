## Run Pipeline

The runner now supports matrix-style execution across one or more models, scenarios, and orchestrations.
It will start a vLLM server automatically for each model configuration.

### Quick Start (CLI)

```bash
python main.py \
	--models openai/Qwen/Qwen3.5-27B \
	--scenarios coding,database \
	--orchestrations graph,star \
	--task-limit 1
```

Notes:

- `--models`, `--scenarios`, and `--orchestrations` accept comma-separated values or repeated flags.
- Supported scenarios: `coding`, `database`, `research`, `bargaining`, `minecraft`, `werewolf`.
- Supported orchestrations: `graph`, `star`, `chain`, `tree`.
- For multiple models, vLLM ports auto-increment from `--base-port`.

### Pipeline Config (JSON)

Use `--pipeline-config` for per-model vLLM settings.

```json
{
	"models": [
		{
			"model_id": "openai/Qwen/Qwen3.5-27B",
			"vllm": {
				"model": "Qwen/Qwen3.5-27B",
				"port": 8000,
				"max_model_len": 262144,
				"reasoning_parser": "qwen3",
				"enable_auto_tool_choice": true,
				"tool_call_parser": "qwen3_coder",
				"extra_args": []
			}
		}
	],
	"scenarios": ["coding", "database"],
	"orchestrations": ["graph", "star"],
	"task_limit": 1,
	"debug_litellm": false
}
```

Run with:

```bash
python main.py --pipeline-config pipeline.json
```

### Outputs

- Per-run logs are written under `logs/benchmark_<timestamp>/`.
- Aggregate matrix summary is written to `logs/pipeline_summary_<timestamp>.json`.