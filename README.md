# Multi-Agent Evaluation Pipeline

Runs the [MARBLE MultiAgentBench](https://github.com/ulab-uiuc/MARBLE) benchmark
against a local language model, using [MASEval](https://github.com/…/maseval) as
the evaluation harness.

## Quick Start – Tiny Models (CPU)

```bash
# Install dependencies
pip install "maseval[multiagentbench]>=0.4.0" litellm fastapi uvicorn transformers accelerate

# Download MARBLE data
python -c "from maseval.benchmark.multiagentbench import ensure_marble_exists; ensure_marble_exists()"

# Run the pipeline with a tiny Qwen model on CPU
python bench_utils/pipeline_runner.py
```

The pipeline automatically:
1. Downloads `Qwen/Qwen2.5-0.5B-Instruct` (≈1 GB, first run only).
2. Starts an OpenAI-compatible HTTP server on `http://127.0.0.1:8000`.
3. Runs the benchmark and writes detailed logs to `logs/`.

Override the model with the `BENCH_TINY_MODEL` environment variable:

```bash
BENCH_TINY_MODEL=Qwen/Qwen2.5-1.5B-Instruct python bench_utils/pipeline_runner.py
```

## Full-Size Models (GPU / vLLM)

For larger models served by vLLM, use `main.py` and start vLLM separately:

```bash
# Start vLLM server (requires GPU)
vllm serve Qwen/Qwen3.5-27B --port 8000 --max-model-len 262144 \
     --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

# Run the benchmark
python main.py
```

## Known Fixes Applied

| Bug | Fix |
|-----|-----|
| `'>=' not supported between instances of 'int' and 'str'` | `BaseEnvironment.max_iterations` is now coerced to `int` (the JSONL stores it as `""`) |
| `gpt-3.5-turbo` model-not-found errors | All model strings are rewritten to point to the local endpoint |
| Empty / reasoning-only LLM responses | Content is normalised before MARBLE parses it |
