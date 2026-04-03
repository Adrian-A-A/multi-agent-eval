from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from bench_utils import build_pipeline_config_from_args, run_pipeline


def _split_csv(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None

    items: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                items.append(item)
    return items or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MARBLE benchmark pipeline.")
    parser.add_argument("--pipeline-config", type=str, default=None, help="Path to a pipeline JSON config.")
    parser.add_argument("--models", nargs="*", default=None, help="Model ids, comma-separated or repeated.")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Scenarios to run, comma-separated or repeated.")
    parser.add_argument(
        "--orchestrations",
        nargs="*",
        default=None,
        help="Orchestration modes to run, comma-separated or repeated.",
    )
    parser.add_argument("--task-limit", type=int, default=1, help="Maximum tasks per scenario.")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port for vLLM instances.")
    parser.add_argument("--max-model-len", type=int, default=262144, help="Maximum model context length.")
    parser.add_argument("--reasoning-parser", type=str, default="qwen3", help="vLLM reasoning parser.")
    parser.add_argument(
        "--disable-auto-tool-choice",
        action="store_true",
        help="Disable vLLM auto tool choice.",
    )
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default="qwen3_coder",
        help="vLLM tool call parser.",
    )
    parser.add_argument("--debug-litellm", action="store_true", help="Enable LiteLLM debug logging.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    args.models = _split_csv(args.models)
    args.scenarios = _split_csv(args.scenarios)
    args.orchestrations = _split_csv(args.orchestrations)

    config = build_pipeline_config_from_args(args)
    result = run_pipeline(config)

    summary_path = Path(result["summary_path"])
    print(json.dumps({"summary_path": str(summary_path), "runs": result["runs"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
