from __future__ import annotations

import json
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any


def _get_eval_passed(eval_data: Any) -> Any:
    if isinstance(eval_data, list) and eval_data:
        first_eval = eval_data[0]
        if isinstance(first_eval, dict):
            return first_eval.get("passed")
    if isinstance(eval_data, dict):
        return eval_data.get("passed")
    return None


def summarize_results(results):
    evaluation_summary = []
    for result in results:
        eval_data = result.get("eval")
        evaluation_summary.append(
            {
                "task_id": result.get("task_id"),
                "status": result.get("status"),
                "passed": _get_eval_passed(eval_data),
                "eval": eval_data,
            }
        )
    return evaluation_summary


def write_evaluation_log(
    eval_log_path: str | Path,
    run_timestamp: str,
    run_dir: str | Path,
    evaluation_summary,
):
    Path(eval_log_path).write_text(
        json.dumps(
            {
                "run_timestamp": run_timestamp,
                "run_directory": str(run_dir),
                "results": evaluation_summary,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def run_benchmark(
    *,
    local_api_base: str,
    local_model_id: str,
    verbose_log_path: str | Path,
    scenario: str = "coding",
    coordinate_mode: str = "graph",
    task_limit: int = 1,
):
    from maseval.benchmark.multiagentbench import (
        MarbleMultiAgentBenchBenchmark,
        configure_model_ids,
        ensure_marble_exists,
        load_tasks,
    )
    from maseval.interface.inference.openai import OpenAIModelAdapter
    from openai import OpenAI

    class MyMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
        def __init__(self, api_base: str):
            self.local_api_base = api_base
            self.local_client = OpenAI(base_url=api_base, api_key="local-token")
            super().__init__()

        def get_model_adapter(self, model_id, **kwargs):
            adapter = OpenAIModelAdapter(
                client=self.local_client,
                model_id=model_id,
            )
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Ensure MARBLE sources are available before loading tasks.
    ensure_marble_exists()
    tasks = load_tasks(scenario, limit=task_limit)
    for task in tasks:
        task.environment_data["coordinate_mode"] = coordinate_mode
    configure_model_ids(tasks, agent_model_id=local_model_id)

    progress_logger = logging.getLogger("progress")
    progress_logger.info("Task setup complete")
    progress_logger.info("Model configured: %s", local_model_id)
    progress_logger.info("Scenario configured: %s", scenario)
    progress_logger.info("Orchestration configured: %s", coordinate_mode)

    benchmark = MyMarbleBenchmark(local_api_base)
    agent_data = {}
    progress_logger.info("Orchestration started")
    with open(verbose_log_path, "a", encoding="utf-8") as verbose_sink:
        with redirect_stdout(verbose_sink), redirect_stderr(verbose_sink):
            results = benchmark.run(tasks, agent_data=agent_data)
    progress_logger.info("Orchestration finished")
    return results
