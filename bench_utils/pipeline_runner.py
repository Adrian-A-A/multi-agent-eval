from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import shutil
import subprocess
import time
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

from .benchmark_runner import run_benchmark, summarize_results, write_evaluation_log
from .litellm_guard import configure_litellm_runtime, install_litellm_completion_guard
from .logging_setup import configure_run_logging
from .marble_patches import install_marble_safety_patches


DEFAULT_LOCAL_MODEL_ID = "openai/Qwen/Qwen3.5-0.8B"
VALID_ORCHESTRATIONS = {"graph", "star", "chain", "tree"}
VALID_SCENARIOS = {"coding", "database", "research", "bargaining", "minecraft", "werewolf"}


@dataclass
class VllmConfig:
    model: str
    port: int = 8000
    max_model_len: int | None = 32768
    reasoning_parser: str | None = "qwen3"
    enable_auto_tool_choice: bool = True
    tool_call_parser: str | None = "qwen3_coder"
    gpu_memory_utilization: float = 0.95
    startup_timeout_seconds: int = 1800
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    model_id: str
    vllm: VllmConfig

    @property
    def action_model_name(self) -> str:
        if self.model_id.startswith("openai/"):
            return self.model_id.replace("openai/", "", 1)
        return self.model_id

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.vllm.port}/v1"


@dataclass
class PipelineConfig:
    models: list[ModelConfig]
    scenarios: list[str]
    orchestrations: list[str]
    task_limit: int = 1
    debug_litellm: bool = False


class VllmServer:
    def __init__(self, config: VllmConfig):
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.log_path: Path | None = None
        self._log_file_handle = None

    def _read_log_tail(self, max_lines: int = 80) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        if not lines:
            return ""
        return "\n".join(lines[-max_lines:])

    def start(self, log_dir: str | Path) -> None:
        vllm_executable = shutil.which("vllm")
        if not vllm_executable:
            virtualenv_executable = Path(sys.executable).resolve().with_name("vllm")
            if virtualenv_executable.exists():
                vllm_executable = str(virtualenv_executable)

        if vllm_executable:
            command = [vllm_executable, "serve", self.config.model, "--port", str(self.config.port)]
        else:
            command = [sys.executable, "-m", "vllm", "serve", self.config.model, "--port", str(self.config.port)]
        if self.config.max_model_len is not None:
            command.extend(["--max-model-len", str(self.config.max_model_len)])
        if self.config.reasoning_parser:
            command.extend(["--reasoning-parser", self.config.reasoning_parser])
        if self.config.enable_auto_tool_choice:
            command.append("--enable-auto-tool-choice")
        if self.config.tool_call_parser:
            command.extend(["--tool-call-parser", self.config.tool_call_parser])
        command.extend(["--gpu-memory-utilization", str(self.config.gpu_memory_utilization)])

        effective_extra_args = list(self.config.extra_args)
        if shutil.which("nvcc") is None:
            progress_logger = logging.getLogger("progress")
            if "--enforce-eager" not in effective_extra_args:
                progress_logger.warning(
                    "nvcc was not found; adding --enforce-eager to reduce JIT kernel usage."
                )
                effective_extra_args.append("--enforce-eager")
        command.extend(effective_extra_args)

        logs_dir = Path(log_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_path = logs_dir / f"vllm_{self.config.port}_{timestamp}.log"
        self._log_file_handle = self.log_path.open("a", encoding="utf-8")

        self.process = subprocess.Popen(
            command,
            stdout=self._log_file_handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )

    def wait_until_ready(self, timeout_seconds: int = 180) -> None:
        url = f"http://127.0.0.1:{self.config.port}/v1/models"
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                error_msg = "vLLM process exited before becoming ready.\n"
                if self.log_path:
                    error_msg += f"vLLM log file: {self.log_path}\n"
                tail = self._read_log_tail()
                if tail:
                    error_msg += f"Last vLLM log lines:\n{tail}"
                raise RuntimeError(error_msg)
            try:
                with request.urlopen(url, timeout=2):
                    return
            except (error.URLError, TimeoutError):
                time.sleep(1)
        message = f"Timed out waiting for vLLM at {url}.\n"
        if self.log_path:
            message += f"vLLM log file: {self.log_path}\n"
        tail = self._read_log_tail()
        if tail:
            message += f"Last vLLM log lines:\n{tail}"
        raise TimeoutError(message)

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        self.process = None
        if self._log_file_handle:
            self._log_file_handle.close()
            self._log_file_handle = None


def _default_vllm_model_from_model_id(model_id: str) -> str:
    if model_id.startswith("openai/"):
        return model_id.replace("openai/", "", 1)
    return model_id


def _validate_pipeline_config(config: PipelineConfig) -> None:
    if not config.models:
        raise ValueError("At least one model must be configured.")

    invalid_scenarios = [scenario for scenario in config.scenarios if scenario not in VALID_SCENARIOS]
    if invalid_scenarios:
        raise ValueError(f"Invalid scenario(s): {invalid_scenarios}. Valid: {sorted(VALID_SCENARIOS)}")

    invalid_orchestrations = [mode for mode in config.orchestrations if mode not in VALID_ORCHESTRATIONS]
    if invalid_orchestrations:
        raise ValueError(
            f"Invalid orchestration(s): {invalid_orchestrations}. Valid: {sorted(VALID_ORCHESTRATIONS)}"
        )


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_pipeline_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if args.pipeline_config:
        raw = json.loads(Path(args.pipeline_config).read_text(encoding="utf-8"))
        model_entries = raw.get("models", [])
        models = [
            ModelConfig(
                model_id=entry["model_id"],
                vllm=VllmConfig(
                    model=entry.get("vllm", {}).get("model", _default_vllm_model_from_model_id(entry["model_id"])),
                    port=int(entry.get("vllm", {}).get("port", 8000)),
                    max_model_len=entry.get("vllm", {}).get("max_model_len", 32768),
                    reasoning_parser=entry.get("vllm", {}).get("reasoning_parser", "qwen3"),
                    enable_auto_tool_choice=bool(entry.get("vllm", {}).get("enable_auto_tool_choice", True)),
                    tool_call_parser=entry.get("vllm", {}).get("tool_call_parser", "qwen3_coder"),
                    gpu_memory_utilization=float(entry.get("vllm", {}).get("gpu_memory_utilization", 0.95)),
                    startup_timeout_seconds=int(entry.get("vllm", {}).get("startup_timeout_seconds", 1800)),
                    extra_args=list(entry.get("vllm", {}).get("extra_args", [])),
                ),
            )
            for entry in model_entries
        ]
        config = PipelineConfig(
            models=models,
            scenarios=_dedupe_preserve_order(list(raw.get("scenarios", ["coding"]))),
            orchestrations=_dedupe_preserve_order(list(raw.get("orchestrations", ["graph"]))),
            task_limit=int(raw.get("task_limit", args.task_limit)),
            debug_litellm=bool(raw.get("debug_litellm", args.debug_litellm)),
        )
        _validate_pipeline_config(config)
        return config

    model_ids = args.models or [DEFAULT_LOCAL_MODEL_ID]
    model_ids = _dedupe_preserve_order(model_ids)
    scenarios = _dedupe_preserve_order(args.scenarios or ["coding"])
    orchestrations = _dedupe_preserve_order(args.orchestrations or ["graph"])

    models = []
    for idx, model_id in enumerate(model_ids):
        models.append(
            ModelConfig(
                model_id=model_id,
                vllm=VllmConfig(
                    model=_default_vllm_model_from_model_id(model_id),
                    port=args.base_port + idx,
                    max_model_len=args.max_model_len,
                    reasoning_parser=args.reasoning_parser,
                    enable_auto_tool_choice=not args.disable_auto_tool_choice,
                    tool_call_parser=args.tool_call_parser,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    startup_timeout_seconds=args.vllm_startup_timeout,
                ),
            )
        )

    config = PipelineConfig(
        models=models,
        scenarios=scenarios,
        orchestrations=orchestrations,
        task_limit=args.task_limit,
        debug_litellm=args.debug_litellm,
    )
    _validate_pipeline_config(config)
    return config


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    _validate_pipeline_config(config)
    configure_litellm_runtime(config.debug_litellm)

    aggregate_results: list[dict[str, Any]] = []

    for model_cfg in config.models:
        progress_logger = logging.getLogger("progress")
        os.environ["BENCH_LOCAL_MODEL_ID"] = model_cfg.model_id
        os.environ["OPENAI_API_BASE"] = model_cfg.api_base
        os.environ["OPENAI_API_KEY"] = "local-testing"

        install_litellm_completion_guard(default_model_id=model_cfg.model_id)
        install_marble_safety_patches(model_cfg.action_model_name)

        for scenario in config.scenarios:
            for orchestration in config.orchestrations:
                run_paths = configure_run_logging(Path("logs"))
                server = VllmServer(model_cfg.vllm)
                progress_logger.info("Starting vLLM server for model: %s", model_cfg.model_id)
                server.start(run_paths.run_dir)
                try:
                    server.wait_until_ready(timeout_seconds=model_cfg.vllm.startup_timeout_seconds)
                    progress_logger.info("vLLM ready at %s", model_cfg.api_base)

                    results = run_benchmark(
                        local_api_base=model_cfg.api_base,
                        local_model_id=model_cfg.model_id,
                        verbose_log_path=run_paths.verbose_log_path,
                        scenario=scenario,
                        coordinate_mode=orchestration,
                        task_limit=config.task_limit,
                    )
                    summary = summarize_results(results)
                    write_evaluation_log(
                        run_paths.eval_log_path,
                        run_paths.run_timestamp,
                        run_paths.run_dir,
                        summary,
                    )

                    aggregate_results.append(
                        {
                            "model_id": model_cfg.model_id,
                            "scenario": scenario,
                            "orchestration": orchestration,
                            "run_directory": str(run_paths.run_dir),
                            "results": summary,
                        }
                    )
                finally:
                    progress_logger.info("Stopping vLLM server for model: %s", model_cfg.model_id)
                    server.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    aggregate_path = Path("logs") / f"pipeline_summary_{timestamp}.json"
    aggregate_path.write_text(json.dumps({"runs": aggregate_results}, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "summary_path": str(aggregate_path),
        "runs": aggregate_results,
    }
