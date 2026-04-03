import json
import importlib
import logging
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import litellm


class ProgressOnlyFilter(logging.Filter):
    """Allow only high-level progress messages on the console."""

    def filter(self, record):
        return record.name == "progress"


def _quiet_non_progress_console_handlers():
    """Prevent non-progress loggers from writing directly to terminal."""

    logger_dict = logging.root.manager.loggerDict
    for logger in logger_dict.values():
        if not isinstance(logger, logging.Logger):
            continue
        if logger.name == "progress":
            continue

        logger.handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        logger.propagate = True


def _install_non_progress_stream_handler_guard():
    """Prevent future non-progress stream handlers from attaching to terminal."""

    if getattr(logging, "_progress_guard_installed", False):
        return

    original_add_handler = logging.Logger.addHandler

    def guarded_add_handler(self, handler):
        is_stream = isinstance(handler, logging.StreamHandler)
        is_file = isinstance(handler, logging.FileHandler)
        if self.name != "progress" and is_stream and not is_file:
            return
        return original_add_handler(self, handler)

    logging.Logger.addHandler = guarded_add_handler
    logging._progress_guard_installed = True



def main():


    # Keep LiteLLM debug noise off by default.
    # Set BENCH_DEBUG_LITELLM=1 to restore full request/response traces.
    debug_litellm = os.getenv("BENCH_DEBUG_LITELLM", "0") == "1"
    if debug_litellm:
        litellm._turn_on_debug()
    else:
        litellm.set_verbose = False
        for noisy_logger in ("LiteLLM", "litellm", "httpx", "openai"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    # --- ENABLE FILE LOGGING ---
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = log_dir / f"benchmark_{run_timestamp}.log"

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Keep console concise: remove existing stream handlers and install a
    # progress-only stream handler. Full detail still goes to the file handler.
    root_logger.handlers = [
        h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)
    ]
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(ProgressOnlyFilter())
    root_logger.addHandler(console_handler)

    has_run_file_handler = any(
        isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == str(run_log_path)
        for h in root_logger.handlers
    )
    if not has_run_file_handler:
        file_handler = RotatingFileHandler(str(run_log_path), maxBytes=10485760, backupCount=5)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    progress_logger = logging.getLogger("progress")
    progress_logger.info("Benchmark run started")
    progress_logger.info("Log file: %s", run_log_path)

    _install_non_progress_stream_handler_guard()
    # --------------------------

    # Import benchmark modules after logging setup so MARBLE logger creation
    # happens with file logging already active.
    from maseval.benchmark.multiagentbench import (
        MarbleMultiAgentBenchBenchmark,
        configure_model_ids,
        load_tasks,
    )

    # Some libraries install their own console handlers; strip them so only
    # `progress` messages hit the terminal while all logs still reach file.
    _quiet_non_progress_console_handlers()

    # Qwen chat templates on vLLM may reject requests containing no user turn.
    # MARBLE can emit system-only prompts in a few internal steps, so we guard
    # all LiteLLM chat calls by appending a minimal user message when needed.
    # We also normalize reasoning-only responses where content can be null.
    _original_completion = litellm.completion

    def _get_role(msg):
        if isinstance(msg, dict):
            return msg.get("role")
        return getattr(msg, "role", None)

    def _message_content(msg):
        if isinstance(msg, dict):
            return msg.get("content")
        return getattr(msg, "content", None)

    def _set_message_content(msg, value):
        if isinstance(msg, dict):
            msg["content"] = value
        else:
            setattr(msg, "content", value)

    def _extract_reasoning(msg):
        if isinstance(msg, dict):
            return msg.get("reasoning")
        return getattr(msg, "reasoning", None)

    def _looks_like_json_prompt(messages):
        combined = "\n".join(
            str(_message_content(m) or "")
            for m in messages
            if _get_role(m) in {"system", "user"}
        ).lower()
        signals = [
            "respond with a json object",
            "return the final output into a json",
            "valid json",
            '"continue"',
            '"tasks"',
            '"solution.py"',
        ]
        return any(signal in combined for signal in signals)

    def _ensure_user_turn(messages):
        has_user = any(_get_role(m) == "user" for m in messages)
        if has_user:
            return list(messages)
        patched = list(messages)
        patched.append({"role": "user", "content": "Please continue."})
        return patched

    def _first_choice(response):
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            return choices[0]
        return None

    def _normalize_first_message_content(response):
        choice = _first_choice(response)
        if not choice:
            return response

        message = getattr(choice, "message", None)
        if message is None:
            return response

        content = _message_content(message)
        if isinstance(content, str) and content.strip():
            return response

        reasoning = _extract_reasoning(message)
        if isinstance(reasoning, str) and reasoning.strip():
            _set_message_content(message, reasoning)
        else:
            _set_message_content(message, "")

        return response

    def _needs_json_retry(response):
        choice = _first_choice(response)
        if not choice:
            return True
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        # Tool-call responses often have empty content by design. Do not retry.
        if finish_reason == "tool_calls" or (tool_calls and len(tool_calls) > 0):
            return False
        content = _message_content(message) if message is not None else None
        missing_content = not (isinstance(content, str) and content.strip())
        return finish_reason == "length" or missing_content

    def _to_int_or_none(value):
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(float(text))
            except Exception:
                return None
        return None

    def _coerce_completion_params(kwargs_dict):
        """Normalize model/numeric params to avoid LiteLLM type/provider errors."""
        params = dict(kwargs_dict)

        default_model = os.getenv("BENCH_LOCAL_MODEL_ID", "openai/Qwen/Qwen3.5-27B")
        model = params.get("model")
        if isinstance(model, str):
            model = model.strip()
            if model == "gpt-3.5-turbo":
                model = default_model
            if "/" in model and not re.match(r"^[a-zA-Z0-9_.-]+/", model):
                # Extremely defensive: malformed prefix-like values fall back.
                model = default_model
            # If model has no provider prefix (e.g. Qwen/Qwen3.5-27B), force openai/.
            if not re.match(r"^[a-zA-Z0-9_.-]+/.+", model):
                model = f"openai/{model}"
            elif model.startswith("Qwen/"):
                model = f"openai/{model}"
            params["model"] = model

        for int_key in ("max_tokens", "timeout", "seed"):
            if int_key in params:
                casted = _to_int_or_none(params.get(int_key))
                if casted is not None:
                    params[int_key] = casted

        for float_key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if float_key in params and isinstance(params.get(float_key), str):
                try:
                    params[float_key] = float(params[float_key].strip())
                except Exception:
                    pass

        return params

    def _completion_with_user_guard(*args, **kwargs):
        local_kwargs = _coerce_completion_params(kwargs)
        messages = local_kwargs.get("messages")
        should_retry_json = False
        if isinstance(messages, list):
            local_kwargs["messages"] = _ensure_user_turn(messages)
            should_retry_json = _looks_like_json_prompt(local_kwargs["messages"])

        try:
            response = _original_completion(*args, **local_kwargs)
        except Exception as completion_error:
            diag = {
                "model": local_kwargs.get("model"),
                "max_tokens": local_kwargs.get("max_tokens"),
                "max_tokens_type": type(local_kwargs.get("max_tokens")).__name__,
                "temperature": local_kwargs.get("temperature"),
                "temperature_type": type(local_kwargs.get("temperature")).__name__,
            }
            logging.getLogger("runner").error(
                "LiteLLM completion failed with normalized params: %s | error=%s",
                diag,
                completion_error,
            )
            raise
        response = _normalize_first_message_content(response)

        if should_retry_json and _needs_json_retry(response):
            retry_kwargs = _coerce_completion_params(local_kwargs)
            retry_messages = list(retry_kwargs.get("messages") or [])
            retry_messages.append(
                {
                    "role": "user",
                    "content": "Return only valid JSON with no extra text.",
                }
            )
            retry_kwargs["messages"] = retry_messages
            current_max_tokens = _to_int_or_none(retry_kwargs.get("max_tokens"))
            if current_max_tokens is not None:
                retry_kwargs["max_tokens"] = min(max(512, current_max_tokens * 2), 4096)
            else:
                retry_kwargs["max_tokens"] = 1024

            retry_response = _original_completion(*args, **retry_kwargs)
            return _normalize_first_message_content(retry_response)

        return response

    litellm.completion = _completion_with_user_guard

    # MARBLE parser safety: tolerate empty/non-string model responses by
    # returning conservative JSON fallbacks instead of raising.
    try:
        if "marble" not in sys.modules:
            marble_pkg = importlib.import_module(
                "maseval.benchmark.multiagentbench.marble.marble"
            )
            sys.modules["marble"] = marble_pkg

        planner_modules = []
        for planner_path in (
            "maseval.benchmark.multiagentbench.marble.marble.engine.engine_planner",
            "marble.engine.engine_planner",
        ):
            try:
                planner_modules.append(importlib.import_module(planner_path))
            except Exception:
                continue

        for planner_module in planner_modules:
            _orig_json_parse = planner_module.json_parse

            def _safe_json_parse(input_str, _orig=_orig_json_parse):
                if not isinstance(input_str, str) or not input_str.strip():
                    return {"continue": True}

                try:
                    return _orig(input_str)
                except Exception:
                    match = re.search(r"\{[\s\S]*\}", input_str)
                    if match:
                        try:
                            return json.loads(match.group(0))
                        except Exception:
                            pass
                    return {"continue": True}

            planner_module.json_parse = _safe_json_parse
    except Exception as patch_error:
        logging.getLogger("runner").warning("Engine planner patch skipped: %s", patch_error)

    # --- CONFIGURATION ---
    LOCAL_API_BASE = "http://localhost:8000/v1"  # vLLM OpenAI-compatible endpoint
    LOCAL_MODEL_ID = "openai/Qwen/Qwen3.5-27B"
    LOCAL_ACTION_MODEL_NAME = LOCAL_MODEL_ID.replace("openai/", "", 1)
    os.environ["BENCH_LOCAL_MODEL_ID"] = LOCAL_MODEL_ID
    # Set global environment variables that LiteLLM will automatically detect
    os.environ["OPENAI_API_BASE"] = LOCAL_API_BASE
    os.environ["OPENAI_API_KEY"] = "local-testing"  # Placeholder required by LiteLLM
    # ---------------------

    # MARBLE action safety: if tool-call args are truncated and miss required
    # `model_name`, inject a default before dispatching to action handlers.
    try:
        def _inject_model_name(action_name, arguments):
            safe_args = dict(arguments or {})
            if action_name in {
                "create_solution",
                "give_advice_and_revise",
                "give_advice_and_revise_code",
                "debug_solution",
                "revise_solution",
            }:
                model_name = safe_args.get("model_name")
                if not isinstance(model_name, str) or not model_name.strip():
                    safe_args["model_name"] = LOCAL_ACTION_MODEL_NAME
            return safe_args

        # Patch both possible BaseEnvironment module paths to avoid alias drift.
        base_env_modules = []
        for base_env_path in (
            "maseval.benchmark.multiagentbench.marble.marble.environments.base_env",
            "marble.environments.base_env",
        ):
            try:
                base_env_modules.append(importlib.import_module(base_env_path))
            except Exception:
                continue

        for base_env_module in base_env_modules:
            BaseEnvironment = base_env_module.BaseEnvironment
            _orig_apply_action = BaseEnvironment.apply_action

            def _safe_apply_action(self, agent_id, action_name, arguments, _orig=_orig_apply_action):
                safe_args = _inject_model_name(action_name, arguments)
                try:
                    return _orig(self, agent_id, action_name, safe_args)
                except Exception as action_error:
                    logging.getLogger("runner").error(
                        "apply_action failed: agent=%s action=%s arg_types=%s error=%s",
                        agent_id,
                        action_name,
                        {k: type(v).__name__ for k, v in (safe_args or {}).items()},
                        action_error,
                    )
                    raise

            BaseEnvironment.apply_action = _safe_apply_action

        # Additional guard at the agent layer ensures model_name is repaired
        # even if environment dispatch is reached through a different binding.
        base_agent_module = importlib.import_module(
            "maseval.benchmark.multiagentbench.marble.marble.agent.base_agent"
        )
        BaseAgent = base_agent_module.BaseAgent
        _orig_act = BaseAgent.act

        def _safe_act(self, task):
            original_apply_action = self.env.apply_action

            def _wrapped_apply_action(agent_id, action_name, arguments):
                safe_args = _inject_model_name(action_name, arguments)
                return original_apply_action(
                    agent_id=agent_id,
                    action_name=action_name,
                    arguments=safe_args,
                )

            self.env.apply_action = _wrapped_apply_action
            try:
                return _orig_act(self, task)
            except Exception as agent_error:
                logging.getLogger("runner").error(
                    "BaseAgent.act failed: agent=%s task_type=%s task_preview=%s error=%s",
                    getattr(self, "agent_id", "unknown"),
                    type(task).__name__,
                    str(task)[:160],
                    agent_error,
                )
                raise
            finally:
                self.env.apply_action = original_apply_action

        BaseAgent.act = _safe_act
    except Exception as action_patch_error:
        logging.getLogger("runner").warning("Action patch skipped: %s", action_patch_error)

    # Load tasks and configure them to use your LOCAL model name
    tasks = load_tasks("coding", limit=1)
    configure_model_ids(tasks, agent_model_id=LOCAL_MODEL_ID)
    progress_logger.info("Task setup complete")
    progress_logger.info("Model configured: %s", LOCAL_MODEL_ID)

    class MyMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
        def get_model_adapter(self, model_id, **kwargs):
            from maseval.interface.openai import OpenAIModelAdapter
            
            # We MUST provide base_url and a placeholder api_key for local use
            adapter = OpenAIModelAdapter(
                model_id=model_id, 
                base_url=LOCAL_API_BASE,
                api_key="local-token" 
            )
            
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Initialize and run
    benchmark = MyMarbleBenchmark()
    agent_data = {} # Marble usually handles model mapping via configure_model_ids
    _quiet_non_progress_console_handlers()
    progress_logger.info("Orchestration started")
    with open(run_log_path, "a", encoding="utf-8") as terminal_sink:
        with redirect_stdout(terminal_sink), redirect_stderr(terminal_sink):
            results = benchmark.run(tasks, agent_data=agent_data)
    progress_logger.info("Orchestration finished")

    # Emit concise terminal progression while detailed traces remain in file logs.
    for result in results:
        progress_logger.info("Task: %s | Status: %s", result["task_id"], result["status"])
        if result['status'] == "setup_failed":
            progress_logger.info("Task setup failed")
        if result.get('eval'):
            progress_logger.info("Passed: %s", result['eval'][0]['passed'])

if __name__ == "__main__":
    main()
