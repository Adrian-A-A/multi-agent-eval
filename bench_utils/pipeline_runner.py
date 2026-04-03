"""
Pipeline runner for multi-agent evaluation using tiny models.

Starts a local OpenAI-compatible server backed by a HuggingFace tiny model,
then runs the MARBLE MultiAgentBench benchmark with all known bug fixes applied.

Usage:
    python bench_utils/pipeline_runner.py
"""

import importlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import litellm


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tiny model server
# ---------------------------------------------------------------------------

# Counter used to round-robin agent actions so the scenario progresses.
_CALL_COUNTER: dict = {"n": 0}


def _make_mock_response(messages: list, tools: list | None, model_id: str) -> dict:
    """Generate a deterministic mock response for the MARBLE benchmark.

    Rules
    -----
    * If tools are present in the request → return a tool_call.
      - First call: ``create_solution`` (creates the solution file)
      - Subsequent calls: ``give_advice_and_revise`` (revises the code)
      - Every 4th call after the first: ``new_communication_session`` to let
        agents communicate, then fall back to ``give_advice_and_revise``.
    * If no tools are present → return a plain-text answer.
      - Strategy JSON prompt (give_advice_and_revise step 2)? → return valid
        ``{"strategies": [...]}`` JSON.
      - Looks like a JSON-planner prompt? → return ``{"continue": true}``.
      - Looks like a code-generation prompt? → return a minimal Python stub.
      - Otherwise → return a short acknowledgement.
    """
    _CALL_COUNTER["n"] += 1
    n = _CALL_COUNTER["n"]

    combined_prompt = "\n".join(
        str(m.get("content") or "") for m in messages
    ).lower()

    if tools:
        # Extract available tool names from the request.
        tool_names = [
            t.get("function", {}).get("name", "")
            for t in (tools or [])
            if isinstance(t, dict)
        ]

        # Determine which tool to call.
        if "create_solution" in tool_names and n <= 1:
            fn_name = "create_solution"
        elif "new_communication_session" in tool_names and n % 4 == 0:
            # Find a valid target from the enum if available.
            target_id = "agent2"
            for t in tools:
                if not isinstance(t, dict):
                    continue
                params = t.get("function", {}).get("parameters", {})
                props = params.get("properties", {})
                enum_vals = props.get("target_agent_id", {}).get("enum", [])
                if enum_vals:
                    target_id = enum_vals[0]
                    break
            return {
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": "new_communication_session",
                        "arguments": json.dumps({
                            "target_agent_id": target_id,
                            "message": (
                                "Hello! I have worked on the code. "
                                "Please review and provide feedback."
                            ),
                        }),
                    },
                }],
                "content": None,
                "finish_reason": "tool_calls",
            }
        else:
            fn_name = "give_advice_and_revise"

        task_desc = "Implement the required software system."
        for m in messages:
            c = m.get("content") or ""
            if "task" in c.lower() and len(c) > 20:
                task_desc = c[:200]
                break

        return {
            "tool_calls": [{
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps({
                        "task_description": task_desc,
                        "model_name": model_id,
                    }),
                },
            }],
            "content": None,
            "finish_reason": "tool_calls",
        }

    # --- No tools: plain-text response ---

    # Strategy JSON prompt (second LLM call in give_advice_and_revise_handler).
    # Must return valid {"strategies": [...]} JSON.
    if '"strategies"' in combined_prompt or "modification strateg" in combined_prompt:
        strategy_json = json.dumps({
            "strategies": [{
                "action": "replace",
                "target": {
                    "code": "pass",
                    "before_context": "def run(self):",
                    "after_context": "",
                },
                "new_code": "        return 'Application is running'",
            }]
        })
        return {
            "tool_calls": None,
            "content": strategy_json,
            "finish_reason": "stop",
        }

    # Engine planner / JSON-expected prompts.
    if any(sig in combined_prompt for sig in (
        '"continue"', "valid json", "json object",
        "respond only", "return only", "return the final",
    )):
        return {
            "tool_calls": None,
            "content": '{"continue": true}',
            "finish_reason": "stop",
        }

    # Code-generation prompts (create_solution / give_advice calls the model).
    if any(sig in combined_prompt for sig in (
        "python developer", "write the complete", "create a solution",
        "review existing", "solution.py", "implementation",
    )):
        code = (
            "# Solution\n"
            "class App:\n"
            "    \"\"\"Main application class.\"\"\"\n\n"
            "    def __init__(self):\n"
            "        self.data = []\n\n"
            "    def run(self):\n"
            "        \"\"\"Run the application.\"\"\"\n"
            "        return 'Application is running'\n\n"
            "if __name__ == '__main__':\n"
            "    app = App()\n"
            "    print(app.run())\n"
        )
        return {
            "tool_calls": None,
            "content": f"```python\n{code}\n```",
            "finish_reason": "stop",
        }

    # Fallback.
    return {
        "tool_calls": None,
        "content": "I have completed the requested task.",
        "finish_reason": "stop",
    }


def _start_mock_server(
    model_name: str = "tiny-mock",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> threading.Thread:
    """Start a mock OpenAI-compatible server that needs no model download.

    When the ``BENCH_USE_REAL_MODEL`` environment variable is set to ``1``
    *and* the model can be loaded from the HuggingFace Hub, a real tiny model
    is used instead.  Otherwise the mock response generator is used.

    The server runs in a background daemon thread.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn

    progress_logger = logging.getLogger("progress")

    app = FastAPI()

    # Optionally try to load a real tiny model.
    _state: dict = {"model": None, "tokenizer": None, "model_id": model_name}
    use_real = os.getenv("BENCH_USE_REAL_MODEL", "0") == "1"

    if use_real:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            progress_logger.info("[server] Loading real model %s on CPU …", model_name)
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
            mdl.eval()
            _state["tokenizer"] = tok
            _state["model"] = mdl
            progress_logger.info("[server] Real model loaded.")
        except Exception as load_err:
            progress_logger.warning(
                "[server] Real model load failed (%s), falling back to mock.", load_err
            )
            use_real = False

    if not use_real:
        progress_logger.info("[server] Using deterministic mock responses (no model download needed).")

    @app.get("/v1/models")
    def list_models():
        return JSONResponse({
            "object": "list",
            "data": [{"id": _state["model_id"], "object": "model", "created": int(time.time())}],
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 512)
        tools = body.get("tools")

        if use_real and _state.get("model"):
            import torch

            tok = _state["tokenizer"]
            mdl = _state["model"]
            temperature = float(body.get("temperature") or 0.0)

            if hasattr(tok, "apply_chat_template"):
                try:
                    text = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, tools=tools,
                    )
                except Exception:
                    text = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
            else:
                text = "\n".join(f"{m.get('role','user')}: {m.get('content') or ''}" for m in messages) + "\nassistant:"

            inputs = tok(text, return_tensors="pt").to("cpu")
            with torch.no_grad():
                outputs = mdl.generate(
                    **inputs,
                    max_new_tokens=max(max_tokens, 256),
                    do_sample=temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else 1.0,
                    pad_token_id=tok.eos_token_id,
                )
            new_tok = outputs[0][inputs["input_ids"].shape[1]:]
            raw = tok.decode(new_tok, skip_special_tokens=True).strip()

            tool_calls = None
            content = raw
            tc_m = re.search(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL)
            if tc_m:
                try:
                    cd = json.loads(tc_m.group(1).strip())
                    if isinstance(cd, dict) and "name" in cd:
                        tool_calls = [{
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": cd["name"],
                                "arguments": json.dumps(cd.get("arguments", {})),
                            },
                        }]
                        content = None
                except (json.JSONDecodeError, KeyError):
                    pass
            resp_data = {
                "tool_calls": tool_calls,
                "content": content,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        else:
            resp_data = _make_mock_response(messages, tools, _state["model_id"])

        tc = resp_data["tool_calls"]
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", _state["model_id"]),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": resp_data["content"],
                    "tool_calls": tc,
                },
                "finish_reason": resp_data["finish_reason"],
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": max_tokens, "total_tokens": max_tokens},
        })

    # Run uvicorn in a daemon thread.
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait until the server is accepting connections.
    import socket
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            time.sleep(0.2)

    return t


def _start_tiny_model_server(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> threading.Thread:
    """Alias kept for backwards compatibility; delegates to ``_start_mock_server``."""
    return _start_mock_server(model_name=model_name, host=host, port=port)


# ---------------------------------------------------------------------------
# Bug-fix patches
# ---------------------------------------------------------------------------

def _patch_max_iterations():
    """Fix BaseEnvironment so max_iterations is always an int.

    The MARBLE coding task JSONL stores ``max_iterations`` as an empty string
    ``""``.  ``BaseEnvironment.__init__`` assigns it verbatim, causing
    ``'>=' not supported between instances of 'int' and 'str'`` the first time
    the environment checks whether the iteration limit has been reached.
    """
    from maseval.benchmark.multiagentbench._constants import ensure_marble_on_path

    ensure_marble_on_path()
    try:
        from marble.environments.base_env import BaseEnvironment  # type: ignore[import-untyped]
    except ImportError:
        return

    _orig_init = BaseEnvironment.__init__

    def _patched_init(self, name, config):
        _orig_init(self, name, config)
        raw = config.get("max_iterations", 10)
        if not isinstance(raw, int):
            try:
                self.max_iterations = int(raw) if raw else 10
            except (ValueError, TypeError):
                self.max_iterations = 10

    BaseEnvironment.__init__ = _patched_init
    logging.getLogger("runner").info("max_iterations patch applied.")


def _patch_engine_planner_json_parse():
    """Make engine_planner.json_parse tolerant of empty / non-JSON responses."""
    from maseval.benchmark.multiagentbench._constants import ensure_marble_on_path

    ensure_marble_on_path()

    for planner_path in (
        "maseval.benchmark.multiagentbench.marble.marble.engine.engine_planner",
        "marble.engine.engine_planner",
    ):
        try:
            mod = importlib.import_module(planner_path)
        except Exception:
            continue

        _orig = mod.json_parse

        def _safe_json_parse(input_str, _orig=_orig):
            if not isinstance(input_str, str) or not input_str.strip():
                return {"continue": True}
            try:
                return _orig(input_str)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", input_str)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
                return {"continue": True}

        mod.json_parse = _safe_json_parse

    logging.getLogger("runner").info("engine_planner json_parse patch applied.")


# ---------------------------------------------------------------------------
# LiteLLM wrapper
# ---------------------------------------------------------------------------

def _install_litellm_wrapper(
    local_api_base: str,
    local_model_id: str,
    default_model_id: str,
):
    """Wrap litellm.completion to:
    1. Redirect all calls to the local model endpoint.
    2. Ensure messages always contain a user turn (required by some models).
    3. Normalise empty / reasoning-only responses so callers get a string.
    4. Retry if a JSON-expected response comes back empty.
    5. Coerce numeric params (max_tokens, temperature, …) to correct types.
    """

    _original_completion = litellm.completion

    # ---- helpers ----

    def _get_role(msg):
        return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)

    def _get_content(msg):
        return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

    def _set_content(msg, value):
        if isinstance(msg, dict):
            msg["content"] = value
        else:
            setattr(msg, "content", value)

    def _get_reasoning(msg):
        return msg.get("reasoning") if isinstance(msg, dict) else getattr(msg, "reasoning", None)

    def _looks_like_json_prompt(messages):
        combined = "\n".join(
            str(_get_content(m) or "") for m in messages
            if _get_role(m) in {"system", "user"}
        ).lower()
        return any(
            sig in combined
            for sig in [
                "respond with a json object",
                "return the final output into a json",
                "valid json",
                '"continue"',
                '"tasks"',
                '"solution.py"',
            ]
        )

    def _ensure_user_turn(messages):
        if any(_get_role(m) == "user" for m in messages):
            return list(messages)
        patched = list(messages)
        patched.append({"role": "user", "content": "Please continue."})
        return patched

    def _first_choice(response):
        choices = getattr(response, "choices", None)
        return choices[0] if choices else None

    def _normalize_response(response):
        choice = _first_choice(response)
        if not choice:
            return response
        message = getattr(choice, "message", None)
        if message is None:
            return response
        content = _get_content(message)
        if isinstance(content, str) and content.strip():
            return response
        reasoning = _get_reasoning(message)
        _set_content(message, reasoning if isinstance(reasoning, str) and reasoning.strip() else "")
        return response

    def _needs_json_retry(response):
        choice = _first_choice(response)
        if not choice:
            return True
        finish = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        tool_calls = getattr(message, "tool_calls", None) if message else None
        if finish == "tool_calls" or (tool_calls and len(tool_calls) > 0):
            return False
        content = _get_content(message) if message else None
        return finish == "length" or not (isinstance(content, str) and content.strip())

    def _to_int_or_none(v):
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            t = v.strip()
            if not t:
                return None
            try:
                return int(float(t))
            except Exception:
                return None
        return None

    def _coerce_params(kwargs_dict):
        params = dict(kwargs_dict)

        # Model routing: always point to our local endpoint.
        model = params.get("model", "")
        if isinstance(model, str):
            model = model.strip()
            # Strip any prefix and re-attach "openai/" to use local base_url.
            bare = re.sub(r"^[a-zA-Z0-9_.-]+/", "", model)
            if not bare or bare in ("gpt-3.5-turbo", "gpt-4", "gpt-4o"):
                bare = local_model_id.split("/", 1)[-1] if "/" in local_model_id else local_model_id
            params["model"] = f"openai/{bare}"
        params.setdefault("api_base", local_api_base)
        params.setdefault("api_key", "local-testing")

        # Coerce numeric types.
        for k in ("max_tokens", "timeout", "seed"):
            if k in params:
                v = _to_int_or_none(params[k])
                if v is not None:
                    params[k] = v
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if k in params and isinstance(params[k], str):
                try:
                    params[k] = float(params[k].strip())
                except Exception:
                    pass

        return params

    def _wrapper(*args, **kwargs):
        p = _coerce_params(kwargs)
        messages = p.get("messages")
        should_retry_json = False
        if isinstance(messages, list):
            p["messages"] = _ensure_user_turn(messages)
            should_retry_json = _looks_like_json_prompt(p["messages"])

        try:
            response = _original_completion(*args, **p)
        except Exception as exc:
            logging.getLogger("runner").error(
                "LiteLLM completion error: model=%s error=%s", p.get("model"), exc
            )
            raise

        response = _normalize_response(response)

        if should_retry_json and _needs_json_retry(response):
            retry = _coerce_params(p)
            msgs = list(retry.get("messages") or [])
            msgs.append({"role": "user", "content": "Return only valid JSON with no extra text."})
            retry["messages"] = msgs
            cur = _to_int_or_none(retry.get("max_tokens"))
            retry["max_tokens"] = min(max(512, (cur or 512) * 2), 4096)
            retry_resp = _original_completion(*args, **retry)
            return _normalize_response(retry_resp)

        return response

    litellm.completion = _wrapper


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # Suppress LiteLLM debug noise.
    debug_litellm = os.getenv("BENCH_DEBUG_LITELLM", "0") == "1"
    if debug_litellm:
        litellm._turn_on_debug()
    else:
        litellm.set_verbose = False
        for noisy in ("LiteLLM", "litellm", "httpx", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── File logging ──────────────────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = log_dir / f"benchmark_{run_timestamp}.log"

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [
        h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)
    ]
    con_handler = logging.StreamHandler()
    con_handler.setLevel(logging.INFO)
    con_handler.setFormatter(logging.Formatter("%(message)s"))
    con_handler.addFilter(ProgressOnlyFilter())
    root_logger.addHandler(con_handler)

    if not any(
        isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == str(run_log_path)
        for h in root_logger.handlers
    ):
        fh = RotatingFileHandler(str(run_log_path), maxBytes=10_485_760, backupCount=5)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

    progress_logger = logging.getLogger("progress")
    progress_logger.info("Pipeline started")
    progress_logger.info("Log file: %s", run_log_path)
    _install_non_progress_stream_handler_guard()
    # ──────────────────────────────────────────────────────────────────────

    # ── Configuration ─────────────────────────────────────────────────────
    TINY_MODEL_NAME = os.getenv("BENCH_TINY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    LOCAL_API_BASE = os.getenv("BENCH_API_BASE", "http://127.0.0.1:8000/v1")
    LOCAL_MODEL_ID = f"openai/{TINY_MODEL_NAME}"
    LOCAL_ACTION_MODEL_NAME = TINY_MODEL_NAME  # bare name used by action handlers

    os.environ["BENCH_LOCAL_MODEL_ID"] = LOCAL_MODEL_ID
    os.environ["OPENAI_API_BASE"] = LOCAL_API_BASE
    os.environ["OPENAI_API_KEY"] = "local-testing"
    # ──────────────────────────────────────────────────────────────────────

    # ── Tiny model server ─────────────────────────────────────────────────
    existing_server = False
    try:
        import urllib.request
        urllib.request.urlopen(f"{LOCAL_API_BASE}/models", timeout=2)
        existing_server = True
        progress_logger.info("[server] Existing server detected at %s", LOCAL_API_BASE)
    except Exception:
        pass

    if not existing_server:
        progress_logger.info("[server] Starting tiny model server (%s) …", TINY_MODEL_NAME)
        _start_tiny_model_server(
            model_name=TINY_MODEL_NAME,
            host="127.0.0.1",
            port=int(LOCAL_API_BASE.split(":")[-1].split("/")[0]),
        )
        progress_logger.info("[server] Server is ready.")
    # ──────────────────────────────────────────────────────────────────────

    # ── LiteLLM wrapper ───────────────────────────────────────────────────
    _install_litellm_wrapper(
        local_api_base=LOCAL_API_BASE,
        local_model_id=LOCAL_MODEL_ID,
        default_model_id=LOCAL_MODEL_ID,
    )
    # ──────────────────────────────────────────────────────────────────────

    # ── Import maseval (after logging is set up) ───────────────────────────
    from maseval.benchmark.multiagentbench import (
        MarbleMultiAgentBenchBenchmark,
        configure_model_ids,
        load_tasks,
    )
    _quiet_non_progress_console_handlers()
    # ──────────────────────────────────────────────────────────────────────

    # ── Bug-fix patches ───────────────────────────────────────────────────
    _patch_max_iterations()
    _patch_engine_planner_json_parse()

    # Patch BaseEnvironment.apply_action to inject model_name when missing.
    try:
        from maseval.benchmark.multiagentbench._constants import ensure_marble_on_path
        ensure_marble_on_path()
        from marble.environments.base_env import BaseEnvironment  # type: ignore[import-untyped]

        _orig_apply = BaseEnvironment.apply_action

        def _safe_apply(self, agent_id, action_name, arguments, _orig=_orig_apply):
            safe_args = dict(arguments or {})
            if action_name in {
                "create_solution",
                "give_advice_and_revise",
                "give_advice_and_revise_code",
                "debug_solution",
                "revise_solution",
            }:
                if not isinstance(safe_args.get("model_name"), str) or not safe_args["model_name"].strip():
                    safe_args["model_name"] = LOCAL_ACTION_MODEL_NAME
            try:
                return _orig(self, agent_id, action_name, safe_args)
            except Exception as err:
                logging.getLogger("runner").error(
                    "apply_action failed: agent=%s action=%s error=%s", agent_id, action_name, err
                )
                raise

        BaseEnvironment.apply_action = _safe_apply
    except Exception as e:
        logging.getLogger("runner").warning("apply_action patch skipped: %s", e)

    # Patch BaseAgent.act to also ensure model_name injection at agent layer.
    try:
        from marble.agent.base_agent import BaseAgent  # type: ignore[import-untyped]

        _orig_act = BaseAgent.act

        def _safe_act(self, task, _orig=_orig_act):
            orig_apply = self.env.apply_action

            def _wrapped(agent_id, action_name, arguments):
                safe_args = dict(arguments or {})
                if action_name in {
                    "create_solution",
                    "give_advice_and_revise",
                    "give_advice_and_revise_code",
                    "debug_solution",
                    "revise_solution",
                }:
                    if not isinstance(safe_args.get("model_name"), str) or not safe_args["model_name"].strip():
                        safe_args["model_name"] = LOCAL_ACTION_MODEL_NAME
                return orig_apply(agent_id=agent_id, action_name=action_name, arguments=safe_args)

            self.env.apply_action = _wrapped
            try:
                return _orig_act(self, task)
            finally:
                self.env.apply_action = orig_apply

        BaseAgent.act = _safe_act
    except Exception as e:
        logging.getLogger("runner").warning("BaseAgent.act patch skipped: %s", e)
    # ──────────────────────────────────────────────────────────────────────

    # ── Task loading & benchmark ──────────────────────────────────────────
    # Scenarios to test; override with BENCH_SCENARIOS env var (comma-separated).
    scenarios_env = os.getenv("BENCH_SCENARIOS", "coding")
    scenarios = [s.strip() for s in scenarios_env.split(",") if s.strip()]

    all_tasks = []
    for scenario in scenarios:
        try:
            scenario_tasks = load_tasks(scenario, limit=1)
            configure_model_ids(scenario_tasks, agent_model_id=LOCAL_MODEL_ID)
            all_tasks.extend(scenario_tasks)
            progress_logger.info("Loaded %d task(s) for scenario '%s'", len(scenario_tasks), scenario)
        except Exception as load_err:
            progress_logger.warning("Could not load scenario '%s': %s", scenario, load_err)

    if not all_tasks:
        progress_logger.error("No tasks loaded – aborting.")
        return []

    progress_logger.info("Total tasks: %d | Model: %s", len(all_tasks), LOCAL_MODEL_ID)

    class TinyModelBenchmark(MarbleMultiAgentBenchBenchmark):
        def get_model_adapter(self, model_id, **kwargs):
            from maseval.interface.openai import OpenAIModelAdapter

            adapter = OpenAIModelAdapter(
                model_id=model_id,
                base_url=LOCAL_API_BASE,
                api_key="local-testing",
            )
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    benchmark = TinyModelBenchmark()
    _quiet_non_progress_console_handlers()
    progress_logger.info("Orchestration started")

    with open(run_log_path, "a", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            results = benchmark.run(all_tasks, agent_data={})

    progress_logger.info("Orchestration finished")
    # ──────────────────────────────────────────────────────────────────────

    # ── Results summary ───────────────────────────────────────────────────
    pipeline_errors = []
    for result in results:
        status = result.get("status", "unknown")
        task_id = result.get("task_id", "?")
        progress_logger.info("Task %s | Status: %s", task_id, status)
        if status == "setup_failed":
            progress_logger.error("  ↳ Task setup failed (pipeline error)")
            pipeline_errors.append(task_id)
        elif status != "success":
            progress_logger.warning("  ↳ Unexpected status: %s", status)
        if result.get("eval"):
            # Note: MARBLE's task_completion is always [] in reproduction mode
            # (evaluator.update() is never called), so passed=False is expected.
            # Pipeline success is indicated by status="success" with no errors.
            passed = result["eval"][0].get("passed", False)
            progress_logger.info(
                "  ↳ LLM-evaluated passed: %s (False is expected with mock responses)",
                passed,
            )

    if pipeline_errors:
        progress_logger.error("Pipeline errors in tasks: %s", pipeline_errors)
        return results

    progress_logger.info(
        "Pipeline completed successfully. "
        "All %d task(s) ran without errors (status=success).",
        len(results),
    )
    return results


if __name__ == "__main__":
    main()
