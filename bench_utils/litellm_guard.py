from __future__ import annotations

import logging
import os
import re

import litellm


def configure_litellm_runtime(debug_enabled: bool) -> None:
    if debug_enabled:
        litellm._turn_on_debug()
        return

    litellm.set_verbose = False
    for noisy_logger in ("LiteLLM", "litellm", "httpx", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def install_litellm_completion_guard(default_model_id: str | None = None) -> None:
    """Patch LiteLLM completion calls to tolerate the benchmark's edge cases."""

    if getattr(litellm, "_benchmark_completion_guard_installed", False):
        return

    original_completion = litellm.completion

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
            str(_message_content(message) or "")
            for message in messages
            if _get_role(message) in {"system", "user"}
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
        has_user = any(_get_role(message) == "user" for message in messages)
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
        params = dict(kwargs_dict)

        default_model = default_model_id or os.getenv("BENCH_LOCAL_MODEL_ID", "openai/Qwen/Qwen3.5-27B")
        model = params.get("model")
        if isinstance(model, str):
            model = model.strip()
            if model == "gpt-3.5-turbo":
                model = default_model
            if "/" in model and not re.match(r"^[a-zA-Z0-9_.-]+/", model):
                model = default_model
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
            response = original_completion(*args, **local_kwargs)
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

            retry_response = original_completion(*args, **retry_kwargs)
            return _normalize_first_message_content(retry_response)

        return response

    litellm.completion = _completion_with_user_guard
    litellm._benchmark_completion_guard_installed = True
