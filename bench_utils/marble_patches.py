from __future__ import annotations

import importlib
import json
import logging
import re
import sys
import traceback


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


def _inject_model_name(action_name, arguments, local_action_model_name):
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
            safe_args["model_name"] = local_action_model_name
    return safe_args


def _patch_json_parse(planner_module):
    original_json_parse = planner_module.json_parse

    def _safe_json_parse(input_str, _orig=original_json_parse):
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


def _patch_base_environment(base_env_module, local_action_model_name):
    BaseEnvironment = base_env_module.BaseEnvironment
    original_init = BaseEnvironment.__init__
    original_apply_action = BaseEnvironment.apply_action

    def _safe_init(self, name, config):
        original_init(self, name, config)
        coerced_iterations = _to_int_or_none(getattr(self, "max_iterations", None))
        if coerced_iterations is None:
            coerced_iterations = 10
        self.max_iterations = coerced_iterations
        coerced_current = _to_int_or_none(getattr(self, "current_iteration", None))
        if coerced_current is None:
            coerced_current = 0
        self.current_iteration = coerced_current

    def _safe_apply_action(self, agent_id, action_name, arguments):
        coerced_iterations = _to_int_or_none(getattr(self, "max_iterations", None))
        if coerced_iterations is None:
            coerced_iterations = 10
        self.max_iterations = coerced_iterations

        coerced_current = _to_int_or_none(getattr(self, "current_iteration", None))
        if coerced_current is None:
            coerced_current = 0
        self.current_iteration = coerced_current

        safe_args = _inject_model_name(action_name, arguments, local_action_model_name)
        try:
            return original_apply_action(self, agent_id, action_name, safe_args)
        except Exception as action_error:
            logging.getLogger("runner").error(
                "apply_action failed: agent=%s action=%s arg_types=%s error=%s\ntraceback:\n%s",
                agent_id,
                action_name,
                {key: type(value).__name__ for key, value in (safe_args or {}).items()},
                action_error,
                traceback.format_exc(),
            )
            raise

    BaseEnvironment.__init__ = _safe_init
    BaseEnvironment.apply_action = _safe_apply_action


def _patch_base_agent(base_agent_module, local_action_model_name):
    BaseAgent = base_agent_module.BaseAgent
    original_act = BaseAgent.act

    def _safe_act(self, task):
        original_apply_action = self.env.apply_action

        def _wrapped_apply_action(agent_id, action_name, arguments):
            safe_args = _inject_model_name(action_name, arguments, local_action_model_name)
            return original_apply_action(
                agent_id=agent_id,
                action_name=action_name,
                arguments=safe_args,
            )

        self.env.apply_action = _wrapped_apply_action
        try:
            return original_act(self, task)
        except Exception as agent_error:
            logging.getLogger("runner").error(
                "BaseAgent.act failed: agent=%s task_type=%s task_preview=%s error=%s\ntraceback:\n%s",
                getattr(self, "agent_id", "unknown"),
                type(task).__name__,
                str(task)[:160],
                agent_error,
                traceback.format_exc(),
            )
            raise
        finally:
            self.env.apply_action = original_apply_action

    BaseAgent.act = _safe_act


def install_marble_safety_patches(local_action_model_name: str) -> None:
    if getattr(sys.modules.get(__name__), "_benchmark_marble_guard_installed", False):
        return

    try:
        if "marble" not in sys.modules:
            marble_pkg = importlib.import_module("maseval.benchmark.multiagentbench.marble.marble")
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
            _patch_json_parse(planner_module)

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
            _patch_base_environment(base_env_module, local_action_model_name)

        base_agent_module = importlib.import_module(
            "maseval.benchmark.multiagentbench.marble.marble.agent.base_agent"
        )
        _patch_base_agent(base_agent_module, local_action_model_name)
    except Exception as patch_error:
        logging.getLogger("runner").warning("Action patch skipped: %s", patch_error)
    finally:
        module = sys.modules.get(__name__)
        if module is not None:
            setattr(module, "_benchmark_marble_guard_installed", True)
