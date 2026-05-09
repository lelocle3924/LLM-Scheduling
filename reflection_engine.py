import copy
import json
import os
import time
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv

import config
from utilities.logger import log_llm_call
from utilities.numeric_precision import dumps_capped, format_decimal
from utilities.stochastic_rollout import random_rollout, stochastic_rollout

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class Reflec:
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template


    @staticmethod
    def _resolve_llm_provider() -> str:
        provider_name = str(getattr(config, "LLM_API_PROVIDER", "openrouter")).strip().lower()
        if provider_name not in {"openrouter", "gemini_direct"}:
            raise ValueError(
                f"Unsupported LLM_API_PROVIDER='{provider_name}'. Use 'openrouter' or 'gemini_direct'."
            )
        return provider_name

    @staticmethod
    def _raise_for_status_with_details(response: requests.Response, provider_name: str) -> None:
        if response.ok:
            return

        response_body = ""
        try:
            response_body = response.text
        except Exception:
            response_body = "<response body unavailable>"

        if len(response_body) > 2000:
            response_body = response_body[:2000] + "...<truncated>"

        raise RuntimeError(
            f"{provider_name} HTTP {response.status_code} {response.reason}. "
            f"Response body: {response_body}"
        )

    @staticmethod
    def _build_openrouter_reasoning_config() -> dict:
        thinking_enabled = bool(getattr(config, "REFLECT_THINKING_ENABLED", False))
        thinking_effort = str(getattr(config, "REFLECT_THINKING_EFFORT", "medium")).strip().lower()
        thinking_max_tokens = int(getattr(config, "REFLECT_THINKING_MAX_TOKENS", 0) or 0)
        thinking_exclude = bool(getattr(config, "LLM_THINKING_EXCLUDE", False))

        if not thinking_enabled:
            return {"effort": "none", "exclude": True}

        reasoning_config = {"enabled": True, "exclude": thinking_exclude}
        if thinking_max_tokens > 0:
            reasoning_config["max_tokens"] = thinking_max_tokens
        else:
            reasoning_config["effort"] = thinking_effort or "medium"
        return reasoning_config

    @staticmethod
    def _call_openrouter(self, history_messages: List[Dict]) -> tuple[str, str]:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY2 (or OPENROUTER_API_KEY) is not set.")

        model_name = str(getattr(config, "REFLECT_MODEL_NAME", ""))
        api_messages = [{"role": "system", "content": str(getattr(config, "CAVEMAN_SYSTEM_PROMPT", ""))}]
        for message in history_messages:
            if not isinstance(message, dict):
                continue
            role_name = str(message.get("role", "user")).strip().lower()
            content_text = str(message.get("content", ""))
            if role_name not in {"user", "assistant"}:
                role_name = "user"
            api_messages.append({"role": role_name, "content": content_text})

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": model_name,
                "messages": api_messages,
                "temperature": float(getattr(config, "REFLECT_LLM_TEMPERATURE", 0.3)),
                "reasoning": self._build_openrouter_reasoning_config(),
            },
        )
        self._raise_for_status_with_details(response, "OpenRouter")
        llm_output = response.json()["choices"][0]["message"]["content"]
        return llm_output, model_name

    def _call_openrouter(self, history_messages: List[Dict]) -> tuple[str, str]:
        prompt_blocks = []
        for message in history_messages:
            if not isinstance(message, dict):
                continue
            role_name = str(message.get("role", "user")).strip().lower()
            content_text = str(message.get("content", "")).strip()
            if not content_text:
                continue
            prompt_blocks.append(f"[{role_name.upper()}]\n{content_text}")
        flattened_prompt = "\n\n".join(prompt_blocks)

        generation_config = {"temperature": float(getattr(config, "REFLECT_LLM_TEMPERATURE", 0.3))}
        if getattr(config, "MAX_TOKENS", 0):
            generation_config["maxOutputTokens"] = int(getattr(config, "MAX_TOKENS", 0))

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": model_name,
                "messages": api_messages,
                "temperature": float(getattr(config, "REFLECT_LLM_TEMPERATURE", 0.3)),
                "reasoning": self._build_openrouter_reasoning_config(),
            },
        )
        self._raise_for_status_with_details(response, "OpenRouter")
        response_data = response.json()
        llm_output = response.json()["choices"][0]["message"]["content"]
        return llm_output, model_name

    def _invoke_reflection_llm(
        self,
        history_messages: List[Dict],
        session_folder: str,
        iteration: int,
        call_type: str,
    ) -> str:
        provider_name = self._resolve_llm_provider()
        start_time = time.time()
        if provider_name == "openrouter":
            llm_output, effective_model = self._call_openrouter(history_messages)
        latency_seconds = time.time() - start_time

        flattened_prompt_for_log = "\n\n---\n\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}"
            for m in history_messages
            if isinstance(m, dict)
        )
        if session_folder:
            log_llm_call(
                session_folder=session_folder,
                iteration=iteration,
                call_type=call_type,
                model_name=f"{provider_name}:{effective_model}",
                prompt_text=flattened_prompt_for_log,
                llm_response=llm_output,
                latency=latency_seconds,
            )
        return (llm_output or "").strip()

    @staticmethod
    def _sort_rollouts(rollouts: List[dict]) -> List[dict]:
        return sorted(
            rollouts,
            key=lambda rollout_item: (
                float(rollout_item.get("tardiness", float("inf"))),
                float(rollout_item.get("makespan", float("inf"))),
            ),
        )

    @staticmethod
    def _dedupe_rollouts(rollouts: List[dict]) -> List[dict]:
        seen: set = set()
        unique: List[dict] = []
        for rollout_item in rollouts:
            key = (
                float(rollout_item.get("tardiness", 0.0) or 0.0),
                float(rollout_item.get("makespan", 0.0) or 0.0),
                str(rollout_item.get("action_tested", "")),
                json.dumps(rollout_item.get("trajectory", []), sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(rollout_item)
        return unique

    def _select_rollout_groups(self, rollouts: List[dict], k: int = 2) -> Tuple[List[dict], List[dict]]:
        if not rollouts:
            return [], []
        sorted_rollouts = self._sort_rollouts(rollouts)
        group_size = min(k, len(sorted_rollouts))
        best_rollouts = sorted_rollouts[:group_size]
        worst_rollouts = sorted_rollouts[-group_size:]
        best_rollouts = self._dedupe_rollouts(best_rollouts)
        worst_rollouts = self._dedupe_rollouts(worst_rollouts)
        return best_rollouts, worst_rollouts

    def _format_rollout_group(self, title: str, rollouts: List[dict]) -> str:
        if not rollouts:
            return f"{title}:\n(none)"

        lines: List[str] = [f"{title}:"]
        for rank_index, rollout_item in enumerate(rollouts, start=1):
            action_tested = str(rollout_item.get("action_tested", "N/A"))
            tardiness_value = float(rollout_item.get("tardiness", 0.0) or 0.0)
            makespan_value = float(rollout_item.get("makespan", 0.0) or 0.0)
            trajectory = list(rollout_item.get("trajectory", []) or [])
            analytics = dict(rollout_item.get("analytics", {}) or {})
            lines.append(
                f"{rank_index}. action_tested={action_tested} | tardiness={format_decimal(tardiness_value)} | "
                f"makespan={format_decimal(makespan_value)}"
            )
            lines.append(f"   trajectory: {' | '.join(trajectory) if trajectory else 'N/A'}")
            lines.append(f"   analytics: {dumps_capped(analytics, ensure_ascii=True)}")
        return "\n".join(lines)

    @staticmethod
    def _build_updated_history_payload(history_payload: List[Dict], prompt_text: str) -> List[Dict]:
        updated_history: List[Dict] = [
            {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
            for message in list(history_payload or [])
            if isinstance(message, dict)
        ]
        updated_history.append({"role": "user", "content": prompt_text})
        return updated_history

    @staticmethod
    def _finalize_history_payload(updated_history: List[Dict], assistant_text: str) -> List[Dict]:
        result = list(updated_history)
        result.append({"role": "assistant", "content": str(assistant_text or "")})
        return result

    def execute_hierarchical_reflection(
        self, state, event_description: str, session_folder: str, iteration: int
    ) -> str:
        """Executes the K-level top-down simulate-reflect-refine loop."""
        if not getattr(config, "USE_REFLECTION", True):
            return "Reflection disabled."

        levels = max(1, int(getattr(config, "REFLECTION_LEVELS", 2)))
        rollout_policy = str(getattr(config, "MCTS_ROLLOUT_POLICY", "random")).lower()
        rollout_func = random_rollout if rollout_policy == "random" else stochastic_rollout

        current_insights = "No previous insights. This is the highest planning level."
        state_summary = state.compile_prompt_elements()["machines_states"]
        history_payload: List[Dict] = []

        for level_index in range(levels - 1, -1, -1):
            rollouts: List[dict] = []

            if level_index > 0:
                planning_level_desc = f"Level {level_index} (Macro: Long-Range Sparse Exploration)"
                budget = int(getattr(config, "REFLECTION_MACRO_ROLLOUTS", 10))
                for _ in range(budget):
                    eval_state = copy.deepcopy(state)
                    tard, traj, analytics = rollout_func(eval_state)
                    rollouts.append(
                        {
                            "tardiness": tard,
                            "makespan": eval_state.current_time,
                            "trajectory": traj,
                            "analytics": analytics,
                            "action_tested": "Random Macro Path",
                        }
                    )
            else:
                planning_level_desc = "Level 0 (Micro: Immediate Action Evaluation)"
                feasible_actions = state.get_feasible_actions()
                budget_per_action = int(getattr(config, "REFLECTION_MICRO_ROLLOUTS_PER_ACTION", 3))

                for action in feasible_actions:
                    for _ in range(budget_per_action):
                        eval_state = copy.deepcopy(state)
                        eval_state.execute_action(action["job"], action["op"], action["machine"])
                        while not eval_state.get_feasible_actions() and not all(
                            s == "completed" for s in eval_state.job_status.values()
                        ):
                            ev_type, _, _ = eval_state.process_next_event()
                            if ev_type is None:
                                break

                        tard, traj, analytics = rollout_func(eval_state)
                        action_str = f"J{action['job']}O{action['op']}->M{action['machine']}"
                        rollouts.append(
                            {
                                "tardiness": tard,
                                "makespan": eval_state.current_time,
                                "trajectory": [action_str] + list(traj),
                                "analytics": analytics,
                                "action_tested": action_str,
                            }
                        )

            best_rollouts, worst_rollouts = self._select_rollout_groups(rollouts)
            formatted_text = "\n\n".join(
                [
                    self._format_rollout_group("Best Rollouts", best_rollouts),
                    self._format_rollout_group("Worst Rollouts", worst_rollouts),
                ]
            )

            prompt_text = self.prompt_template.replace("{current_time}", format_decimal(state.current_time))
            prompt_text = prompt_text.replace("{event_description}", str(event_description))
            prompt_text = prompt_text.replace("{planning_level_description}", planning_level_desc)
            prompt_text = prompt_text.replace("{previous_insights}", current_insights)
            prompt_text = prompt_text.replace("{state_summary}", state_summary)
            prompt_text = prompt_text.replace("{formatted_rollouts_text}", formatted_text)

            updated_history = self._build_updated_history_payload(history_payload, prompt_text)

            try:
                current_insights = self._invoke_reflection_llm(
                    updated_history,
                    session_folder=session_folder,
                    iteration=iteration,
                    call_type=f"Hierarchical_Reflection_L{level_index}",
                )
            except Exception as reflection_error:
                current_insights = f"Reflection failed at Level {level_index}: {reflection_error}"
                history_payload = self._finalize_history_payload(updated_history, current_insights)
                break

            history_payload = self._finalize_history_payload(updated_history, current_insights)

        safe_event_name = "".join(
            character for character in event_description if character.isalpha() or character.isdigit() or character == " "
        ).rstrip()
        if not safe_event_name:
            safe_event_name = "event"
        time_suffix = int(round(float(getattr(state, "current_time", 0.0))))
        dump_path = os.path.join(
            session_folder,
            f"reflection_trace_{safe_event_name.replace(' ', '_')}_T{time_suffix}.md",
        )
        self.dump_history_payload_markdown(dump_path, history_payload)

        return current_insights

    def dump_history_payload_markdown(self, output_path: str, history_payload: List[Dict]) -> None:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        markdown_lines: List[str] = ["# Reflection Conversation History", ""]
        for message_index, message in enumerate(history_payload or [], start=1):
            role_name = str(message.get("role", "unknown")).strip().lower()
            content_text = str(message.get("content", "")).strip()
            markdown_lines.append(f"## Message {message_index} - {role_name}")
            markdown_lines.append("")
            markdown_lines.append(content_text if content_text else "(empty)")
            markdown_lines.append("")

        with open(output_path, "w", encoding="utf-8") as markdown_file:
            markdown_file.write("\n".join(markdown_lines).strip() + "\n")
