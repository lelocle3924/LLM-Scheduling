import json
import math
import os
import re
import time

import requests
from dotenv import load_dotenv

import config
from utilities.logger import log_llm_call
from utilities.numeric_precision import dumps_capped

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class LLMAgent:
    """The Unified LLM Brain for policy and prior estimations."""

    DEFAULT_LESSONS_TEXT = "No lessons gathered yet. Prioritize keeping machines busy."

    def __init__(self, action_prompt_template: str, prior_prompt_template: str):
        self.action_prompt = action_prompt_template
        self.prior_prompt = prior_prompt_template

    def _resolve_llm_provider(self) -> str:
        provider = str(getattr(config, "LLM_API_PROVIDER", "openrouter")).strip().lower()
        if provider not in {"openrouter", "gemini"}:
            raise ValueError(
                f"Unsupported LLM_API_PROVIDER='{provider}'. Use 'openrouter' or 'gemini'."
            )
        return provider

    @staticmethod
    def _raise_for_status_with_response_details(response: requests.Response, provider_name: str) -> None:
        if response.ok:
            return

        response_body_preview = ""
        try:
            response_body_preview = response.text
        except Exception:
            response_body_preview = "<response body unavailable>"

        if len(response_body_preview) > 2000:
            response_body_preview = response_body_preview[:2000] + "...<truncated>"

        raise RuntimeError(
            f"{provider_name} HTTP {response.status_code} {response.reason}. "
            f"Response body: {response_body_preview}"
        )

    @staticmethod
    def _resolve_prefixed_config(prefix: str, suffix: str, default_value):
        return getattr(config, f"{prefix}_{suffix}", default_value)

    @staticmethod
    def _to_gemini_model_name(model_name: str) -> str:
        normalized_model_name = str(model_name).strip()
        if "/" in normalized_model_name:
            return normalized_model_name.split("/", 1)[1]
        return normalized_model_name

    def _build_openrouter_reasoning_config(self, prefix: str) -> dict:
        thinking_enabled = bool(self._resolve_prefixed_config(prefix, "THINKING_ENABLED", False))
        thinking_effort = str(
            self._resolve_prefixed_config(prefix, "THINKING_EFFORT", "medium")
        ).strip().lower()
        thinking_max_tokens = int(self._resolve_prefixed_config(prefix, "THINKING_MAX_TOKENS", 0) or 0)
        thinking_exclude = bool(getattr(config, "LLM_THINKING_EXCLUDE", False))

        if not thinking_enabled:
            return {"effort": "none", "exclude": True}

        reasoning_config = {"enabled": True, "exclude": thinking_exclude}
        if thinking_max_tokens > 0:
            reasoning_config["max_tokens"] = thinking_max_tokens
        else:
            reasoning_config["effort"] = thinking_effort or "medium"
        return reasoning_config

    def _call_openrouter(
        self,
        prompt_text: str,
        model_name: str,
        temperature: float,
        prefix: str,
    ) -> tuple[str, str]:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY2 (or OPENROUTER_API_KEY) is not set.")

        reasoning_config = self._build_openrouter_reasoning_config(prefix)
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": config.CAVEMAN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "temperature": float(temperature),
                "reasoning": reasoning_config,
            },
        )
        self._raise_for_status_with_response_details(response, "OpenRouter")
        llm_output = response.json()["choices"][0]["message"]["content"]
        return llm_output, str(model_name)


    def _call_api(
        self,
        prompt_text: str,
        model_name: str,
        temperature: float,
        prefix: str,
        session_folder: str = None,
        iteration: int = 0,
        call_type: str = "General",
    ) -> str:
        for attempt in range(config.MAX_RETRIES):
            try:
                start_time = time.time()
                provider = self._resolve_llm_provider()
                llm_output, effective_model_name = self._call_openrouter(
                    prompt_text=prompt_text,
                    model_name=model_name,
                    temperature=temperature,
                    prefix=prefix,
                )
                latency = time.time() - start_time

                if session_folder:
                    log_llm_call(
                        session_folder=session_folder,
                        iteration=iteration,
                        call_type=call_type,
                        model_name=f"{provider}:{effective_model_name}",
                        prompt_text=prompt_text,
                        llm_response=llm_output,
                        latency=latency,
                    )
                return llm_output
            except Exception as api_error:
                print(f"API attempt {attempt + 1} failed for {call_type}: {api_error}")
        return ""

    @staticmethod
    def _extract_key_insights_text(lessons_text: str) -> str:
        """Keep only <key_insights>...</key_insights> block when present."""
        if not lessons_text:
            return ""
        key_insights_match = re.search(
            r"<key_insights>(.*?)</key_insights>",
            lessons_text,
            re.DOTALL | re.IGNORECASE,
        )
        if not key_insights_match:
            return lessons_text
        extracted_key_insights = key_insights_match.group(1).strip()
        return extracted_key_insights or lessons_text

    def _load_lessons_text(self, session_folder: str) -> tuple[str, int]:
        lessons_file_path = os.path.join(session_folder, "lessons.md")
        if not session_folder or not os.path.exists(lessons_file_path):
            return self.DEFAULT_LESSONS_TEXT, 0

        try:
            with open(lessons_file_path, "r", encoding="utf-8") as lessons_file:
                lessons_text = lessons_file.read().strip()
        except OSError:
            return self.DEFAULT_LESSONS_TEXT, 0

        if not lessons_text:
            return self.DEFAULT_LESSONS_TEXT, 0

        lessons_text = self._extract_key_insights_text(lessons_text)

        non_empty_lines = [line for line in lessons_text.splitlines() if line.strip()]
        return lessons_text, len(non_empty_lines)

    @staticmethod
    def _to_float_or_none(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_scores(self, raw_scores: dict[str, float], action_count: int) -> dict:
        if action_count <= 0:
            return {}

        normalized_scores = {
            str(action_index): float(raw_scores.get(str(action_index), 0.0))
            for action_index in range(action_count)
        }
        total_score = sum(normalized_scores.values())

        if total_score <= 0.0:
            uniform_probability = 1.0 / action_count
            return {str(action_index): uniform_probability for action_index in range(action_count)}

        return {
            action_index: score / total_score
            for action_index, score in normalized_scores.items()
        }

    def _parse_operation_scores(self, parsed_json: dict) -> dict[str, float]:
        """Parse model scores defensively to avoid crashing on partial JSON outputs."""
        operation_scores = parsed_json.get("operation_scores")
        if not isinstance(operation_scores, dict):
            return {}

        parsed_scores: dict[str, float] = {}
        for key, value in operation_scores.items():
            numeric_value = self._to_float_or_none(value)
            if numeric_value is None:
                continue
            if not math.isfinite(numeric_value):
                continue
            parsed_scores[str(key)] = numeric_value
        return parsed_scores

    def get_action(
        self,
        state,
        feasible_actions: list,
        strategic_experience: str,
        session_folder: str,
        iteration: int,
    ) -> dict:
        prompt_inputs = state.compile_prompt_elements()
        prompt_inputs["strategic_experience"] = self._extract_key_insights_text(strategic_experience)

        prompt_text = self.action_prompt.replace("{snapshot['timestamp']}", str(prompt_inputs["timestamp"]))
        prompt_text = prompt_text.replace("{Machines States}", prompt_inputs["machines_states"])
        prompt_text = prompt_text.replace("{Emergency Jobs}", prompt_inputs["emergency_jobs"])
        prompt_text = prompt_text.replace("{Strategic Experience}", prompt_inputs["strategic_experience"])
        prompt_text = prompt_text.replace("{Ready Operations}", prompt_inputs["ready_operations"])
        prompt_text = prompt_text.replace("{Full State Information}", prompt_inputs["full_state"])
        prompt_text = prompt_text.replace("{actions_json}", prompt_inputs["actions_json"])

        llm_output = self._call_api(
            prompt_text=prompt_text,
            model_name=str(getattr(config, "PRIOR_MODEL_NAME", "")),
            temperature=float(getattr(config, "PRIOR_LLM_TEMPERATURE", 0.3)),
            prefix="PRIOR",
            session_folder=session_folder,
            iteration=iteration,
            call_type="Action_Policy",
        )

        match = re.search(r"\{.*?\}", llm_output, re.DOTALL)
        if match:
            try:
                decision = json.loads(match.group(0))
                for action in feasible_actions:
                    if (
                        action["job"] == decision.get("job")
                        and action["op"] == decision.get("op")
                        and action["machine"] == decision.get("machine")
                    ):
                        return decision
            except json.JSONDecodeError:
                pass
        return None

    def get_priors(
        self,
        state,
        feasible_actions: list,
        strategic_experience: str,
        session_folder: str,
        iteration: int,
    ) -> dict:
        if not feasible_actions:
            return {}

        prompt_inputs = state.compile_prompt_elements()
        lessons_text, _ = self._load_lessons_text(session_folder)
        indexed_actions = []
        for action_index, action in enumerate(feasible_actions):
            action_copy = dict(action)
            job_id = action_copy.get("job")
            due_date = self._to_float_or_none(action_copy.get("due_date"))
            remaining_work = None
            if hasattr(state, "_calculate_rem_work") and job_id is not None:
                try:
                    remaining_work = float(state._calculate_rem_work(job_id))
                except Exception:
                    remaining_work = None

            if "slack" not in action_copy:
                if due_date is not None and remaining_work is not None:
                    action_copy["slack"] = due_date - float(getattr(state, "current_time", 0.0)) - remaining_work
                else:
                    action_copy["slack"] = None

            if "is_critical" not in action_copy:
                if remaining_work is not None:
                    all_remaining_work = []
                    for candidate_action in feasible_actions:
                        candidate_job_id = candidate_action.get("job")
                        if candidate_job_id is None or not hasattr(state, "_calculate_rem_work"):
                            continue
                        try:
                            all_remaining_work.append(float(state._calculate_rem_work(candidate_job_id)))
                        except Exception:
                            continue
                    max_remaining_work = max(all_remaining_work) if all_remaining_work else None
                    action_copy["is_critical"] = bool(
                        max_remaining_work is not None
                        and remaining_work == max_remaining_work
                        and remaining_work > 0.0
                    )
                else:
                    action_copy["is_critical"] = False

            indexed_actions.append({"index": str(action_index), **action_copy})

        prompt_inputs["actions_json"] = dumps_capped(indexed_actions, indent=2)
        prompt_inputs["strategic_experience"] = self._extract_key_insights_text(strategic_experience)

        prompt_text = self.prior_prompt.replace("{snapshot['timestamp']}", str(prompt_inputs["timestamp"]))
        prompt_text = prompt_text.replace("{Machines States}", prompt_inputs["machines_states"])
        prompt_text = prompt_text.replace("{Emergency Jobs}", prompt_inputs["emergency_jobs"])
        prompt_text = prompt_text.replace("{Ready Operations}", prompt_inputs["ready_operations"])
        prompt_text = prompt_text.replace("{Full State Information}", prompt_inputs["full_state"])
        prompt_text = prompt_text.replace("{Strategic Experience}", prompt_inputs["strategic_experience"])
        prompt_text = prompt_text.replace("{actions_json}", prompt_inputs["actions_json"])
        prompt_text = prompt_text.replace("{lessons_text}", lessons_text)

        llm_output = self._call_api(
            prompt_text=prompt_text,
            model_name=str(getattr(config, "PRIOR_MODEL_NAME", "")),
            temperature=float(getattr(config, "PRIOR_LLM_TEMPERATURE", 0.3)),
            prefix="PRIOR",
            session_folder=session_folder,
            iteration=iteration,
            call_type="Prior_Probabilities",
        )

        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        raw_scores = {}
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    raw_scores = self._parse_operation_scores(data)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return self._normalize_scores(raw_scores, len(feasible_actions))

    def get_explore_decision(self, state, strategic_experience: str, session_folder: str, iteration: int) -> bool:
        if not self.explore_prompt:
            return False

        prompt_inputs = state.compile_prompt_elements()
        prompt_inputs["strategic_experience"] = strategic_experience

        prompt_text = self.explore_prompt.replace("{snapshot['timestamp']}", str(prompt_inputs["timestamp"]))
        prompt_text = prompt_text.replace("{Machines States}", prompt_inputs["machines_states"])
        prompt_text = prompt_text.replace("{Ready Operations}", prompt_inputs["ready_operations"])
        prompt_text = prompt_text.replace("{Full State Information}", prompt_inputs["full_state"])
        prompt_text = prompt_text.replace("{Strategic Experience}", prompt_inputs["strategic_experience"])
        lower_bound = prompt_inputs.get("lower_bound", str(prompt_inputs["timestamp"]))
        prompt_text = prompt_text.replace("{snapshot['lower_bound']}", lower_bound)

        llm_output = self._call_api(
            prompt_text=prompt_text,
            model_name=str(getattr(config, "PRIOR_MODEL_NAME", "")),
            temperature=float(getattr(config, "PRIOR_LLM_TEMPERATURE", 0.3)),
            prefix="PRIOR",
            session_folder=session_folder,
            iteration=iteration,
            call_type="Explore_Decision",
        )
        match = re.search(r"<decision>(EXPLORE|CONTINUE)</decision>", llm_output, re.IGNORECASE)
        if match:
            decision = match.group(1).upper()
            return decision == "EXPLORE"
        return False