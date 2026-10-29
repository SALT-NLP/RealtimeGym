import os
import queue
import re
import threading
import time
from collections import defaultdict
from typing import Any, NoReturn, Optional, Union

import pandas as pd
import yaml
from openai import OpenAI

from ..budget import budget_line  # noqa: F401  (re-exported for agents)
from transformers import AutoTokenizer

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables only


class BaseAgent:
    def __init__(
        self,
        prompts: Any,  # noqa: ANN401 - prompts is a dynamically loaded module
        file: str,
        time_unit: str,
    ) -> None:
        self.prompts = prompts
        self.file = file
        self.time_unit = time_unit
        self.model1 = None
        self.model2 = None
        self.tokenizer = None
        self.llm1 = None
        self.llm2 = None
        self.internal_budget = 0

        self.logs = defaultdict(list)
        # better not set log_thinking to True for time-based budget, since storing logs can be slow and interfere with timing
        self.log_thinking = True if time_unit == "token" else False
        self.action = prompts.DEFAULT_ACTION
        self.to_flush = ""
        self.to_flush_turn = 0
        self.gen_turn = 0
        self.gen_accum = 0
        self.gen_text = ""
        self.gen_token = []
        self.gen_token_num = 0
        self.planning_queue = None
        self.planning_done = None
        self.planning_reasoning = False
        self.plan = ""
        self.state_string = ""

        # New API: store current observation
        self.current_observation = None

    def _resolve_env_var(
        self, value: Union[str, int, float, bool, None]
    ) -> Union[str, int, float, bool, None]:
        """
        Resolve environment variable references in config values.
        Supports ${VAR_NAME} syntax.

        Args:
            value: String that may contain ${VAR_NAME} references

        Returns:
            Resolved string with environment variables substituted
        """
        if not isinstance(value, str):
            return value

        # Match ${VAR_NAME} pattern
        pattern = r"\$\{([^}]+)\}"

        def replace_env_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' not found. "
                    f"Please set it in your .env file or environment."
                )
            return env_value

        return re.sub(pattern, replace_env_var, value)

    def config_model1(self, model1_config: str, internal_budget: int) -> None:
        with open(model1_config, "r") as f:
            self.model1_config = yaml.safe_load(f)

        # Resolve environment variables in api_key
        api_key = self._resolve_env_var(self.model1_config["api_key"])
        assert isinstance(api_key, str), "API key must be a string after resolution."

        self.llm1 = OpenAI(
            api_key=api_key,
            base_url=self.model1_config.get("url"),
        )
        self.model1 = self.model1_config["model"]
        self.internal_budget = internal_budget

    def config_model2(self, model2_config: str) -> None:
        with open(model2_config, "r") as f:
            self.model2_config = yaml.safe_load(f)

        # Resolve environment variables in api_key
        api_key = self._resolve_env_var(self.model2_config["api_key"])
        assert isinstance(api_key, str), "API key must be a string after resolution."

        self.llm2 = OpenAI(
            api_key=api_key,
            base_url=self.model2_config.get("url"),
        )
        self.model2 = self.model2_config["model"]
        if "tokenizer" in self.model2_config:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model2_config["tokenizer"]
            )

    def observe(self, observation: dict[str, Any]) -> None:
        """
        Receive and store the current observation from the environment.

        Args:
            observation (dict): Observation containing state information
        """
        self.current_observation = observation

    def think(self, timeout: Optional[float] = None) -> NoReturn:
        """
        Process the current observation and decide on an action.

        Args:
            timeout: Time/token budget for thinking (optional, uses internal_budget if not provided)
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def act(self) -> Optional[str]:
        """
        Return the chosen action. Returns None if no action was decided.

        Returns:
            action (str or None): The action to take, or None for default action
        """
        return self.action

    def log(self, reward: float, reset: bool) -> None:
        assert self.current_observation is not None, "Current observation is not set!"
        self.logs["render"].append(self.current_observation["state_string"])
        self.logs["action"].append(self.action)
        self.logs["reward"].append(reward)
        if reset:
            self.plan = ""
            if self.time_unit == "token":
                while self.gen_text != "":
                    self.planning_inference([], 80000, 0)
            else:
                while self.gen_text != "":
                    self.planning_inference([], 10.0 + self.internal_budget, 0)
            self.to_flush = ""
        df = pd.DataFrame(self.logs)
        df.to_csv(self.file)

    def resume_from_checkpoint(self, env: Any, checkpoint_file: str) -> None:  # noqa: ANN401
        df = pd.read_csv(checkpoint_file)
        self.logs = df.to_dict("list")  # remove unnamed column
        self.logs.pop("Unnamed: 0", None)
        self.truncate_logs()
        for a in self.logs["action"]:
            env.act(a)
        df = pd.DataFrame(self.logs)
        df.to_csv(self.file)

    def truncate_logs(self) -> NoReturn:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def generate(
        self,
        llm: OpenAI,
        model: str,
        messages: list[dict[str, str]],
        sampling_params: dict[str, Any],
    ) -> tuple[str, int]:
        params = sampling_params
        params["messages"] = messages
        params["model"] = model
        while True:
            try:
                text = ""
                response = llm.chat.completions.create(**params)
                if (
                    hasattr(response.choices[0].message, "reasoning_content")
                    and response.choices[0].message.reasoning_content is not None
                ):
                    text = (
                        "<think>"
                        + response.choices[0].message.reasoning_content
                        + "\n</think>\n"
                    )
                if response.choices[0].message.content is not None:
                    text += response.choices[0].message.content
                if (
                    "gemini" in model
                ):  ### GEMINI EXCEPTION: completion_tokens do not include thinking tokens
                    token_num = (
                        response.usage.total_tokens - response.usage.prompt_tokens
                    )
                else:
                    token_num = response.usage.completion_tokens
                return text, token_num
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

    def start_planning_stream(
        self,
        llm: OpenAI,
        model: str,
        messages: list[dict[str, str]],
        sampling_params: dict[str, Any],
    ) -> None:
        self.planning_queue = queue.Queue()
        self.planning_done = threading.Event()
        params = sampling_params
        params["messages"] = messages
        params["model"] = model
        params["stream"] = True

        def planning_worker() -> None:
            assert self.planning_queue is not None, "Planning queue is not initialized!"
            assert self.planning_done is not None, (
                "Planning done event is not initialized!"
            )
            try:
                stream_obj = llm.chat.completions.create(**params)
                for chunk in stream_obj:
                    self.planning_queue.put(chunk)
                self.planning_done.set()
            except Exception as e:
                print(f"Streaming error: {e}")
                self.planning_done.set()

        threading.Thread(target=planning_worker, daemon=True).start()

    def get_planning_chunks(self) -> tuple[str, int]:
        text = ""
        token_num = 0
        assert self.planning_queue is not None, "Planning queue is not initialized!"
        while not self.planning_queue.empty():
            chunk = self.planning_queue.get()
            if (
                hasattr(chunk.choices[0].delta, "reasoning_content")
                and chunk.choices[0].delta.reasoning_content is not None
            ):
                if not self.planning_reasoning:
                    text += "<think>"
                    self.planning_reasoning = True
                text += chunk.choices[0].delta.reasoning_content
            if (
                hasattr(chunk.choices[0].delta, "content")
                and chunk.choices[0].delta.content is not None
            ):
                if self.planning_reasoning:
                    text += "\n</think>\n"
                    self.planning_reasoning = False
                text += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage is not None:
                token_num = chunk.usage.completion_tokens
        return text, token_num

    def is_planning_finished(self) -> bool:
        assert self.planning_done is not None, "Planning done event is not initialized!"
        return self.planning_done.is_set()

    def start_reactive_stream(
        self,
        llm: OpenAI,
        model: str,
        messages: list[dict[str, str]],
        sampling_params: dict[str, Any],
        max_time: float,
    ) -> tuple[str, int]:
        params = sampling_params
        params["messages"] = messages
        params["model"] = model
        params["stream"] = True
        start_time = time.time()
        stream_obj = llm.chat.completions.create(**params)
        text, token_num = "", 0
        for chunk in stream_obj:
            if time.time() - start_time > max_time:
                break
            if (
                hasattr(chunk.choices[0].delta, "content")
                and chunk.choices[0].delta.content is not None
            ):
                text += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage is not None:
                token_num = chunk.usage.completion_tokens
        current_time = time.time()
        if max_time - (current_time - start_time) > 0:
            time.sleep(max_time - (current_time - start_time))
        return text, token_num

    def reactive_inference(
        self, messages: list[dict[str, str]], budget: float
    ) -> tuple[str, int]:
        assert self.model1 is not None, "Reactive LLM is not initialized!"
        sampling_params = self.model1_config.get("inference_parameters", {})
        assert isinstance(self.llm1, OpenAI), "LLM1 is not an instance of OpenAI!"
        if self.time_unit == "token":
            if "max_completion_tokens" in sampling_params:
                sampling_params["max_completion_tokens"] = min(
                    self.internal_budget,
                    sampling_params["max_completion_tokens"],
                )
            else:
                sampling_params["max_tokens"] = min(
                    self.internal_budget,
                    sampling_params.get("max_tokens", 80000),
                )
            text, token_num = self.generate(
                self.llm1, self.model1, messages, sampling_params
            )
        else:
            text, token_num = self.start_reactive_stream(
                self.llm1,
                self.model1,
                messages,
                sampling_params,
                self.internal_budget,
            )
            time.sleep(budget - self.internal_budget)
        if "<think>" in text and "</think>" not in text:
            text += "</think>"
        if "oxed" in text.split("</think>")[-1]:
            return text, token_num
        ### s1 budget forcing ###
        text += "\nTherefore, the final answer is \\boxed{"
        max_attempt = 3
        while max_attempt > 0:
            max_attempt -= 1
            new_params = {
                "model": self.model1,
                "messages": messages + [{"role": "assistant", "content": text}],
                "temperature": 0,
                "top_p": 1,
            }
            if "max_completion_tokens" in sampling_params:
                new_params["max_completion_tokens"] = 1
            else:
                new_params["max_tokens"] = 1
            try:
                response = self.llm1.chat.completions.create(**new_params)
                if (
                    response.choices[0].message.content.strip()[0]
                    in self.prompts.ALL_ACTIONS
                ):
                    text += response.choices[0].message.content.strip()[0] + "}"
                    break
            except Exception:
                time.sleep(0.2)
            if max_attempt == 0:
                text += self.prompts.DEFAULT_ACTION + "}"
        return text, token_num

    def planning_inference(
        self, messages: list[dict[str, str]], budget: float, game_turn: int
    ) -> tuple[str, int, int]:
        assert self.model2 is not None, "Planning LLM is not initialized!"
        token_num = 0
        assert isinstance(self.llm2, OpenAI), "LLM2 is not an instance of OpenAI!"
        sampling_params = self.model2_config.get("inference_parameters", {})
        if self.time_unit == "token":
            if messages != []:
                self.gen_turn = game_turn
                self.gen_accum = -self.internal_budget
                self.gen_text, self.gen_token_num = self.generate(
                    self.llm2, self.model2, messages, sampling_params
                )
                if self.tokenizer is not None:
                    self.gen_token = self.tokenizer.encode(self.gen_text)
                token_num = self.gen_token_num
            self.gen_accum += budget
            can_flush = (
                self.gen_accum >= self.gen_token_num or self.tokenizer is not None
            )
            if can_flush:
                self.to_flush_turn = self.gen_turn
                if self.gen_accum >= self.gen_token_num:
                    self.to_flush = self.gen_text
                elif self.tokenizer is not None:
                    self.to_flush = self.tokenizer.decode(
                        self.gen_token[: self.gen_accum],
                        skip_special_tokens=True,
                    )
            text = self.to_flush
            turn = self.to_flush_turn
            self.to_flush = ""
            # check for completion: set the system in idle state
            if self.gen_accum + self.internal_budget >= self.gen_token_num:
                self.gen_text = ""
        else:
            if messages != []:
                self.gen_turn = game_turn
                self.start_planning_stream(
                    self.llm2, self.model2, messages, sampling_params
                )
            time.sleep(budget - self.internal_budget)
            new_text, token_num = self.get_planning_chunks()
            self.gen_text += new_text
            text = self.gen_text
            turn = self.gen_turn
            if self.is_planning_finished():
                self.gen_text = ""
        return text, token_num, turn


def extract_boxed(text: str, default_value: str = "") -> str:
    """
    Extracts the \boxed{...} text from the input string.
    """
    pattern = r"oxed{"
    start_index = text.rfind(pattern)
    if start_index == -1:
        # Try to extract content enclosed in triple backticks if \boxed{...} is not found
        triple_backtick_pattern = r"```(.*?)```"
        matches = re.findall(triple_backtick_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return default_value
    start_index += len(pattern) - 1
    stack = []
    for i in range(start_index, len(text)):
        if text[i] == "{":
            stack.append("{")
        elif text[i] == "}":
            if stack:
                stack.pop()
            if not stack:
                return text[start_index + 1 : i].strip()
    return default_value if default_value else text[start_index + 1 :].strip()
