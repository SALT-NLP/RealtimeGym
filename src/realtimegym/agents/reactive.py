import os
import re
from typing import Any, Optional

from .base import BaseAgent, budget_line, extract_boxed


class ReactiveAgent(BaseAgent):
    def __init__(
        self,
        prompts: Any,  # noqa: ANN401 - prompts is a dynamically loaded module
        file: str,
        time_unit: str,
        model1_config: str,
        internal_budget: int,
    ) -> None:
        super().__init__(prompts, file, time_unit)
        self.config_model1(model1_config, internal_budget)

    def truncate_logs(self) -> None:
        return

    def think(self, timeout: Optional[float] = None) -> None:
        assert self.current_observation is not None and timeout is not None
        budget = timeout
        observation = self.current_observation
        prompt_gen = self.prompts.state_to_description(
            observation["state"], mode="reactive"
        )
        # The agent is otherwise never told its budget -- internal_budget is
        # applied as an API max_tokens cap only -- so it cannot allocate its
        # thinking. Opt in with RTG_ANNOUNCE_BUDGET=1.
        if os.environ.get("RTG_ANNOUNCE_BUDGET") == "1":
            prompt_gen += budget_line(self.internal_budget)
        messages = [{"role": "user", "content": prompt_gen}]
        text, token_num = self.reactive_inference(messages, budget)
        self.action = re.sub(
            r"[^" + self.prompts.ALL_ACTIONS + "]", "", extract_boxed(text)
        )
        if self.action == "":
            self.action = self.prompts.DEFAULT_ACTION
        if self.log_thinking:
            self.logs["plan"].append("N/A")
            self.logs["model1_prompt"].append(messages[-1]["content"])
            self.logs["model1_response"].append(text)
        self.logs["model1_token_num"].append(token_num)
