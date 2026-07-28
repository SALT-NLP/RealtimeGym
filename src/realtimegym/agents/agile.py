import re
from typing import Any, Optional

from .base import BaseAgent, extract_boxed


class AgileThinker(BaseAgent):
    def __init__(
        self,
        prompts: Any,  # noqa: ANN401 - prompts is a dynamically loaded module
        file: str,
        time_unit: str,
        model1_config: str,
        model2_config: str,
        internal_budget: int,
    ) -> None:
        super().__init__(prompts, file, time_unit)
        self.config_model1(model1_config, internal_budget)
        self.config_model2(model2_config)
        if self.time_unit == "token":
            assert self.tokenizer is not None, (
                "Tokenizer must be initialized for agile agent under token-based time unit."
            )

    def truncate_logs(self) -> None:
        final_step = 0
        for i in range(len(self.logs["action"])):
            if "model2_prompt" in self.logs and self.logs["model2_prompt"][i] != "":
                final_step = i
        for col in self.logs:
            self.logs[col] = self.logs[col][:final_step]

    def think(self, timeout: Optional[float] = None) -> None:
        assert self.current_observation is not None and timeout is not None
        budget = timeout
        observation = self.current_observation
        self.state_string = observation["state_string"]
        game_turn = observation["game_turn"]
        prompt_gen = self.prompts.state_to_description(
            observation["state"], mode="agile"
        )
        prompt = ""
        # Gap 3: start a fresh planning generation either when the previous one
        # has finished (gen_text == "") or when the live plan has gone stale and
        # should be re-grounded on the current observation.
        refresh = self.should_refresh_plan(observation)
        prior_reasoning = self.plan_so_far
        if self.gen_text == "" or refresh:
            plan_prompt = prompt_gen["planning"]
            if refresh and self.refresh_carryover and prior_reasoning:
                plan_prompt += (
                    "\n\n**Your earlier reasoning (computed on a now-outdated state; "
                    "reuse what is still valid and revise the rest):**\n"
                    + prior_reasoning
                )
            messages = [{"role": "user", "content": plan_prompt}]
            prompt = plan_prompt
            self.plan_anchor_state_string = self.state_string
            if refresh:
                self.refresh_count += 1
        else:
            messages = []
        text, token_num, turn = self.planning_inference(messages, budget, game_turn)
        # Track reasoning revealed for the current generation (cumulative prefix
        # in token mode); carried into the new prompt on a refresh when enabled.
        self.plan_so_far = text
        self.plan = f"""**Guidance from a Previous Thinking Model:** Turn \\( t_1 = {turn} \\)\n{text}"""
        if self.log_thinking:
            self.logs["plan"].append(self.plan)
            self.logs["model2_prompt"].append(prompt)
            self.logs["model2_response"].append(text)
            self.logs["refreshed"].append(int(refresh))
            self.logs["plan_age"].append(game_turn - turn)
        self.logs["model2_token_num"].append(token_num)

        prompt = prompt_gen["reactive"]
        if self.plan is not None:
            lines = self.plan.split("\n")
            for line in lines:
                prompt += f"> {line.strip()}\n"
        messages = [{"role": "user", "content": prompt}]
        text, token_num = self.reactive_inference(messages, self.internal_budget)
        self.action = re.sub(
            r"[^" + self.prompts.ALL_ACTIONS + "]", "", extract_boxed(text)
        )
        if self.log_thinking:
            self.logs["model1_prompt"].append(prompt)
            self.logs["model1_response"].append(text)
        self.logs["model1_token_num"].append(token_num)
