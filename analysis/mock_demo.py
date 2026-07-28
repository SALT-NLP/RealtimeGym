#!/usr/bin/env python
"""
Mechanism demonstration for Gap-3 (planning-thread refresh) — NO API KEY NEEDED.

A deterministic *scripted* model stands in for the LLM and drives the **real**
Freeway environment through AgileThinker's actual code path (`think` →
`planning_inference` → `reactive_inference`). Because the scripted planning
"generation" spans several environment steps (like a real long reasoning trace
revealed under a token budget), we can measure how stale the plan the agent is
acting on becomes — `plan_age` = env steps since that plan was grounded.

This demonstrates that the mechanism does what it claims:
  * vanilla AgileThinker acts on plans that grow arbitrarily stale;
  * refresh caps plan age by re-grounding on the current observation.

IMPORTANT: this is a MECHANISM result, not a task-performance result. The
scripted model does not play well (it just stays put); scores are meaningless
here. Whether refresh *improves reward* requires the real LLM A/B (needs keys).
"""
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openai import OpenAI

import realtimegym
import realtimegym.prompts.freeway as fw_prompts
from realtimegym.agents.agile import AgileThinker
from realtimegym.agents.base import BaseAgent

PALETTE = {"vanilla": "#000000", "refresh(every=3)": "#E69F00"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_figures")

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.grid": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888",
})


class _WordTok:
    """Trivial tokenizer so token-mode partial-reveal logic works offline."""
    def encode(self, text):
        return text.split()

    def decode(self, toks, skip_special_tokens=True):
        return " ".join(toks)


class MockAgileThinker(AgileThinker):
    """AgileThinker with the two LLM calls replaced by a scripted model.

    - planning call returns a fixed "plan" but reports a large token count, so
      in token mode the plan takes ~plan_tokens/budget env steps to complete —
      exactly the regime where a plan goes stale.
    - reactive call returns a fixed boxed action (so no network budget-forcing).
    """
    def __init__(self, prompts, file, internal_budget=2048, plan_tokens=40000):
        BaseAgent.__init__(self, prompts, file, time_unit="token")
        self.model1, self.model2 = "mock-reactive", "mock-planning"
        self.model1_config = {"inference_parameters": {}}
        self.model2_config = {"inference_parameters": {}}
        self.internal_budget = internal_budget
        self.tokenizer = _WordTok()
        # placeholder clients only satisfy isinstance() checks; generate() is
        # overridden so no network call is ever made.
        self.llm1 = self.llm2 = OpenAI(api_key="mock-not-used", base_url="http://localhost")
        self._plan_tokens = plan_tokens

    def generate(self, llm, model, messages, sampling_params):  # noqa: ANN001
        if model == self.model2:  # planning "generation" (long, multi-step)
            return ("Plan: assess lanes, then advance. \\boxed{S}", self._plan_tokens)
        return ("Hold position this step. \\boxed{S}", 300)  # reactive


def run_episode(refresh_kwargs, seed, max_steps=100, time_pressure=4096):
    env, _, _ = realtimegym.make("Freeway-v1", seed=seed)  # v1 = medium load
    tmp = os.path.join(tempfile.gettempdir(), f"mock_{seed}.csv")
    agent = MockAgileThinker(fw_prompts, file=tmp, internal_budget=2048)
    agent.configure_refresh(**refresh_kwargs)
    obs, done = env.reset()
    steps = 0
    while not done and steps < max_steps:
        agent.observe(obs)
        agent.think(timeout=time_pressure)
        action = agent.act()
        obs, done, reward, reset = env.step(action)
        agent.log(reward, reset)  # faithful: also drains plan on episode reset
        steps += 1
    ages = [int(a) for a in agent.logs.get("plan_age", []) if a is not None]
    return ages, agent.refresh_count


def main():
    os.makedirs(OUT, exist_ok=True)
    policies = {
        "vanilla": dict(refresh_plan=False),
        "refresh(every=3)": dict(refresh_plan=True, refresh_every=3),
    }
    seeds = range(6)
    results = {}
    for name, kw in policies.items():
        all_ages, total_refresh = [], 0
        for s in seeds:
            ages, nref = run_episode(kw, seed=s)
            all_ages += ages
            total_refresh += nref
        results[name] = dict(ages=np.array(all_ages), refreshes=total_refresh)

    # ---- summary table ----
    print("\n=== MECHANISM DEMO (scripted model — NOT task performance) ===")
    print(f"{'policy':18} {'decisions':>9} {'median age':>11} {'max age':>8} "
          f"{'% age>=3':>9} {'refreshes':>10}")
    for name, r in results.items():
        a = r["ages"]
        if len(a) == 0:
            continue
        print(f"{name:18} {len(a):>9d} {np.median(a):>11.1f} {a.max():>8d} "
              f"{100 * np.mean(a >= 3):>8.1f}% {r['refreshes']:>10d}")

    # ---- figure: plan-age CDF ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, r in results.items():
        a = np.sort(r["ages"])
        if len(a) == 0:
            continue
        ys = np.arange(1, len(a) + 1) / len(a)
        ax.plot(a, ys, "-", color=PALETTE[name], lw=2.2, label=name)
    ax.set_xlabel("plan age at decision time (env steps since the acting plan was grounded)")
    ax.set_ylabel("cumulative fraction of decisions")
    ax.set_ylim(0, 1)
    ax.set_title("Mechanism check: refresh caps how stale the acting plan gets\n"
                 "(scripted model, real Freeway env — staleness, not task score)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = os.path.join(OUT, "mock_plan_age_cdf.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {path}")


if __name__ == "__main__":
    main()
