"""The one place the agent's token budget is turned into words.

Imported by both the harness (agents/reactive.py) and the data builder
(analysis/build_sft.py) so the prompts a model is trained on and the prompts it
is served are byte-identical. A model conditioned on one wording and served
another is conditioned on nothing.

Deliberately free of heavy imports: build_sft.py runs on a laptop with no
openai/torch installed, and agents/base.py pulls both.
"""


def budget_line(n: int) -> str:
    return f"\n\nYou have about {n} tokens to answer. Spend them accordingly."
