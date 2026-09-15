"""Shared train/eval protocol for structured multi-image CoT supervision."""

from __future__ import annotations

import re


SYSTEM_PROMPT = """You are a multi-image visual reasoning assistant.
Inspect the provided images and the question carefully. Use only visual facts that are relevant and grounded in the images.
First write a short evidence-based reasoning trace, then give the final answer.

Always use exactly this output structure:
<evidence>
E1 | support=[image indices] | concise observable fact
</evidence>
<synthesis>
one short sentence that connects the evidence to the answer
</synthesis>
<answer>
FINAL_ANSWER
</answer>

Rules:
- Use at most two evidence lines and the fewest supporting images needed.
- Do not enumerate irrelevant images or invent visual facts.
- The synthesis may not introduce facts absent from the evidence.
- Put only the exact required answer in <answer>: an option letter for multiple-choice questions, or the requested class/value for direct-answer questions.
- Even if the user asks for only a letter or answer, keep the required structure and put that answer inside <answer>.
- Do not put any final answer outside <answer>.
"""


def add_system(messages: list[dict]) -> list[dict]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def answer_payload(raw: str) -> str:
    """Return only the final explicit answer block.

    A truncated evidence/synthesis section must not be treated as an answer;
    missing ``<answer>...</answer>`` is intentionally an unparsed prediction.
    """
    blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw or "", flags=re.I | re.S)
    if blocks:
        return blocks[-1].strip()
    return ""
