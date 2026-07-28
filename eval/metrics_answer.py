"""Answer-quality metrics via LLM-as-judge (RAGAS-style, implemented directly).

Two scores per question, one judge call:
- faithfulness (0-1): fraction of claims in the answer supported by the
  provided context passages. Low = hallucination.
- correctness (0-1): does the answer actually answer the question, consistent
  with the expected answer hint from the golden set?

Judge returns strict JSON. We use the judge model (separate from the agent
model) so eval batches can run on a different quota/model tier.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import APIError, InternalServerError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.factory import get_chat_client, get_judge_model_name

_RETRY = (RateLimitError, InternalServerError, APIError)

JUDGE_SYSTEM = """You are a strict evaluation judge for a RAG (retrieval-augmented generation) system on cybersecurity documentation.

You will receive: a QUESTION, the CONTEXT passages the system retrieved, the system's ANSWER, and an EXPECTED_HINT (a short phrase the correct answer should reflect).

Score two things:

1. faithfulness (0.0-1.0): What fraction of the factual claims in ANSWER are directly supported by CONTEXT? Ignore stylistic text. 1.0 = every claim grounded; 0.0 = fully fabricated. If the answer correctly says it lacks information, score 1.0.

2. correctness (0.0-1.0): Does ANSWER correctly and substantively answer QUESTION, consistent with EXPECTED_HINT? The hint is a signal, not the full answer - a correct answer may phrase things differently. 1.0 = fully correct; 0.5 = partially; 0.0 = wrong or non-answer.

Respond with ONLY this JSON, no markdown fences, no commentary:
{"faithfulness": <float>, "correctness": <float>, "reasoning": "<one short sentence>"}"""

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

EXPECTED_HINT:
{hint}"""


@dataclass
class AnswerScores:
    faithfulness: float
    correctness: float
    reasoning: str


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_RETRY),
)
def _judge_call(question: str, context: str, answer: str, hint: str) -> str:
    resp = get_chat_client().chat.completions.create(
        model=get_judge_model_name(),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                question=question, context=context, answer=answer, hint=hint,
            )},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_judge_json(raw: str) -> AnswerScores:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        d: dict[str, Any] = json.loads(cleaned)
        return AnswerScores(
            faithfulness=max(0.0, min(1.0, float(d.get("faithfulness", 0.0)))),
            correctness=max(0.0, min(1.0, float(d.get("correctness", 0.0)))),
            reasoning=str(d.get("reasoning", ""))[:300],
        )
    except Exception:
        f = re.search(r'"faithfulness"\s*:\s*([0-9.]+)', raw)
        c = re.search(r'"correctness"\s*:\s*([0-9.]+)', raw)
        if f and c:
            return AnswerScores(
                faithfulness=max(0.0, min(1.0, float(f.group(1)))),
                correctness=max(0.0, min(1.0, float(c.group(1)))),
                reasoning="(recovered from truncated JSON)",
            )
        return AnswerScores(0.0, 0.0, f"judge output unparseable: {raw[:120]}")


def judge_answer(question: str, context: str, answer: str, hint: str) -> AnswerScores:
    """Score one (question, context, answer) triple. One LLM call."""
    raw = _judge_call(question, context, answer, hint)
    return _parse_judge_json(raw)