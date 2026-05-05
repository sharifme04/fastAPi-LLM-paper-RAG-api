"""Answer generation: feed retrieved chunks to Claude, parse the JSON answer.

We force a single non-streaming JSON call so structured outputs are easy
to validate. Streaming can be layered on later via a separate endpoint.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from app.config import get_settings
from app.services.retriever import Retrieved
from app.utils.exceptions import GenerationError

logger = logging.getLogger("paper_rag")
settings = get_settings()

SYSTEM_PROMPT = """You are a careful research assistant. You will be given a user question and a numbered list of context passages drawn from scientific papers. Your job:

1. Answer the question using ONLY the provided context. If the context does not contain the answer, say so explicitly.
2. Cite the passages you use by their numbers in square brackets, e.g. [1], [2].
3. Do not invent citations or facts. Do not hallucinate page numbers.
4. Keep the answer concise — a few sentences to a short paragraph unless the question demands more.

Return your response as JSON with this exact shape:
{
  "answer": "string — the natural-language answer with [n] citations inline",
  "used_passages": [int, int, ...]   // 1-based passage numbers actually used
}

Output ONLY the JSON object, no markdown, no code fences, no preamble.
"""


def _build_user_message(question: str, retrieved: list[Retrieved]) -> str:
    parts = [f"Question: {question}", "", "Context passages:"]
    for i, r in enumerate(retrieved, start=1):
        loc = ""
        if r.page_start is not None:
            loc = f" (p.{r.page_start}{'' if r.page_end == r.page_start else f'-{r.page_end}'})"
        parts.append(f"[{i}] from {r.document_filename}{loc}:\n{r.text}\n")
    return "\n".join(parts)


async def generate_answer(
    question: str,
    retrieved: list[Retrieved],
) -> dict[str, Any]:
    """Call Claude with the retrieved context and return parsed structured answer.

    Returns a dict with keys: answer (str), used_passages (list[int]),
    input_tokens (int), output_tokens (int).
    """
    if not retrieved:
        return {
            "answer": "I cannot answer — no relevant context was retrieved.",
            "used_passages": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    user_msg = _build_user_message(question, retrieved)

    try:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except anthropic.APIError as e:
        logger.error("Anthropic error during generation: %s", e)
        raise GenerationError(
            f"Claude API call failed: {getattr(e, 'message', str(e))}",
            detail={"api_error": str(e)},
        )

    raw = "".join(b.text for b in response.content if getattr(b, "type", "text") == "text").strip()

    # Strip accidental fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        # remove leading "json\n" if any
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON answer: %s", raw[:200])
        raise GenerationError(
            "Claude did not return valid JSON.",
            detail={"raw_excerpt": raw[:200], "error": str(e)},
        )

    if "answer" not in parsed:
        raise GenerationError(
            "Claude response missing 'answer' field.",
            detail={"keys": list(parsed.keys())},
        )

    answer = str(parsed["answer"])
    used = parsed.get("used_passages") or []
    if not isinstance(used, list):
        used = []
    used = [int(x) for x in used if isinstance(x, (int, float))]

    return {
        "answer": answer,
        "used_passages": used,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
