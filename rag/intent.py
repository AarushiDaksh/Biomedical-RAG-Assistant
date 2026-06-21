import json
from typing import List

VALID_INTENTS = {"summary", "qa", "compare", "smalltalk"}

CLASSIFY_SYSTEM = """You are an intent router for a biomedical paper assistant.
Classify the user's latest message into exactly one intent:
- "summary": wants an overview/summary/key findings/limitations/methods of a paper.
- "qa": asks a specific question answerable from a passage.
- "compare": wants two or more papers compared/contrasted.
- "smalltalk": greeting/thanks/chit-chat, not about paper content.
Use the conversation to resolve pronouns like "it" or "this paper".
Return ONLY JSON: {"intent": "summary|qa|compare|smalltalk"}"""


def _format_history(history: List[dict], limit: int = 6) -> str:
    turns = history[-limit:] if history else []
    return "\n".join(f"{t.get('role', '?')}: {t.get('content', '')}" for t in turns) or "(none)"


def _extract_intent(raw: str) -> str:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            label = json.loads(raw[start:end]).get("intent", "")
            if label in VALID_INTENTS:
                return label
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    return "qa"


async def classify_intent(question: str, history: List[dict], llm) -> str:
    user = f"Conversation so far:\n{_format_history(history)}\n\nLatest message:\n{question}"
    raw = (await llm.ainvoke([("system", CLASSIFY_SYSTEM), ("user", user)])).content
    return _extract_intent(raw)
