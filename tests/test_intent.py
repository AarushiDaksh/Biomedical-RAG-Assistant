import asyncio
from unittest.mock import MagicMock
from rag.intent import classify_intent, VALID_INTENTS


class FakeLLM:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, messages):
        return MagicMock(content=self._content)


def _run(coro):
    return asyncio.run(coro)


def test_classifies_summary():
    llm = FakeLLM('{"intent": "summary"}')
    assert _run(classify_intent("give me a summary of it", [], llm)) == "summary"


def test_classifies_compare():
    llm = FakeLLM('{"intent": "compare"}')
    assert _run(classify_intent("how do these two papers differ?", [], llm)) == "compare"


def test_unknown_label_falls_back_to_qa():
    llm = FakeLLM('{"intent": "banana"}')
    assert _run(classify_intent("anything", [], llm)) == "qa"


def test_unparseable_falls_back_to_qa():
    llm = FakeLLM("the model rambled with no json")
    assert _run(classify_intent("anything", [], llm)) == "qa"


def test_valid_intents_set():
    assert VALID_INTENTS == {"summary", "qa", "compare", "smalltalk"}
