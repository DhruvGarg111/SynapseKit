from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.gemini import GeminiLLM

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeGeminiModel:
    def __init__(self) -> None:
        self.content: Any = None

    async def generate_content_async(self, content, **kwargs):
        self.content = content

        async def stream():
            yield SimpleNamespace(text="ok")

        return stream()


@pytest.mark.asyncio
async def test_gemini_message_stream_passes_image_part_to_model():
    model = FakeGeminiModel()
    llm = GeminiLLM(LLMConfig(model="gemini-2.0-flash", api_key="[REDACTED]", provider="gemini"))
    llm._model = model

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect the page"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(_TINY_PNG).decode()
                    },
                },
            ],
        }
    ]

    output = "".join([token async for token in llm.stream_with_messages(messages)])

    assert output == "ok"
    assert isinstance(model.content, list)
    assert any(getattr(part, "size", None) == (1, 1) for part in model.content)
    assert any(isinstance(part, str) and "Inspect the page" in part for part in model.content)
