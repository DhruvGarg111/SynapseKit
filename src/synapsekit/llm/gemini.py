from __future__ import annotations

import base64
import io
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig, _messages_to_prompt


class GeminiLLM(BaseLLM):
    """Google Gemini provider with async streaming."""

    supports_multimodal = True

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._model: Any = None

    def _get_model(self):
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai required: pip install synapsekit[gemini]"
                ) from None
            genai.configure(api_key=self.config.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.config.model,
                system_instruction=self.config.system_prompt,
            )
        return self._model

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        model = self._get_model()
        async for chunk in await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": kw.get("temperature", self.config.temperature),
                "max_output_tokens": kw.get("max_tokens", self.config.max_tokens),
            },
            stream=True,
        ):
            if chunk.text:
                self._output_tokens += 1
                yield chunk.text

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        model = self._get_model()
        content = self._messages_to_gemini_content(messages)
        response = await model.generate_content_async(
            content,
            generation_config={
                "temperature": kw.get("temperature", self.config.temperature),
                "max_output_tokens": kw.get("max_tokens", self.config.max_tokens),
            },
            stream=True,
        )
        async for chunk in response:
            if chunk.text:
                self._output_tokens += 1
                yield chunk.text

    @classmethod
    def _messages_to_gemini_content(cls, messages: list[dict[str, Any]]) -> Any:
        """Convert OpenAI/Anthropic message blocks to Gemini content parts."""
        parts: list[Any] = []
        for message in messages:
            role = str(message.get("role", "user")).capitalize()
            value = message.get("content", "")
            if isinstance(value, str):
                if value:
                    parts.append(f"{role}: {value}")
                continue
            if not isinstance(value, list):
                parts.append(f"{role}: {value}")
                continue

            text_parts: list[str] = []
            image_parts: list[Any] = []
            for block in value:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = str(block.get("text", ""))
                    if text:
                        text_parts.append(text)
                elif block_type == "image_url":
                    image_url = block.get("image_url", {})
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
                    image_parts.append(cls._gemini_image_part(str(url or "")))
                elif block_type == "image":
                    source = block.get("source") or {}
                    if source.get("type") == "base64":
                        image_parts.append(
                            cls._gemini_image_bytes(
                                str(source.get("data", "")),
                                str(source.get("media_type", "image/png")),
                            )
                        )
                    elif source.get("type") == "url":
                        image_parts.append(
                            {
                                "file_data": {
                                    "mime_type": str(source.get("media_type", "image/png")),
                                    "file_uri": str(source.get("url", "")),
                                }
                            }
                        )
            if text_parts:
                parts.append(f"{role}: {' '.join(text_parts)}")
            parts.extend(image_parts)

        if not parts:
            return ""
        return parts[0] if len(parts) == 1 else parts

    @classmethod
    def _gemini_image_part(cls, url: str) -> Any:
        if url.startswith("data:"):
            header, separator, encoded = url.partition(",")
            if separator:
                media_type = header[5:].split(";", 1)[0] or "image/png"
                return cls._gemini_image_bytes(encoded, media_type)
        return {"file_data": {"mime_type": "image/png", "file_uri": url}}

    @staticmethod
    def _gemini_image_bytes(encoded: str, media_type: str) -> Any:
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return f"[Invalid image content: {media_type}]"
        try:
            from PIL import Image

            return Image.open(io.BytesIO(raw)).convert("RGB")
        except ImportError:
            return {
                "inline_data": {
                    "mime_type": media_type,
                    "data": raw,
                }
            }
        except (OSError, ValueError):
            return {
                "inline_data": {
                    "mime_type": media_type,
                    "data": raw,
                }
            }

    async def _call_with_tools_impl(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Native function-calling. Returns {"content": str|None, "tool_calls": list|None}."""
        import google.generativeai as genai

        model = self._get_model()

        # Convert OpenAI tool schema → Gemini function declarations
        func_decls = []
        for t in tools:
            fn = t["function"]
            func_decls.append(
                genai.protos.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=self._convert_params(fn.get("parameters", {})),
                )
            )
        gemini_tools = [genai.protos.Tool(function_declarations=func_decls)]

        # Build prompt from messages
        prompt = _messages_to_prompt(messages)

        response = await model.generate_content_async(
            prompt,
            tools=gemini_tools,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
        )

        # Parse response parts for function calls vs text
        tool_calls = []
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args)
                        if part.function_call.args
                        else {},
                    }
                )
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if tool_calls:
            return {"content": None, "tool_calls": tool_calls}
        return {"content": "".join(text_parts) if text_parts else "", "tool_calls": None}

    @staticmethod
    def _convert_params(params: dict) -> dict:
        """Convert JSON Schema parameters to Gemini-compatible format."""
        if not params:
            return {}
        # Gemini accepts a subset of JSON Schema
        result: dict[str, Any] = {"type": params.get("type", "object").upper()}
        if "properties" in params:
            result["properties"] = {}
            for name, prop in params["properties"].items():
                result["properties"][name] = {
                    "type": prop.get("type", "string").upper(),
                    "description": prop.get("description", ""),
                }
        if "required" in params:
            result["required"] = params["required"]
        return result
