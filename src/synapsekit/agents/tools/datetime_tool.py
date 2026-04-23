from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..base import BaseTool, ToolResult


class DateTimeTool(BaseTool):
    """Get current date/time or parse/format dates."""

    name = "datetime"
    description = (
        "Get the current date and time, or parse/format a date string. "
        "Input: optional action ('now', 'parse', 'format'), value, and format string."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: 'now' (default), 'parse', or 'format'",
                "enum": ["now", "parse", "format"],
                "default": "now",
            },
            "value": {
                "type": "string",
                "description": "Date string to parse or datetime to format",
                "default": "",
            },
            "fmt": {
                "type": "string",
                "description": "strftime/strptime format string (default: ISO 8601)",
                "default": "",
            },
            "tz": {
                "type": "string",
                "description": "Timezone: 'utc' or offset like '+05:30' (default: local)",
                "default": "",
            },
        },
        "required": [],
    }

    async def run(
        self,
        action: str = "now",
        value: str = "",
        fmt: str = "",
        tz: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        try:
            if action == "now":
                now = datetime.now(UTC) if tz.lower() == "utc" else datetime.now()
                if fmt:
                    return ToolResult(output=now.strftime(fmt))
                return ToolResult(output=now.isoformat())

            elif action == "parse":
                if not value:
                    return ToolResult(output="", error="No value provided to parse.")
                dt = datetime.strptime(value, fmt) if fmt else datetime.fromisoformat(value)
                return ToolResult(output=dt.isoformat())

            elif action == "format":
                if not value:
                    return ToolResult(output="", error="No value provided to format.")
                dt = datetime.fromisoformat(value)
                output_fmt = fmt or "%Y-%m-%d %H:%M:%S"
                return ToolResult(output=dt.strftime(output_fmt))

            else:
                return ToolResult(output="", error=f"Unknown action: {action!r}")
        except Exception as e:
            return ToolResult(output="", error=f"DateTime error: {e}")
