from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import BaseTool, ToolResult


class FileListTool(BaseTool):
    """List files and directories at a given path."""

    name = "file_list"
    description = (
        "List files and directories in a given path. "
        "Input: directory path, optional recursive flag and glob pattern."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list"},
            "recursive": {
                "type": "boolean",
                "description": "List recursively (default: false)",
                "default": False,
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter (e.g. '*.py')",
                "default": "",
            },
        },
        "required": ["path"],
    }

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir else None

    async def run(
        self,
        path: str = "",
        recursive: bool = False,
        pattern: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        path = path or kwargs.get("input", ".")
        if self._base_dir is not None and not Path(path).resolve().is_relative_to(self._base_dir):
            return ToolResult(
                output="", error="Access denied: path is outside the allowed directory."
            )
        if not os.path.isdir(path):
            return ToolResult(output="", error=f"Not a directory: {path!r}")

        try:
            if recursive:
                import fnmatch

                entries = []
                for root, dirs, files in os.walk(path):
                    for name in dirs + files:
                        full = os.path.join(root, name)
                        rel = os.path.relpath(full, path)
                        if pattern and not fnmatch.fnmatch(name, pattern):
                            continue
                        suffix = "/" if os.path.isdir(full) else ""
                        entries.append(rel + suffix)
                entries.sort()
            else:
                import fnmatch

                items = sorted(os.listdir(path))
                entries = []
                for name in items:
                    if pattern and not fnmatch.fnmatch(name, pattern):
                        continue
                    full = os.path.join(path, name)
                    suffix = "/" if os.path.isdir(full) else ""
                    entries.append(name + suffix)

            if not entries:
                return ToolResult(output="(empty directory)")
            return ToolResult(output="\n".join(entries))
        except PermissionError:
            return ToolResult(output="", error=f"Permission denied: {path!r}")
        except Exception as e:
            return ToolResult(output="", error=f"Error listing directory: {e}")
