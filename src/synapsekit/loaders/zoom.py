"""Zoom cloud-recording loader."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote, urlparse

from ._http_utils import http_client, safe_followup_url
from ._record_utils import records_to_documents, response_json
from .base import Document


class ZoomLoader:
    """Load Zoom cloud recordings for a user."""

    def __init__(
        self,
        access_token: str | None = None,
        user_id: str = "me",
        token: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        include_transcripts: bool = True,
        endpoint_url: str = "https://api.zoom.us/v2",
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        access_token = access_token or token
        if not access_token:
            raise ValueError("access_token must be provided")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._access_token = access_token
        self._user_id = user_id
        self._from_date = from_date
        self._to_date = to_date
        self._limit = limit
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._include_transcripts = include_transcripts
        self._endpoint_url = endpoint_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        records: list[dict[str, Any]] = []
        next_page_token: str | None = None
        headers = {"Authorization": f"Bearer {self._access_token}"}
        with http_client(self._client, self._timeout, "zoom") as client:
            while True:
                params: dict[str, Any] = {"page_size": min(self._limit or 300, 300)}
                if self._from_date:
                    params["from"] = self._from_date
                if self._to_date:
                    params["to"] = self._to_date
                if next_page_token:
                    params["next_page_token"] = next_page_token
                payload = response_json(
                    client.get(
                        f"{self._endpoint_url}/users/{quote(self._user_id, safe='')}/recordings",
                        params=params,
                        headers=headers,
                    )
                )
                batch = payload.get("meetings", [])
                if isinstance(batch, list):
                    for item in batch:
                        if self._limit is not None and len(records) >= self._limit:
                            break
                        if not isinstance(item, dict):
                            continue
                        record = dict(item)
                        if self._include_transcripts:
                            transcript = self._download_transcripts(
                                record, client, headers, self._endpoint_url
                            )
                            if transcript:
                                record["transcript"] = transcript
                        records.append(record)
                if self._limit is not None and len(records) >= self._limit:
                    break
                next_page_token = payload.get("next_page_token")
                if not next_page_token:
                    break

        documents = records_to_documents(
            records,
            "zoom",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            user_id=self._user_id,
        )
        for document in documents:
            document.metadata["meeting_id"] = document.metadata.get(
                "uuid", document.metadata.get("id")
            )
        return documents

    @staticmethod
    def _download_transcripts(
        record: dict[str, Any],
        client: Any,
        headers: dict[str, str],
        endpoint_url: str,
    ) -> str:
        existing = record.get("transcript")
        if existing:
            return str(existing)

        transcripts: list[str] = []
        recording_files = record.get("recording_files", [])
        for recording in recording_files if isinstance(recording_files, list) else []:
            if not isinstance(recording, dict):
                continue
            recording_type = str(
                recording.get("file_type")
                or recording.get("recording_type")
                or recording.get("file_extension")
                or ""
            ).lower()
            if "transcript" not in recording_type:
                continue
            download_url = recording.get("download_url")
            if not download_url:
                continue
            endpoint_host = urlparse(endpoint_url).hostname or ""
            suffixes = (
                ("zoom.us",)
                if endpoint_host == "zoom.us" or endpoint_host.endswith(".zoom.us")
                else ()
            )
            safe_url = safe_followup_url(
                str(download_url), endpoint_url, allowed_host_suffixes=suffixes
            )
            if safe_url is None:
                continue
            response = client.get(safe_url, headers=headers)
            response.raise_for_status()
            content = getattr(response, "content", b"")
            if isinstance(content, bytes):
                text = content.decode("utf-8", errors="replace")
            else:
                text = str(content) if content else str(getattr(response, "text", ""))
            if text:
                transcripts.append(text)
        return "\n\n".join(transcripts)

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
