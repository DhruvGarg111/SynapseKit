"""Gmail message and thread loader."""

from __future__ import annotations

import asyncio
import base64
import binascii
from typing import Any

from .base import Document


class GmailLoader:
    """Load Gmail messages or threads as :class:`Document` objects.

    Credentials are loaded lazily so importing ``synapsekit.loaders`` does not
    require Google packages. A pre-built ``service`` is useful for tests and
    applications that already manage Google authentication.
    """

    _SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    def __init__(
        self,
        credentials_path: str | None = None,
        credentials_dict: dict[str, Any] | None = None,
        credentials: Any | None = None,
        service: Any | None = None,
        user_id: str = "me",
        query: str | None = None,
        thread_id: str | None = None,
        message_id: str | None = None,
        max_results: int = 100,
        mode: str = "messages",
    ) -> None:
        if service is None and not (credentials_path or credentials_dict or credentials):
            raise ValueError("credentials or service must be provided")
        if thread_id and message_id:
            raise ValueError("thread_id and message_id cannot both be provided")
        if max_results <= 0:
            raise ValueError("max_results must be greater than 0")
        if mode not in {"messages", "threads"}:
            raise ValueError("mode must be either 'messages' or 'threads'")

        self._credentials_path = credentials_path
        self._credentials_dict = credentials_dict
        self._credentials = credentials
        self._service = service
        self._user_id = user_id
        self._query = query
        self._thread_id = thread_id
        self._message_id = message_id
        self._max_results = max_results
        self._mode = mode

    def load(self) -> list[Document]:
        service = self._service if self._service is not None else self._build_service()
        if self._message_id:
            message = (
                self._messages(service)
                .get(userId=self._user_id, id=self._message_id, format="full")
                .execute()
            )
            return [self._message_to_document(message)]
        if self._thread_id:
            thread = (
                self._threads(service)
                .get(userId=self._user_id, id=self._thread_id, format="full")
                .execute()
            )
            return [self._thread_to_document(thread)]
        if self._mode == "threads":
            return self._load_threads(service)
        return self._load_messages(service)

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _build_service(self) -> Any:
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Gmail dependencies required: pip install synapsekit[gmail]"
            ) from None

        if self._credentials is not None:
            creds = self._credentials
        elif self._credentials_path:
            creds = service_account.Credentials.from_service_account_file(
                self._credentials_path, scopes=self._SCOPES
            )
        else:
            creds = service_account.Credentials.from_service_account_info(
                self._credentials_dict, scopes=self._SCOPES
            )
        return build("gmail", "v1", credentials=creds, cache_discovery=False)

    @staticmethod
    def _users(service: Any) -> Any:
        return service.users()

    def _messages(self, service: Any) -> Any:
        return self._users(service).messages()

    def _threads(self, service: Any) -> Any:
        return self._users(service).threads()

    def _load_messages(self, service: Any) -> list[Document]:
        message_ids: list[dict[str, Any]] = []
        page_token: str | None = None
        while len(message_ids) < self._max_results:
            kwargs: dict[str, Any] = {
                "userId": self._user_id,
                "maxResults": min(100, self._max_results - len(message_ids)),
            }
            if self._query:
                kwargs["q"] = self._query
            if page_token:
                kwargs["pageToken"] = page_token
            result = self._messages(service).list(**kwargs).execute()
            message_ids.extend(result.get("messages", []))
            page_token = result.get("nextPageToken")
            if not page_token or not result.get("messages"):
                break

        documents: list[Document] = []
        for item in message_ids[: self._max_results]:
            message = (
                self._messages(service)
                .get(userId=self._user_id, id=item["id"], format="full")
                .execute()
            )
            documents.append(self._message_to_document(message))
        return documents

    def _load_threads(self, service: Any) -> list[Document]:
        thread_ids: list[dict[str, Any]] = []
        page_token: str | None = None
        while len(thread_ids) < self._max_results:
            kwargs: dict[str, Any] = {
                "userId": self._user_id,
                "maxResults": min(100, self._max_results - len(thread_ids)),
            }
            if self._query:
                kwargs["q"] = self._query
            if page_token:
                kwargs["pageToken"] = page_token
            result = self._threads(service).list(**kwargs).execute()
            thread_ids.extend(result.get("threads", []))
            page_token = result.get("nextPageToken")
            if not page_token or not result.get("threads"):
                break

        documents: list[Document] = []
        for item in thread_ids[: self._max_results]:
            thread = (
                self._threads(service)
                .get(userId=self._user_id, id=item["id"], format="full")
                .execute()
            )
            documents.append(self._thread_to_document(thread))
        return documents

    def _message_to_document(self, message: dict[str, Any]) -> Document:
        headers = self._headers(message)
        subject = headers.get("subject", "")
        body = self._body(message.get("payload", {}))
        text = "\n".join(part for part in (subject, body) if part)
        metadata = {
            "source": "gmail",
            "message_id": message.get("id"),
            "thread_id": message.get("threadId"),
            "label_ids": message.get("labelIds", []),
            "subject": subject,
            "from": headers.get("from"),
            "to": headers.get("to"),
            "date": headers.get("date"),
            "snippet": message.get("snippet"),
        }
        return Document(text=text, metadata=metadata)

    def _thread_to_document(self, thread: dict[str, Any]) -> Document:
        messages = thread.get("messages", [])
        message_docs = [self._message_to_document(message) for message in messages]
        text = "\n\n".join(doc.text for doc in message_docs if doc.text)
        subject = next(
            (doc.metadata["subject"] for doc in message_docs if doc.metadata.get("subject")),
            "",
        )
        message_metadata = [dict(doc.metadata) for doc in message_docs]
        label_ids: list[str] = []
        for metadata in message_metadata:
            for label_id in metadata.get("label_ids", []):
                if label_id not in label_ids:
                    label_ids.append(label_id)
        return Document(
            text=text,
            metadata={
                "source": "gmail",
                "thread_id": thread.get("id"),
                "message_ids": [doc.metadata["message_id"] for doc in message_docs],
                "subject": subject,
                "label_ids": label_ids,
                "messages": message_metadata,
            },
        )

    @staticmethod
    def _headers(message: dict[str, Any]) -> dict[str, str]:
        headers = message.get("payload", {}).get("headers", [])
        return {
            str(item.get("name", "")).lower(): str(item.get("value", ""))
            for item in headers
            if item.get("name")
        }

    def _body(self, payload: dict[str, Any]) -> str:
        parts: list[str] = []
        body = payload.get("body", {})
        if body.get("data"):
            parts.append(self._decode(body["data"]))
        for part in payload.get("parts", []) or []:
            text = self._body(part)
            if text:
                parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _decode(value: str) -> str:
        try:
            return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode(
                "utf-8", errors="replace"
            )
        except (ValueError, TypeError, binascii.Error):
            return value
