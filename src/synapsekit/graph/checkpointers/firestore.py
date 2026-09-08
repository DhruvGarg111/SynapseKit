"""Firestore-backed graph checkpoints."""

from __future__ import annotations

from typing import Any

from ..._json import dumps as _json_dumps
from ..._json import loads as _json_loads
from ...memory.backends._common import AsyncCheckpointerMixin, encode_document_id
from .base import BaseCheckpointer


class FirestoreCheckpointer(AsyncCheckpointerMixin, BaseCheckpointer):
    """Persist one latest JSON checkpoint per graph in Firestore."""

    def __init__(
        self,
        project_id: str = "synapsekit-local",
        collection: str = "graph_checkpoints",
        *,
        credentials_path: str | None = None,
        database: str = "(default)",
        client: Any | None = None,
    ) -> None:
        if not project_id or not collection:
            raise ValueError("project_id and collection must be provided")
        self._project_id = project_id
        self._collection_name = collection
        self._credentials_path = credentials_path
        self._database = database
        self._client = client
        self._collection: Any | None = None

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        if self._client is None:
            try:
                from google.cloud import firestore
            except ImportError:
                raise ImportError(
                    "google-cloud-firestore is required for FirestoreCheckpointer; "
                    "install it with `pip install synapsekit[firestore]`"
                ) from None
            kwargs: dict[str, Any] = {
                "project": self._project_id,
                "database": self._database,
            }
            if self._credentials_path:
                from google.oauth2 import service_account

                kwargs["credentials"] = service_account.Credentials.from_service_account_file(
                    self._credentials_path
                )
            self._client = firestore.Client(**kwargs)
        self._collection = self._client.collection(self._collection_name)
        return self._collection

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        self._ensure_collection().document(encode_document_id(graph_id)).set(
            {"graph_id": graph_id, "step": int(step), "state": _json_dumps(state)}
        )

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        snapshot = self._ensure_collection().document(encode_document_id(graph_id)).get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict()
        return int(data["step"]), dict(_json_loads(data["state"]))

    def delete(self, graph_id: str) -> None:
        self._ensure_collection().document(encode_document_id(graph_id)).delete()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None
