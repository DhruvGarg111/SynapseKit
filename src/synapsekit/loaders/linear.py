"""Linear issue/project loader."""

from __future__ import annotations

import asyncio
from typing import Any

from ._http_utils import http_client
from ._record_utils import records_to_documents, response_json
from .base import Document


class LinearLoader:
    """Load Linear issues or projects through the Linear GraphQL API."""

    def __init__(
        self,
        api_key: str,
        resource: str = "issues",
        team_id: str | None = None,
        project_id: str | None = None,
        query: str | None = None,
        variables: dict[str, Any] | None = None,
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
        limit: int = 100,
        endpoint_url: str = "https://api.linear.app/graphql",
        client: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be provided")
        if resource not in {"issues", "projects"}:
            raise ValueError("resource must be either 'issues' or 'projects'")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        if not endpoint_url:
            raise ValueError("endpoint_url must be provided")

        self._api_key = api_key
        self._resource = resource
        self._team_id = team_id
        self._project_id = project_id
        self._query = query
        self._variables = dict(variables or {})
        self._text_fields = text_fields
        self._metadata_fields = metadata_fields
        self._limit = limit
        self._endpoint_url = endpoint_url
        self._client = client
        self._timeout = timeout

    def load(self) -> list[Document]:
        variables = {"first": self._limit, **self._variables}
        if self._team_id:
            variables.setdefault("teamId", self._team_id)
        if self._project_id:
            variables.setdefault("projectId", self._project_id)
        query = self._query or self._default_query()
        headers = {"Authorization": self._api_key, "Content-Type": "application/json"}

        with http_client(self._client, self._timeout, "linear") as client:
            response = client.post(
                self._endpoint_url,
                json={"query": query, "variables": variables},
                headers=headers,
            )
            payload = response_json(response)

        if payload.get("errors"):
            raise RuntimeError(f"Linear GraphQL request failed: {payload['errors']}")
        records = self._extract_records(payload.get("data", {}))
        return records_to_documents(
            records,
            "linear",
            self._text_fields,
            self._metadata_fields,
            self._limit,
            resource=self._resource,
        )

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _default_query(self) -> str:
        if self._resource == "issues":
            filters: list[str] = []
            variable_declarations: list[str] = []
            if self._team_id:
                filters.append("team: {id: {eq: $teamId}}")
                variable_declarations.append("$teamId: ID")
            if self._project_id:
                filters.append("project: {id: {eq: $projectId}}")
                variable_declarations.append("$projectId: ID")
            filter_argument = f", filter: {{{', '.join(filters)}}}" if filters else ""
            variable_declaration = f", {', '.join(variable_declarations)}" if filters else ""
            fields = "id identifier title description url state { name }"
        else:
            filter_argument = ", filter: {id: {eq: $projectId}}" if self._project_id else ""
            variable_declaration = ", $projectId: ID" if self._project_id else ""
            fields = "id name description url state { name }"
        return (
            f"query($first: Int!{variable_declaration}) {{ {self._resource}"
            f"(first: $first{filter_argument}) {{ nodes {{ {fields} }} }} }}"
        )

    def _extract_records(self, data: Any) -> list[dict[str, Any]]:
        value = data.get(self._resource, []) if isinstance(data, dict) else []
        if isinstance(value, dict):
            value = value.get("nodes", value.get("edges", []))
        if isinstance(value, list) and value and isinstance(value[0], dict) and "node" in value[0]:
            value = [item["node"] for item in value]
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
