"""Synchronous MCP client used by the (sync) LangGraph nodes.

One background event loop holds the MCP session open; calls are submitted to it and awaited
from the caller's thread. Connects in-process to an `MCPServer`, or over stdio to
`python -m copilot.mcp_server` (or any compatible server) via `StdioServerParameters`.
"""

import asyncio
import json
import shlex
import threading
from functools import lru_cache
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from copilot.agents.interactions import Interaction
from copilot.agents.patient import Patient
from copilot.agents.safety import SafetyIssue
from copilot.config import get_settings
from copilot.mcp_server.fhir import from_bundle
from copilot.mcp_server.server import GuidelineResults, InteractionReport
from copilot.rag.vector_store import SearchHit


class ToolCallError(Exception):
    pass


class McpToolbox:
    def __init__(self, server: MCPServer | StdioServerParameters, timeout_s: float = 120.0):
        self.timeout_s = timeout_s
        self._lock = threading.Lock()  # one in-flight call per session keeps ordering simple
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        ready = threading.Event()
        self._session = asyncio.run_coroutine_threadsafe(self._hold(server, ready), self._loop)
        if not ready.wait(timeout_s):
            raise ToolCallError("MCP server did not start")
        if self._session.done():  # connection failed during startup
            self._session.result()

    async def _hold(self, server, ready: threading.Event) -> None:
        # anyio requires the session to be entered and exited in the same task.
        self._stop = asyncio.Event()
        try:
            async with Client(server, read_timeout_seconds=self.timeout_s) as client:
                self._client = client
                ready.set()
                await self._stop.wait()
        finally:
            ready.set()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout_s)

    def call(self, name: str, arguments: dict[str, Any]) -> dict:
        with self._lock:
            result = self._submit(self._client.call_tool(name, arguments))
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise ToolCallError(f"{name}: {text}")
        if result.structured_content is not None:
            return result.structured_content
        # Tools returning a bare dict send it as JSON text instead of structured content.
        text = next((c.text for c in result.content if getattr(c, "text", None)), "{}")
        return json.loads(text)

    def list_tools(self) -> list[str]:
        return [t.name for t in self._submit(self._client.list_tools()).tools]

    # typed wrappers -------------------------------------------------------------------

    def search_guidelines(self, query: str, k: int = 5) -> list[SearchHit]:
        results = GuidelineResults.model_validate(
            self.call("search_guidelines", {"query": query, "k": k})
        )
        return [SearchHit(chunk=h.chunk, score=h.relevance) for h in results.hits]

    def get_patient(self, patient_id: str) -> Patient:
        return from_bundle(self.call("get_patient_record", {"patient_id": patient_id}))

    def check_drugs(
        self, drugs: list[str], patient_id: str | None = None
    ) -> tuple[list[Interaction], list[SafetyIssue]]:
        report = InteractionReport.model_validate(
            self.call("check_drug_interaction", {"drugs": drugs, "patient_id": patient_id})
        )
        return report.interactions, report.patient_issues

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        try:
            self._session.result(self.timeout_s)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)


class McpRetriever:
    """Retriever-compatible adapter so workflow retrieval goes through the MCP tool."""

    def __init__(self, toolbox: McpToolbox):
        self.toolbox = toolbox

    def retrieve(self, query: str, k: int | None = None, query_filter=None) -> list[SearchHit]:
        if query_filter is not None:
            raise ValueError("query_filter is not supported over MCP")
        return self.toolbox.search_guidelines(query, k=k or 5)


@lru_cache
def get_toolbox() -> McpToolbox:
    settings = get_settings()
    if settings.mcp_server_command:
        command, *args = shlex.split(settings.mcp_server_command)
        return McpToolbox(StdioServerParameters(command=command, args=args))
    from copilot.mcp_server.server import build_server
    from copilot.rag.retriever import get_retriever

    return McpToolbox(build_server(get_retriever))
