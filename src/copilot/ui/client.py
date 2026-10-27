"""HTTP client for the copilot API, used by the Streamlit UI."""

import os

import httpx


class ApiError(Exception):
    pass


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout_s: float = 600.0):
        self.base_url = (
            base_url or os.environ.get("COPILOT_API_URL", "http://localhost:8000")
        ).rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise ApiError(f"API unreachable at {self.base_url}: {e}") from e
        if response.is_error:
            detail = (
                response.json().get("detail")
                if response.headers.get("content-type", "").startswith("application/json")
                else response.text
            )
            raise ApiError(f"{response.status_code}: {detail}")
        return response.json()

    def health(self) -> bool:
        try:
            return self._request("GET", "/health").get("status") == "ok"
        except ApiError:
            return False

    def chat(self, question: str, thread_id: str | None) -> dict:
        return self._request("POST", "/chat", json={"question": question, "thread_id": thread_id})

    def patients(self) -> list[dict]:
        return self._request("GET", "/patients")

    def analyze(self, patient_id: str) -> dict:
        return self._request("POST", "/analyze", json={"patient_id": patient_id})

    def review(
        self, thread_id: str, decision: str, comment: str = "", recommendations=None
    ) -> dict:
        body = {"decision": decision, "comment": comment}
        if recommendations is not None:
            body["recommendations"] = recommendations
        return self._request("POST", f"/analyze/{thread_id}/review", json=body)

    def resume(self, thread_id: str) -> dict:
        return self._request("POST", f"/analyze/{thread_id}/resume")

    def feedback(self, **item) -> dict:
        return self._request("POST", "/feedback", json=item)
