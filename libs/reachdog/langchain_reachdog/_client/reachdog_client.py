"""Thin HTTP client for the Reach Dog Title and Taxonomy API (revision b03e832).

Every call returns an ApiResponse with the status, the response headers and the body
intact. Non-2xx responses raise ApiError carrying the server's error code. Transport
retries apply only to connection errors and 5xx responses, always with the same
Idempotency-Key, bounded by a total time budget. A new key is never minted on retry.
The API key and the Authorization header are never written to logs or captures.
"""
import os
import time
from dataclasses import dataclass, field

import httpx

from .canonical import validate_key
from .captures import scrub_headers

ENV_KEY = "REACHDOG_API_KEY"
ENV_URL = "REACHDOG_API_URL"
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})
TERMINAL_CATALOG_STATUS = frozenset({"completed", "failed", "cancelled"})


class ClientConfigError(Exception):
    pass


class ApiError(Exception):
    """A non-2xx response. `code` is the server's error code when the body carried one."""

    def __init__(self, status: int, code: str, message: str, *, retryable: bool = False,
                 request_id: str | None = None, details=None, headers=None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message
        self.retryable, self.request_id = retryable, request_id
        self.details = details if details is not None else {}
        self.headers = headers or {}

    @classmethod
    def from_response(cls, response: "ApiResponse") -> "ApiError":
        body = response.body if isinstance(response.body, dict) else {}
        return cls(response.status, body.get("code") or f"http_{response.status}",
                   body.get("message") or "The API returned an error without a message.",
                   retryable=bool(body.get("retryable")), request_id=body.get("request_id") or response.headers.get("x-request-id"),
                   details=body.get("details"), headers=response.headers)


@dataclass
class ApiResponse:
    status: int
    headers: dict = field(default_factory=dict)
    body: object = None

    @property
    def result_id(self):
        return self.headers.get("x-result-id")

    @property
    def location(self):
        return self.headers.get("location")

    def as_capture(self) -> dict:
        return {"status": self.status, "headers": dict(self.headers), "body": self.body}


class ReachDogClient:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None, timeout: float = 60.0,
                 retry_budget_seconds: float = 90.0, max_retries: int = 3, transport=None, sleep=time.sleep):
        self.base_url = (base_url or os.environ.get(ENV_URL) or "").rstrip("/")
        api_key = api_key or os.environ.get(ENV_KEY)
        if not self.base_url:
            raise ClientConfigError(f"{ENV_URL} is not set; the API base URL is required.")
        if not api_key:
            raise ClientConfigError(f"{ENV_KEY} is not set; a bearer key with the needed scopes is required.")
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self.timeout, self.retry_budget, self.max_retries, self._sleep = timeout, retry_budget_seconds, max_retries, sleep
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)
        self.exchanges: list[dict] = []  # scrubbed request/response pairs for captures

    def close(self):
        self._http.close()

    def __repr__(self):
        return f"ReachDogClient(base_url={self.base_url!r})"  # never the key

    # ----- transport -----
    def request(self, method: str, path: str, *, json=None, params=None, idempotency_key: str | None = None) -> ApiResponse:
        headers = {"Accept": "application/json"}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = validate_key(idempotency_key)
        deadline = time.monotonic() + self.retry_budget
        attempt = 0
        while True:
            attempt += 1
            try:
                raw = self._http.request(method, path, json=json, params=params, headers={**headers, **self._auth})
            except httpx.TransportError as exc:
                if attempt > self.max_retries or time.monotonic() >= deadline:
                    raise ApiError(0, "transport_error", f"{type(exc).__name__} after {attempt} attempt(s); the same Idempotency-Key may be retried later.", retryable=True) from exc
                self._sleep(min(2 ** attempt, 10))
                continue
            response = ApiResponse(raw.status_code, scrub_headers(raw.headers), _body(raw))
            self.exchanges.append({"request": {"method": method, "path": path, "params": params, "json": json, "headers": headers},
                                   "response": response.as_capture()})
            if raw.status_code in RETRYABLE_STATUS and attempt <= self.max_retries and time.monotonic() < deadline:
                self._sleep(_retry_after(response.headers, attempt))
                continue
            if raw.status_code >= 400:
                error = ApiError.from_response(response)
                error.ambiguous = attempt > 1
                raise error
            return response

    # ----- catalog_local_v1 -----
    def submit_catalog(self, records: list[dict], parameters: list[str], *, idempotency_key: str, refine_probes: bool = False) -> ApiResponse:
        body = {"workflow": "catalog_local_v1", "records": records, "parameters": list(parameters), "refine_probes": bool(refine_probes)}
        return self.request("POST", "/v1/batches", json=body, idempotency_key=idempotency_key)

    def accept_quote(self, batch_id: str, acceptance: str) -> ApiResponse:
        return self.request("POST", f"/v1/batches/{batch_id}/accept", json={"acceptance": acceptance})

    def retry_payment(self, batch_id: str, acceptance: str) -> ApiResponse:
        return self.request("POST", f"/v1/batches/{batch_id}/retry-payment", json={"acceptance": acceptance})

    def batch_status(self, batch_id: str) -> ApiResponse:
        return self.request("GET", f"/v1/batches/{batch_id}")

    def cancel_batch(self, batch_id: str) -> ApiResponse:
        return self.request("POST", f"/v1/batches/{batch_id}/cancel")

    def usage(self) -> ApiResponse:
        return self.request("GET", "/v1/usage")

    def poll_catalog(self, batch_id: str, *, wait_seconds: float = 1800.0, interval: float = 5.0) -> ApiResponse:
        """Poll until the catalog reaches a terminal status. The caller inspects the status."""
        deadline = time.monotonic() + wait_seconds
        while True:
            response = self.batch_status(batch_id)
            body = response.body if isinstance(response.body, dict) else {}
            if body.get("status") in TERMINAL_CATALOG_STATUS or body.get("status") == "awaiting_acceptance":
                return response
            if time.monotonic() >= deadline:
                raise ApiError(0, "poll_timeout", f"Catalog {batch_id} is still {body.get('status')!r} after {wait_seconds:.0f}s; resume later with the same request.", retryable=True)
            self._sleep(_retry_after(response.headers, 1, default=interval))

    def results_pages(self, results_url: str, *, page_size: int = 100):
        """Yield every page of a results URL, passing next_cursor back until has_more is false."""
        cursor = None
        while True:
            params = {"page_size": page_size}
            if cursor:
                params["cursor"] = cursor
            response = self.request("GET", results_url, params=params)
            yield response
            body = response.body if isinstance(response.body, dict) else {}
            if not body.get("has_more"):
                return
            cursor = body.get("next_cursor")
            if not cursor:
                raise ApiError(0, "pagination_error", "has_more is true but next_cursor is missing.")

    def collect_catalog_results(self, results_url: str, *, page_size: int = 100) -> dict:
        return merge_pages([r.body for r in self.results_pages(results_url, page_size=page_size)])

    # ----- single-title, research, content -----
    def title_match(self, body: dict, *, idempotency_key: str) -> ApiResponse:
        return self.request("POST", "/v1/title-matches", json=body, idempotency_key=idempotency_key)

    def market_research(self, body: dict, *, idempotency_key: str) -> ApiResponse:
        return self.request("POST", "/v1/market-research", json=body, idempotency_key=idempotency_key)

    def content_generation(self, body: dict, *, idempotency_key: str) -> ApiResponse:
        return self.request("POST", "/v1/content-generations", json=body, idempotency_key=idempotency_key)

    def read_saved(self, resource: str, result_id: str) -> ApiResponse:
        if resource not in ("title-matches", "market-research", "content-generations"):
            raise ValueError("resource must be title-matches, market-research or content-generations")
        return self.request("GET", f"/v1/{resource}/{result_id}")


def merge_pages(pages: list[dict]) -> dict:
    """Join result pages: list outputs concatenate; scalar outputs and processing appear once."""
    if not pages:
        raise ValueError("No result pages were returned.")
    merged = {"workflow": pages[0].get("workflow"), "processing": pages[0].get("processing"),
              "expires_at": pages[0].get("expires_at"), "outputs": {}, "page_count": len(pages)}
    for page in pages:
        for name, value in (page.get("outputs") or {}).items():
            if isinstance(value, list):
                merged["outputs"].setdefault(name, []).extend(value)
            elif name not in merged["outputs"]:
                merged["outputs"][name] = value  # scalars repeat on every page; keep the first
    return merged


def _body(raw: httpx.Response):
    if not raw.content:
        return None
    try:
        return raw.json()
    except ValueError:
        return raw.text


def _retry_after(headers: dict, attempt: int, default: float | None = None) -> float:
    value = headers.get("retry-after")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default if default is not None else min(2 ** attempt, 10)
