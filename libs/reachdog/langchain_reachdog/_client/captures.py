"""Exact request/response/header captures, keyed by cache identity, with expiry.

A capture holds the request as sent (minus the Authorization header), the response
status, headers and body, and the identity that produced it. A capture whose
`expires_at` has passed is reported as expired and never reused.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from .canonical import canonical_hash, canonical_json

SECRET_HEADERS = frozenset({"authorization", "x-api-key", "cookie", "set-cookie", "proxy-authorization"})


class CaptureError(Exception):
    pass


def scrub_headers(headers) -> dict:
    """Lower-case header names and drop credentials. Idempotency-Key is kept; it is not a secret."""
    return {k.lower(): v for k, v in dict(headers or {}).items() if k.lower() not in SECRET_HEADERS}


def parse_time(value) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class CaptureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, identity: dict) -> Path:
        return self.root / f"{canonical_hash(identity)}.json"

    def save(self, identity: dict, *, request: dict, response: dict, label: str) -> Path:
        """Store one exchange. `label` says what it is (for example 'live', 'synthetic')."""
        if not label:
            raise CaptureError("A capture needs a label saying whether it is live or synthetic.")
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "identity": identity,
            "label": label,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "request": {**request, "headers": scrub_headers(request.get("headers"))},
            "response": {**response, "headers": scrub_headers(response.get("headers"))},
            "expires_at": _expires_at(response),
        }
        text = canonical_json(record)
        if "authorization" in text.lower() and '"authorization"' in text.lower():
            raise CaptureError("Refusing to write a capture that contains an Authorization header.")
        path = self.path_for(identity)
        path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
        return path

    def load(self, identity: dict, *, now: datetime | None = None) -> dict | None:
        """Return a usable capture, or None. Expired captures raise so the caller can say why."""
        path = self.path_for(identity)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise CaptureError(f"Capture at {path} does not match the requested identity.")
        expiry = parse_time(record.get("expires_at"))
        if expiry is not None and expiry <= (now or datetime.now(timezone.utc)):
            raise CaptureError(f"Capture at {path} expired at {record['expires_at']}; it is not reused.")
        return record


def _expires_at(response: dict):
    body = response.get("body")
    if isinstance(body, dict) and body.get("expires_at"):
        return body["expires_at"]
    headers = scrub_headers(response.get("headers"))
    return headers.get("x-result-expires-at")
