"""Canonical JSON, body hashes and idempotency keys.

The canonical form matches the developer console's test rule
(rd-console/src/__tests__/fixtures/contracts/canonical.js): keys sorted, separators
"," and ":", non-ASCII escaped as backslash-u. The API's own persistence digest uses
ensure_ascii=False and differs for non-ASCII titles; that is expected and neither
side is changed to match the other.
"""
import hashlib
import json

KEY_MIN = 1
KEY_MAX = 200
KEY_PREFIX = "rdt1"


class CanonicalError(ValueError):
    pass


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def cache_identity(*, environment: str, account_ref: str, operation: str, payload,
                   api_revision: str, options=None) -> dict:
    """The full identity of a request for reuse. Never includes a secret.

    account_ref is a non-secret label for the account (for example the tenant or key
    name), never the API key itself.
    """
    for name, value in (("environment", environment), ("account_ref", account_ref),
                        ("operation", operation), ("api_revision", api_revision)):
        if not isinstance(value, str) or not value.strip():
            raise CanonicalError(f"{name} must be a non-empty string")
    if looks_like_secret(account_ref):
        raise CanonicalError("account_ref looks like a credential; use a non-secret label")
    return {"environment": environment, "account_ref": account_ref, "operation": operation,
            "api_revision": api_revision, "options": options or {},
            "payload_hash": canonical_hash(payload)}


def idempotency_key(identity: dict) -> str:
    """Derive the Idempotency-Key for an identity. Stable across key order and runs."""
    digest = canonical_hash(identity)
    key = f"{KEY_PREFIX}-{identity['operation']}-{digest}"
    validate_key(key)
    return key


def validate_key(key: str) -> str:
    if not isinstance(key, str) or not (KEY_MIN <= len(key) <= KEY_MAX):
        raise CanonicalError(f"Idempotency-Key must be {KEY_MIN}-{KEY_MAX} characters, got {len(key) if isinstance(key, str) else type(key).__name__}")
    return key


def looks_like_secret(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("tca_", "bearer ", "sk-")) or (len(value) >= 40 and " " not in value and any(c.isdigit() for c in value) and any(c.isalpha() for c in value) and value.isalnum())
