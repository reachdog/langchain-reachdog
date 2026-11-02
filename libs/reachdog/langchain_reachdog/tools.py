"""LangChain tool interfaces. Only the host application supplies clients, limits and approval."""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from threading import RLock
from typing import Callable, Literal
from urllib.parse import urlsplit

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from ._client import API_REVISION
from ._client.canonical import cache_identity, canonical_json, idempotency_key
from ._client.costs import receipt
from ._client.reachdog_client import ApiError, ReachDogClient


def checked(amount):
    if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
        raise ToolException("Missing or invalid Reach Dog billing amount")
    return amount


class ReachDogBudget:
    def __init__(self, max_tokens: int | None = None):
        self.limit = None if max_tokens is None else checked(max_tokens)
        self._lock = RLock()
        self._amounts = {}
        self._active = set()
        self._catalog_busy = set()
        self.shown = {}
        self.accepted = {}

    @contextmanager
    def catalog_turn(self, batch_id):
        # Serialize status observations with payment transitions for the same batch.
        with self._lock:
            if batch_id in self._catalog_busy:
                raise ToolException("Batch operation in progress; resume with status")
            self._catalog_busy.add(batch_id)
        try:
            yield
        finally:
            with self._lock:
                self._catalog_busy.remove(batch_id)

    @contextmanager
    def reserve(self, key, amount):
        with self._lock:
            if key in self._active:
                raise ToolException("Operation already in progress")
            previous = self._amounts.get(key)
            hold = max(previous or 0, checked(amount))
            if self.limit is not None and sum(self._amounts.values()) - (previous or 0) + hold > self.limit:
                raise ToolException("Run budget exceeded")
            self._active.add(key)
            self._amounts[key] = hold
        try:
            yield
        except ApiError as exc:
            if previous is None and not getattr(exc, "ambiguous", True) and exc.status in (400,401,402,403,422,429):
                self.settle(key, 0)
            raise
        finally:
            with self._lock:
                self._active.remove(key)

    def settle(self, key, amount):
        with self._lock:
            if key in self._amounts:
                self._amounts[key] = checked(amount)


class SingleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    body: dict
    confirm: bool = False
    run_label: str | None = None


class CatalogInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["quote", "accept", "status", "retry_payment"]
    body: dict | None = None
    batch_id: str | None = None
    acceptance: str | None = None
    run_label: str | None = None


def decline(_details):
    return False


class _Tool(BaseTool):
    client: ReachDogClient = Field(exclude=True)
    budget: ReachDogBudget = Field(exclude=True)
    account_label: str = Field(default_factory=lambda: os.environ["REACHDOG_ACCOUNT_LABEL"], exclude=True)
    require_confirmation: Callable[[dict], bool] = Field(default=decline, exclude=True)

    def key(self, operation, body, run_label):
        identity = cache_identity(environment=urlsplit(self.client.base_url).netloc, account_ref=self.account_label,
                                  operation=operation, payload=body, api_revision=API_REVISION,
                                  options={} if run_label is None else {"run_label":run_label})
        return idempotency_key(identity)

    def single(self, operation, body, confirm, run_label):
        key = self.key(operation, body, run_label)
        parameters = body.get("parameters")
        if not isinstance(parameters, list) or not parameters:
            raise ToolException("A nonempty parameter list is required")
        price = self.client.usage().body.get("parameter_price")
        if price is None:
            raise ToolException("no price available for this account")
        estimate = {"estimated_tokens":len(parameters)*checked(price), "basis":"estimate, not a locked quote"}
        if not confirm or self.require_confirmation({"operation":operation,"body":body,**estimate}) is not True:
            return {**estimate, "approval":"declined" if confirm else "preview"}
        with self.budget.reserve(key, estimate["estimated_tokens"]):
            response = getattr(self.client, operation)(body, idempotency_key=key)
            charged = receipt(response.headers)
            if charged is not None:
                self.budget.settle(key, charged)
            return {"approval":"approved by user","result":response.body,"charged_tokens":charged,
                    "charge_state":response.headers.get("x-charge-state")}


class ReachDogTitleTool(_Tool):
    name: str = "enrich_title"
    description: str = "Preview or confirm one title match with the supplied API request body."
    args_schema: type[BaseModel] = SingleInput

    def _run(self, body, confirm=False, run_label=None):
        return self.single("title_match", body, confirm, run_label)


class ReachDogResearchTool(_Tool):
    name: str = "research_market"
    description: str = "Preview or confirm one market research request."
    args_schema: type[BaseModel] = SingleInput

    def _run(self, body, confirm=False, run_label=None):
        return self.single("market_research", body, confirm, run_label)


class ReachDogCatalogTool(_Tool):
    name: str = "enrich_catalog"
    description: str = "Quote a catalog, explicitly accept it, resume its status or retry a refused payment."
    args_schema: type[BaseModel] = CatalogInput
    wait_seconds: float = Field(default=120, ge=0, le=120, exclude=True)

    def display(self, state):
        billing = state["billing"]
        self.budget.shown[state["id"]] = canonical_json([billing["quote"], billing["acceptance"]])
        return {"batch_id":state["id"], "quote":billing["quote"], "acceptance":billing["acceptance"],
                "expires_at":state["expires_at"],"approval":"preview"}

    def status(self, batch_id):
        try:
            state = self.client.poll_catalog(batch_id, wait_seconds=self.wait_seconds).body
        except ApiError as exc:
            if exc.code != "poll_timeout":
                raise
            return {"batch_id":batch_id,"status":"poll_timeout","recovery":"status"}
        billing = state["billing"]
        if billing["state"] == "refused":
            self.budget.settle(batch_id, 0)
            retry = state.get("error", {}).get("code") in ("insufficient_credits","wallet_frozen")
            return {"batch_id":batch_id,"status":state["status"],"error":state.get("error"),
                    "recovery":"retry_payment" if retry else None}
        if state["status"] == "completed" and state["result_available"] and billing["state"] == "settled":
            self.budget.settle(batch_id, billing["receipt"]["deducted"])
            result = self.client.collect_catalog_results(f"/v1/batches/{batch_id}/results")
            return {"batch_id":batch_id,"status":"completed","result":result,"receipt":billing["receipt"]}
        return state

    def _run(self, action, body=None, batch_id=None, acceptance=None, run_label=None):
        if action == "quote":
            if not body or batch_id or acceptance:
                raise ToolException("Supply a catalog request body to preview")
            payload = {"workflow":"catalog_local_v1", **body}
            if payload["workflow"] != "catalog_local_v1":
                raise ToolException("Only the quoted catalog workflow is supported")
            key = self.key("catalog", payload, run_label)
            state = self.client.request("POST", "/v1/batches", json=payload, idempotency_key=key).body
            if state["status"] == "awaiting_acceptance":
                return self.display(state)
            with self.budget.catalog_turn(state["id"]):
                return self.status(state["id"])
        if not batch_id or any(ch in batch_id for ch in "/?#") or batch_id in (".",".."):
            raise ToolException("Invalid batch ID")
        with self.budget.catalog_turn(batch_id):
            return self._existing(action, batch_id, acceptance)

    def _existing(self, action, batch_id, acceptance):
        if action == "status":
            return self.status(batch_id)
        state = self.client.batch_status(batch_id).body
        billing = state["billing"]
        binding = canonical_json([billing["quote"],billing["acceptance"]])
        if action == "retry_payment":
            if (billing["state"] != "refused" or state.get("error", {}).get("code") not in ("insufficient_credits","wallet_frozen")
                    or self.budget.accepted.get(batch_id) != binding):
                raise ToolException("Retry requires a recoverable refusal for the original accepted quote")
            method = self.client.retry_payment
        elif state["status"] != "awaiting_acceptance":
            return self.status(batch_id)
        elif self.budget.shown.get(batch_id) != binding or acceptance != billing["acceptance"]:
            return self.display(state)
        else:
            method = self.client.accept_quote
        details = {"batch_id":batch_id,"quote":billing["quote"],"action":action}
        if self.require_confirmation(details) is not True:
            return {**details,"approval":"declined"}
        with self.budget.reserve(batch_id, billing["quote"]["amount"]):
            self.budget.accepted[batch_id] = binding
            method(batch_id, billing["acceptance"])
            return {**self.status(batch_id),"approval":"approved by user"}
