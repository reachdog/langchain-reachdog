from threading import Barrier, Event
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from langchain_core.tools import ToolException
from langchain_tests.unit_tests import ToolsUnitTests
from langchain_reachdog import ReachDogBudget, ReachDogClient, ReachDogCatalogTool, ReachDogResearchTool, ReachDogTitleTool
from langchain_reachdog._client.reachdog_client import ApiError

def setup(handler, cap=4, approve=lambda _: True, **opts):
    client = ReachDogClient(base_url="https://api.test",api_key="test",transport=httpx.MockTransport(handler),sleep=lambda _:None,**opts)
    return dict(client=client,budget=ReachDogBudget(cap),account_label="unit",require_confirmation=approve)

body = {"title":"Tea","parameters":["brand"]}
def response(**fields):
    return httpx.Response(200,json=fields)

class TestTitleStandard(ToolsUnitTests):
    tool_constructor = ReachDogTitleTool
    @property
    def tool_constructor_params(self): return setup(lambda _:response())
    tool_invoke_params_example = {"body":body}

class TestResearchStandard(TestTitleStandard):
    tool_constructor = ReachDogResearchTool
    tool_invoke_params_example = {"body":{"subject":{"type":"category","value":"tea"},"parameters":["market_coverage"]}}

class TestCatalogStandard(TestTitleStandard):
    tool_constructor = ReachDogCatalogTool
    tool_invoke_params_example = {"action":"status","batch_id":"job"}

def test_preview_decline_missing_price_and_shared_budget():
    posts=[]
    def handler(r):
        if r.method=="GET": return response(parameter_price=3)
        posts.append(r.url.path)
        return httpx.Response(200,json={"brand":"Tea"},headers={"x-charge-state":"settled","x-charged-tokens":"3"})
    config=setup(handler)
    title=ReachDogTitleTool(**config)
    assert title.invoke({"body":body})["estimated_tokens"]==3
    tool=ReachDogTitleTool(**{k:v for k,v in config.items() if k!="require_confirmation"})
    assert tool.invoke({"body":body,"confirm":True})["approval"]=="declined"
    assert not posts
    assert title.invoke({"body":body,"confirm":True})["charged_tokens"]==3
    with pytest.raises(ToolException,match="budget"):
        ReachDogResearchTool(**config).invoke({"body":{"parameters":["market_coverage"]},"confirm":True})
    with pytest.raises(ToolException,match="no price"):
        ReachDogTitleTool(**setup(lambda _:response())).invoke({"body":body,"confirm":True})

def test_reserve_serializes_cap_check_and_insert():
    budget = ReachDogBudget(4)
    lock = budget._lock
    first_checked = Event()
    contenders = Barrier(2, timeout=5)

    # Pause A between checking the cap and recording it. B must either block
    # acquiring the lock, or (without the lock) reach its own unchecked write.
    class ObservedLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                if not contenders.broken and not inserted.is_set():
                    contenders.wait()
                lock.acquire()

        def __exit__(self, *exc):
            lock.release()

    class ObservedAmounts(dict):
        def __setitem__(self, key, value):
            if key == "a":
                first_checked.set()
            contenders.wait()
            super().__setitem__(key, value)
            inserted.set()

    inserted = Event()
    budget._lock = ObservedLock()
    budget._amounts = ObservedAmounts()

    def admit(key):
        try:
            with budget.reserve(key, 4):
                return "admitted"
        except ToolException as exc:
            assert "budget exceeded" in str(exc)
            return "blocked"

    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(admit, "a")
        assert first_checked.wait(5)
        second = workers.submit(admit, "b")
        assert [first.result(), second.result()] == ["admitted", "blocked"]


def test_lost_response_and_retry():
    keys=[]
    def handler(r):
        if r.method=="GET":return response(parameter_price=4)
        keys.append(r.headers["idempotency-key"])
        if len(keys)==1:raise httpx.ReadError("lost response")
        if len(keys)==2:return httpx.Response(503,headers={"Retry-After":"0"})
        return response(brand="Tea")
    tool=ReachDogTitleTool(**setup(handler,max_retries=1))
    with pytest.raises(ApiError):tool.invoke({"body":body,"confirm":True})
    with pytest.raises(ToolException,match="budget"):tool.invoke({"body":body,"confirm":True,"run_label":"new"})
    tool.invoke({"body":body,"confirm":True})
    assert len(keys)==3 and len(set(keys))==1
    keys.clear()
    def limited(r):
        if r.method=="GET":return response(parameter_price=1)
        keys.append(1);return httpx.Response(429,json={"code":"quota_exceeded"})
    with pytest.raises(ApiError):ReachDogTitleTool(**setup(limited)).invoke({"body":body,"confirm":True})
    assert keys==[1]

def test_catalog_recovery_and_refusal():
    calls=[]
    billing={"quote":{"amount":4},"acceptance":"binding","state":"awaiting_acceptance"}
    state={"id":"job","expires_at":"later","status":"awaiting_acceptance","billing":billing,"result_available":False}
    def handler(r):
        path=r.url.path
        calls.append(path)
        if path.endswith("accept") or path.endswith("retry-payment"):
            assert "idempotency-key" not in r.headers
            state.update(status="running");billing.update(state="running")
        if path.endswith("/results"):
            second="cursor" in r.url.params
            return response(workflow="catalog_local_v1",outputs={"groups":[2 if second else 1]},has_more=not second,next_cursor=None if second else "next")
        return response(**state)
    tool=ReachDogCatalogTool(**setup(handler),wait_seconds=0)
    accept={"action":"accept","batch_id":"job","acceptance":"binding"}
    assert tool.invoke(accept)["approval"]=="preview"
    tool.invoke({"action":"quote","body":{"records":[{"title":"Tea","client_product_id":"1"}],"parameters":["catalog_groups"]}})
    billing["quote"]["amount"]=3
    assert tool.invoke(accept)["approval"]=="preview"
    assert tool.invoke(accept)["status"]=="poll_timeout"
    tool.invoke(accept)
    assert sum(p.endswith("/accept") for p in calls)==1
    state.update(status="failed",error={"code":"wallet_frozen"});billing.update(state="refused")
    assert tool.invoke({"action":"status","batch_id":"job"})["recovery"]=="retry_payment"
    state["error"]["code"]="final"
    with pytest.raises(ToolException):tool.invoke({"action":"retry_payment","batch_id":"job"})
    state["error"]["code"]="wallet_frozen"
    tool.invoke({"action":"retry_payment","batch_id":"job"})
    state.update(status="completed",result_available=True);billing.update(state="settled",receipt={"deducted":3})
    assert tool.invoke({"action":"status","batch_id":"job"})["result"]["outputs"]["groups"]==[1,2]

CATALOG_ACCEPT = {"action":"accept", "batch_id":"job", "acceptance":"binding"}


def catalog_scenario(after_accept, approve=lambda _: True):
    quoted = {"id":"job", "expires_at":"later", "status":"awaiting_acceptance",
              "billing":{"quote":{"amount":4}, "acceptance":"binding", "state":"awaiting_acceptance"}}
    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(request.url.path)
            return response(**quoted)
        if "/v1/batches/job/accept" in posts:
            return after_accept(request)
        return response(**quoted)

    config = setup(handler, approve=approve)
    tool = ReachDogCatalogTool(**config, wait_seconds=0)
    tool.invoke({"action":"quote", "body":{"records":[{"title":"Tea","client_product_id":"1"}],
                                          "parameters":["catalog_groups"]}})
    return tool, config["budget"], posts


@pytest.mark.parametrize("phase", ["status", "results"])
def test_catalog_read_error_preserves_committed_hold(phase):
    def read(request):
        if phase == "results" and not request.url.path.endswith("/results"):
            return response(status="completed", result_available=True,
                            billing={"state":"settled","receipt":{"deducted":4}})
        return httpx.Response(429, json={"code":"quota_exceeded"})

    tool, budget, posts = catalog_scenario(read)
    with pytest.raises(ApiError) as failed:
        tool.invoke(CATALOG_ACCEPT)
    assert failed.value.status == 429
    assert posts == ["/v1/batches", "/v1/batches/job/accept"]
    assert budget._amounts["job"] == 4


def test_catalog_serializes_status_during_accept():
    polling, release = Event(), Event()
    def read(_request):
        # Only the first read waits. A competing read would return immediately.
        if not polling.is_set():
            polling.set()
            assert release.wait(5)
        return response(status="running", billing={"state":"running"})

    tool, _, _ = catalog_scenario(read)
    with ThreadPoolExecutor(1) as workers:
        acceptance = workers.submit(tool.invoke, CATALOG_ACCEPT)
        try:
            assert polling.wait(5)
            with pytest.raises(ToolException, match="Batch operation in progress"):
                tool.invoke({"action":"status","batch_id":"job"})
        finally:
            release.set()
        assert acceptance.result()["approval"] == "approved by user"


def test_catalog_decline_stops_accept_post():
    tool, budget, posts = catalog_scenario(lambda _: response(status="running"), approve=lambda _: False)
    assert tool.invoke(CATALOG_ACCEPT)["approval"] == "declined"
    assert posts == ["/v1/batches"]
    assert not budget._amounts


def test_known_no_charge_post_error_releases_cap():
    posts = []
    def handler(request):
        if request.method == "GET":
            return response(parameter_price=4)
        posts.append(request.headers["idempotency-key"])
        if len(posts) == 1:
            return httpx.Response(429,json={"code":"quota_exceeded"})
        return response(brand="Tea")

    title = ReachDogTitleTool(**setup(handler))
    with pytest.raises(ApiError) as failed:
        title.invoke({"body":body,"confirm":True})
    assert failed.value.status == 429
    assert title.invoke({"body":body,"confirm":True,"run_label":"other"})["approval"] == "approved by user"
    assert len(set(posts)) == 2


def test_refused_catalog_payment_releases_cap():
    tool, budget, posts = catalog_scenario(lambda _: response(
        status="failed", error={"code":"wallet_frozen"}, billing={"state":"refused"}))
    assert tool.invoke(CATALOG_ACCEPT)["recovery"] == "retry_payment"
    assert posts[-1] == "/v1/batches/job/accept"
    with budget.reserve("unrelated", 4):
        assert budget._amounts["unrelated"] == 4

def test_known_no_charge_retry_restores_previous_hold():
    budget = ReachDogBudget(4)
    with budget.reserve("job", 4):
        pass
    budget.settle("job", 0)  # A prior refusal has released this quote.
    with pytest.raises(ApiError):
        with budget.reserve("job", 4):
            error = ApiError(429, "quota_exceeded", "limited")
            error.ambiguous = False
            raise error
    with budget.reserve("other", 4):
        assert budget._amounts["job"] == 0
