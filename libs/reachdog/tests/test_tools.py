import json
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

def test_concurrency():
    tool=ReachDogTitleTool(**setup(lambda r:response(parameter_price=4) if r.method=="GET" else response()))
    def call(label):
        try: return tool.invoke({"body":body,"confirm":True,"run_label":label})["approval"]
        except ToolException: return "blocked"
    with ThreadPoolExecutor(2) as workers:
        assert sorted(workers.map(call,["a","b"]))==["approved by user","blocked"]

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
