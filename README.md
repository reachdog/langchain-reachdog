# LangChain Reach Dog tools

Build an installable wheel with `python -m build --wheel libs/reachdog`. Install it with pip, or use `pip install './libs/reachdog[test]'` from a checkout for development. Run `python -m pytest libs/reachdog/tests` for the offline checks.

Construct ReachDogClient with the HTTPS API origin and bearer key (or set REACHDOG_API_URL and REACHDOG_API_KEY). Pass the same ReachDogBudget instance and stable, nonsecret account_label to every tool in one agent run. The budget counts Reach Dog billing tokens. Supply a host-owned require_confirmation(details) callback which displays the operation and amount and returns True only after approval. The default declines.

```python
from langchain_reachdog import ReachDogBudget, ReachDogClient, ReachDogTitleTool

client = ReachDogClient()
tool = ReachDogTitleTool(client=client, budget=ReachDogBudget(max_tokens=limit),
                         account_label=account_label, require_confirmation=approve)
preview = tool.invoke({"body": {"title": title, "parameters": ["product_type"]}})
```

### Live checks after operator approval

Inspect the title preview, invoke again with confirm=True, and check the settlement receipt. Repeat with ReachDogResearchTool using a research subject and supported parameters. Use the catalog tool's quote action to obtain a batch ID, binding and quote. Accept that binding only after reading it. Use status to resume the same batch. A recoverable wallet refusal offers retry_payment, which requires approval again. The `examples/group_csv.py` example takes a CSV, a host-provided model and approval callback to group products and generate descriptions. Changing run_label requests a fresh billable operation.

The vendored client and helpers are based on reachdog/reachdog-ecommerce-toolkit at 6437f3b. The client records ambiguous retries for budget accounting.
