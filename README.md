# LangChain Reach Dog tools

Add product enrichment, market research and catalog grouping to a LangChain application.

```sh
pip install langchain-reachdog==0.1.0
```

Sign in to the [Reach Dog developer console](https://platform.reach.dog/keys), create your account's API key, and set `REACHDOG_API_KEY` in your environment. Use GitHub sign-in for the existing developer free-token access. Your API key's scopes must cover the operations you select.

## First request

This example previews a product request and asks you to approve the estimate before sending it.

```python
import json
import os
from langchain_reachdog import ReachDogBudget, ReachDogClient, ReachDogTitleTool


def approve(details):
    print(json.dumps(details, indent=2))
    return input("Approve this request? [y/N] ").strip().lower() == "y"


client = ReachDogClient(
    base_url="https://api.reach.dog",
    api_key=os.environ["REACHDOG_API_KEY"],
)
budget = ReachDogBudget()
tool = ReachDogTitleTool(
    client=client,
    budget=budget,
    account_label="my-application",
    require_confirmation=approve,
)
request = {"body": {"title": input("Product title: "), "parameters": ["product_type"]}}
print(tool.invoke(request))
print(tool.invoke({**request, "confirm": True}))
```

`ReachDogResearchTool` and `ReachDogCatalogTool` share the same client, budget and approval callback. Set `ReachDogBudget(max_tokens=YOUR_LIMIT)` to constrain estimated spending across an agent run. The unit is Reach Dog billing tokens. The default approval callback declines. Catalog processing uses quote, accept and status actions, and payment retry requires approval again. Reuse `run_label` when recovering the same operation. Change it when deliberately requesting new work.

See the [API reference](https://platform.reach.dog/docs/api-reference) and [catalog workflow](https://platform.reach.dog/docs/catalog-workflow) for scopes, request bodies and payment recovery. Report bugs in [GitHub issues](https://github.com/reachdog/langchain-reachdog/issues) with the package version and error code.

The vendored client and helpers are based on reachdog/reachdog-ecommerce-toolkit at 6437f3b. The client records ambiguous retries for budget accounting. The package is MIT licensed. API calls use your Reach Dog account's available tokens.

## Build from source

Build the wheel and source archive with `python -m build libs/reachdog`. For development, install `./libs/reachdog[test]` and run `python -m pytest libs/reachdog/tests`.
