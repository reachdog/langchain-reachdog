"""Group a product CSV, then draft collection descriptions with a host-selected chat model."""
import csv
import json
from langchain_reachdog import ReachDogBudget, ReachDogCatalogTool, ReachDogClient


def group_csv(path, model, *, account_label, max_tokens, approve):
    client = ReachDogClient()
    try:
        tool = ReachDogCatalogTool(client=client, account_label=account_label,
                                   budget=ReachDogBudget(max_tokens), require_confirmation=approve)
        with open(path, encoding="utf-8-sig", newline="") as source:
            records = [{"client_product_id":r["client_product_id"],"title":r["title"]} for r in csv.DictReader(source)]
        quote = tool.invoke({"action":"quote","body":{"records":records,"parameters":["catalog_groups","group_members"]}})
        if "quote" in quote:
            result = tool.invoke({"action":"accept","batch_id":quote["batch_id"],"acceptance":quote["acceptance"]})
        else:
            result = quote
        if result.get("status") != "completed":
            return result
        return model.invoke("Draft collection descriptions using only these group labels and members:\n" +
                            json.dumps(result["result"]["outputs"]))
    finally:
        client.close()
