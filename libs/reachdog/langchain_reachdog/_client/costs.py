"""Client-side recipe budgets using the existing usage price and settlement headers."""
import math
from .reachdog_client import ApiError
from .errors import RecipeBlocked


def limit(value, name):
    if value is None or value == '':
        return None
    if type(value) is bool or not str(value).isdigit():
        raise RecipeBlocked("invalid_cost_limit", f"{name} must be a nonnegative integer.")
    return int(value)


def receipt(headers):
    headers = {k.lower(): v for k, v in headers.items()}
    value = headers.get("x-charged-tokens")
    return int(value) if headers.get("x-charge-state") == "settled" and str(value).isdigit() else None


def for_run(context, options):
    if context['mode'] != 'live':
        return None
    client = context.get('client')
    if client is None:
        raise RecipeBlocked('no_client', 'Live mode needs REACHDOG_API_URL and REACHDOG_API_KEY.')
    budget = getattr(client, '_recipe_budget', None)
    if budget is None:
        budget = Budget(client, max_credits=options.get('max_credits'), max_calls=options.get('max_calls'))
        client._recipe_budget = budget
    elif (budget.max_credits, budget.max_calls) != (limit(options.get('max_credits'), 'max_credits'), limit(options.get('max_calls'), 'max_calls')):
        raise RecipeBlocked('cost_limit_changed', 'Use a new client/run to change the spending limits.')
    return budget


class Budget:
    def __init__(self, client, *, max_credits=None, max_calls=None):
        self.client = client
        self.max_credits = limit(max_credits, 'max_credits')
        self.max_calls = limit(max_calls, 'max_calls')
        self.reserved = 0
        self.calls = 0
        self.price = None

    def preview(self, plans, echo):
        pending = [p for p in plans if not p['reuse']]
        if not pending:
            return {'estimated_credits': 0, 'live_calls': 0, 'reused_calls': len(plans)}
        # This is read-only. Missing pricing never falls back to a guessed tariff.
        try:
            usage = self.client.usage().body
            price = usage.get('parameter_price') if isinstance(usage, dict) else None
        except ApiError as exc:
            raise RecipeBlocked('price_unavailable', 'Cannot read the current parameter price; no paid call was made in this stage.') from exc
        if type(price) not in (int, float) or not math.isfinite(price) or price < 0:
            raise RecipeBlocked('price_unavailable', 'The API did not return a valid parameter price.')
        self.price = price
        amount = sum(len(p['body']['parameters']) * price for p in pending)
        estimate = {'estimated_credits': amount, 'parameter_price': price, 'live_calls': len(pending),
                    'reused_calls': len(plans) - len(pending), 'previously_reserved_credits': self.reserved,
                    'max_credits': self.max_credits, 'max_calls': self.max_calls,
                    'basis': 'Current /v1/usage price; estimate, not a locked server quote. Later stages are checked separately.'}
        echo(f"Cost preview: {len(pending)} live calls, {amount:g} credits at {price:g} per parameter; "
             f"{len(plans) - len(pending)} reused captures. Earlier calls reserved {self.reserved:g} credits.")
        self.check(len(pending), amount)
        return estimate

    def check(self, count, amount):
        if self.max_calls is not None and self.calls + count > self.max_calls:
            raise RecipeBlocked('call_limit', f'This stage would exceed the run limit of {self.max_calls} paid calls.')
        if self.max_credits is not None and self.reserved + amount > self.max_credits:
            raise RecipeBlocked('credit_limit', f'This stage would exceed the run limit of {self.max_credits} estimated credits.')

    def reserve(self, plan):
        amount = len(plan['body']['parameters']) * self.price
        self.check(1, amount)
        # Keep the allowance reserved even on an ambiguous failure. A lost response
        # is not proof of no charge. Transport retries keep the existing request key.
        self.calls += 1
        self.reserved += amount

    def catalog_quote(self, amount, echo):
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise RecipeBlocked('price_unavailable', 'The catalog quote has no valid amount.')
        echo(f'Catalog quote: {amount:g} credits; run limit {self.max_credits}.')
        self.check(1, amount)


def record_totals(manifest, rows):
    live = [r for r in rows if r['source'] == 'live']
    amounts = [r.get('charged_credits') for r in live]
    total = sum(amounts) if all(v is not None for v in amounts) else None
    titles = [r for r in live if r.get('operation') is None and r['status'] == 'returned']
    summary = (f"settled {len(titles)} title call(s), {len(titles[0]['request']['parameters'])} parameters each; "
               if titles else 'no live title call returned; ')
    for operation in ('research', 'content'):
        count = sum(r.get('operation') == operation and r['status'] == 'returned' for r in live)
        if count:
            summary += f'{operation}: {count} settled call(s); '
    manifest.charged(total, None, summary + 'Settlement receipt totals from response headers; server replays can include an earlier charge. '
                     'Local captures add zero. Missing receipts and failed requests leave the total unknown; this is not a wallet balance delta.')
