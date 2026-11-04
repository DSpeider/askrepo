"""Invoice totals for the tiny sample."""

from billing.tax import apply_tax


def calculate_invoice_total(lines: list[int], rate: float = 0.0) -> float:
    """Sum line items, then apply the given tax rate."""
    subtotal = sum(lines)
    if rate:
        return apply_tax(subtotal, rate)
    return float(subtotal)
