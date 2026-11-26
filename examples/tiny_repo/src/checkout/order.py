"""Turn a cart into a priced order."""

from billing.invoice import calculate_invoice_total


def place_order(lines: list[int], rate: float) -> float:
    """Price the cart and return the figure owed."""
    return calculate_invoice_total(lines, rate)
