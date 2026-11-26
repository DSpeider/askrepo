"""HTTP entry for the sample checkout flow."""

from checkout.order import place_order


def handle_checkout(lines: list[int], rate: float) -> float:
    """Shopper-facing gateway. Accept the cart and hand it off."""
    return place_order(lines, rate)
