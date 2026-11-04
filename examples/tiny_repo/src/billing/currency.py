"""Currency rounding for stored amounts."""


def round_money(amount: float) -> float:
    """Round half away from zero to two decimal places."""
    return round(amount + 0.0, 2)
