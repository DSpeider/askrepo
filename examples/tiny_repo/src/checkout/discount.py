"""Cart reductions applied before totals are finalized."""


def apply_discount(amount: float, percent: float) -> float:
    """Reduce a cart amount by a fraction. 0.15 means fifteen percent off."""
    return amount * (1.0 - percent)
