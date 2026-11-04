"""Tax helper for the tiny sample."""


def apply_tax(amount: float, rate: float) -> float:
    """Return amount plus amount * rate.

    A rate of 0.2 on 100 yields 120.
    """
    return amount + (amount * rate)
