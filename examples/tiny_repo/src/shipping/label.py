"""Warehouse label text."""


def render_label(order_id: str, city: str) -> str:
    """Format a warehouse label. Does not price or tax the order."""
    return f"{order_id} -> {city}"
