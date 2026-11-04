"""Plain-text receipt rendering."""


def format_receipt(order_id: str, total: float) -> str:
    """Render receipt lines. Delivery is a different module."""
    return f"Order {order_id} total {total:.2f}"
