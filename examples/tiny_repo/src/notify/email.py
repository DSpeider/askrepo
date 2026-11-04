"""Outbound receipt messages."""

from notify.receipt import format_receipt


def send_receipt(order_id: str, total: float, address: str) -> str:
    """Build the message sent after payment."""
    body = format_receipt(order_id, total)
    return f"To: {address}\n{body}"
