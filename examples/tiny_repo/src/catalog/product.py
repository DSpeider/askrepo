"""Catalog lookup for the tiny sample."""


def lookup_sku(sku: str) -> str:
    """Return the shelf name for a sku. Prices live elsewhere."""
    names = {"sku-1": "Notebook", "sku-2": "Pencil"}
    return names.get(sku, "Unknown item")
