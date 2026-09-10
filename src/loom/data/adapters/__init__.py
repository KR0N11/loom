"""One adapter per data source (Adapter pattern)."""

from loom.data.adapters.base import DataSourceAdapter
from loom.data.adapters.registry import get_adapter, list_adapters

__all__ = ["DataSourceAdapter", "get_adapter", "list_adapters"]
