"""Lookup of adapters by name so CLI/API/eval can say `--dataset nfip`."""

from __future__ import annotations

from loom.data.adapters.base import DataSourceAdapter


def list_adapters() -> dict[str, type[DataSourceAdapter]]:
    # Imported lazily so importing the registry never pulls in requests/openpyxl.
    from loom.data.adapters.nfip import NFIPClaimsAdapter
    from loom.data.adapters.toronto_ttc import TorontoTTCAdapter

    return {TorontoTTCAdapter.name: TorontoTTCAdapter, NFIPClaimsAdapter.name: NFIPClaimsAdapter}


class UnknownDatasetError(KeyError):
    pass


def get_adapter(name: str) -> DataSourceAdapter:
    adapters = list_adapters()
    # A typo in --dataset must fail here, not as a DuckDB "table not found" later.
    if name not in adapters:
        raise UnknownDatasetError(f"unknown dataset {name!r}; known: {sorted(adapters)}")
    return adapters[name]()
