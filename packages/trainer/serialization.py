"""Canonical serialization and hashing for experiment specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def importable_type_path(value: type[Any]) -> str:
    if not isinstance(value,type):raise TypeError("configuration factory must be a class")
    module=getattr(value,"__module__","");qualname=getattr(value,"__qualname__","")
    if not module or not qualname or "<locals>" in qualname or not all(part.isidentifier() for part in (*module.split("."),*qualname.split("."))):
        raise ValueError("configuration factory must be a top-level importable class")
    return f"{module}:{qualname}"


def canonical_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        result={}
        for item in fields(value):
            if item.metadata.get("canonical_exclude"):
                continue
            field_value=getattr(value,item.name)
            if item.metadata.get("canonical_omit_default"):
                default = item.default
                if default is MISSING and item.default_factory is not MISSING:
                    default = item.default_factory()
                if default is not MISSING and field_value == default:
                    continue
            if item.metadata.get("canonical_mapping"):
                field_value=dict(field_value)
            result[item.name]=canonical_data(field_value)
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value,type):
        return importable_type_path(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("configuration mappings must use string keys")
        return {key: canonical_data(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [canonical_data(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"configuration value {value!r} is not serializable")


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(canonical_data(value), sort_keys=True, separators=None if indent else (",", ":"), indent=indent)


def config_hash(value: Any, length: int = 12) -> str:
    if length <= 0:
        raise ValueError("hash length must be positive")
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]
