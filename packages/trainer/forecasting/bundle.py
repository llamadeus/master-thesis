"""Forecasting bundle contract validation."""

from __future__ import annotations

import json
from pathlib import Path


def load_bundle_manifest(path: Path) -> dict:
    raw = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    validate_bundle_manifest(raw)
    return raw


def validate_bundle_manifest(raw: dict, *, expected: dict | None = None) -> None:
    required = ("model_kwargs", "node_ids", "source_ids", "hydraulic_timestep_seconds",
                "history_feature_contract", "future_known_feature_contract", "heads")
    missing = [name for name in required if name not in raw]
    if missing: raise ValueError(f"forecast bundle is missing contracts: {missing}")
    names = [head.get("name") for head in raw["heads"]]
    bundle_kind = raw.get("kind")
    headless = bundle_kind == "next-latent-lstm-encoder"
    auxiliary_next_latent = bundle_kind == "auxiliary-next-latent-lstm-encoder"
    if ((not names and not headless) or len(set(names)) != len(names)):
        raise ValueError("forecast bundle head names must be non-empty and unique")
    if len(set(raw["node_ids"])) != len(raw["node_ids"]) or len(set(raw["source_ids"])) != len(raw["source_ids"]):
        raise ValueError("forecast bundle spatial ordering contains duplicates")
    if raw.get("kind") == "latent-dynamics-forecast":
        latent = raw.get("latent_contract")
        if not isinstance(latent, dict):
            raise ValueError("latent-dynamics forecast bundle is missing its latent contract")
        if (latent.get("name") != "z_t" or latent.get("source") != "encoder-top-hidden-state" or
                not isinstance(latent.get("size"), int) or latent["size"] <= 0 or
                not isinstance(latent.get("history_steps"), int) or latent["history_steps"] <= 0):
            raise ValueError("latent-dynamics forecast bundle has an invalid latent contract")
    if headless:
        latent = raw.get("latent_contract")
        if names:
            raise ValueError("next-latent encoder bundle must not contain forecast heads")
        if (not isinstance(latent, dict) or latent.get("name") != "z_t" or
                latent.get("source") != "layer-normalized-encoder-top-hidden-state" or
                not isinstance(latent.get("size"), int) or latent["size"] <= 0 or
                not isinstance(latent.get("history_steps"), int) or
                latent["history_steps"] <= 0):
            raise ValueError("next-latent encoder bundle has an invalid latent contract")
    if auxiliary_next_latent:
        latent = raw.get("latent_contract")
        if names != ["chlorine"]:
            raise ValueError(
                "auxiliary next-latent encoder bundle requires one chlorine head"
            )
        if (not isinstance(latent, dict) or latent.get("name") != "z_t" or
                latent.get("source") != "layer-normalized-encoder-top-hidden-state" or
                not isinstance(latent.get("size"), int) or latent["size"] <= 0 or
                not isinstance(latent.get("history_steps"), int) or
                latent["history_steps"] <= 0):
            raise ValueError(
                "auxiliary next-latent encoder bundle has an invalid latent contract"
            )
    if expected:
        for key, value in expected.items():
            if raw.get(key) != value: raise ValueError(f"forecast bundle contract mismatch: {key}")
