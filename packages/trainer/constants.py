"""Shared physical and operational constants for trainer configurations."""

MAX_CHLORINE_INJECTION = 500_000.0

# Hanoi junctions that must never be selected as arsenic event sources.
ARSENIC_EVENT_EXCLUDED_NODE_IDS = frozenset({"13", "22"})

# Current V70 actuator contract.
V70_MAX_CHLORINE_INJECTION = 500_000.0
V70_MAX_CHLORINE_DELTA = 50_000.0
