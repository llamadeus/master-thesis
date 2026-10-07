from __future__ import annotations

import numpy as np

from packages.trainer import RewardInput


def reward_input(
    chlorine: list[float],
    *,
    injection: float = 0.0,
    raw_action: float = 0.0,
    step: int = 49,
) -> RewardInput:
    count = len(chlorine)
    zeros = np.zeros(count, dtype=np.float32)
    return RewardInput(
        nodes_quality=np.asarray(chlorine, dtype=np.float32),
        contaminant_values=zeros,
        pressure=zeros,
        demand=zeros,
        flow=np.zeros(1, dtype=np.float32),
        current_injection=injection,
        raw_action=raw_action,
        step=step,
        current_time=step * 1_800,
        node_ids=tuple(str(index) for index in range(count)),
        link_ids=("L",),
        event_active=False,
        event_strength_by_node=zeros,
    )

