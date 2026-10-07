"""Calendar-time helpers shared by forecasting and control features."""

from __future__ import annotations

import numpy as np


SECONDS_PER_DAY = 24 * 3600
DAYS_PER_YEAR = 365


def calendar_time_seconds(
    episode_time_seconds,
    calendar_start_day: int,
):
    """Return continuous calendar seconds without wrapping the weekly phase."""
    values = np.asarray(episode_time_seconds, dtype=np.float64)
    return values + calendar_start_day * SECONDS_PER_DAY


def cyclic_time_features(
    episode_time_seconds,
    calendar_start_day: int,
) -> np.ndarray:
    """Encode daily, weekly, and yearly phase from continuous calendar time."""
    values = calendar_time_seconds(episode_time_seconds, calendar_start_day)
    return cyclic_time_features_from_time_of_year(values)


def cyclic_time_features_from_time_of_year(time_of_year_seconds) -> np.ndarray:
    """Encode each cyclic phase independently from continuous seconds."""
    values = np.asarray(time_of_year_seconds, dtype=np.float64)
    periods = (SECONDS_PER_DAY, 7 * SECONDS_PER_DAY, DAYS_PER_YEAR * SECONDS_PER_DAY)
    columns = []
    for period in periods:
        angle = 2 * np.pi * values / period
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.stack(columns, axis=-1).astype(np.float32)
