# Copyright 2026 LAPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core numerical operations for LAPO process supervision.

This module deliberately contains no trainer or distributed-runtime dependencies.  Keeping
the equations here makes the implementation easy to audit against Section 3.3 of the paper
and easy to test in isolation.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np


def replace_token_span(
    tokens: Sequence[int], start: int, end: int, placeholder: Sequence[int]
) -> list:
    """Replace one complete turn while preserving every downstream token."""

    if not 0 <= start <= end <= len(tokens):
        raise ValueError("invalid token span")
    return list(tokens[:start]) + list(placeholder) + list(tokens[end:])


@dataclass(frozen=True)
class ProcessAdvantages:
    """Intermediate and final values from LAPO process-advantage construction."""

    scales: np.ndarray
    bounded: list[np.ndarray]
    normalized: list[np.ndarray]
    gated: list[np.ndarray]


def _as_float_rows(rows: Sequence[Iterable[float]]) -> list[np.ndarray]:
    return [np.asarray(list(row), dtype=np.float32) for row in rows]


def standardize_outcomes(
    rewards: Sequence[float], group_ids: Sequence[Any], epsilon: float
) -> np.ndarray:
    """Compute the within-question GRPO outcome advantage (paper Eq. 8)."""

    rewards_array = np.asarray(rewards, dtype=np.float32)
    groups = np.asarray(group_ids, dtype=object)
    if rewards_array.shape[0] != groups.shape[0]:
        raise ValueError("rewards and group_ids must have the same length")

    advantages = np.zeros_like(rewards_array)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        values = rewards_array[indices]
        advantages[indices] = (values - values.mean()) / (values.std() + epsilon)
    return advantages


def build_process_advantages(
    raw_gain_rows: Sequence[Iterable[float]],
    group_ids: Sequence[Any],
    epsilon: float,
    eligible_rows: Optional[Sequence[Iterable[bool]]] = None,
) -> ProcessAdvantages:
    """Apply robust scaling, tanh, group normalization, and sign gating.

    The grouping spans every valid trajectory-turn pair sampled for the same question,
    exactly as specified by paper Eqs. 9--13.  Rows retain their original turn alignment;
    non-search and final-answer turns should be represented by a zero raw gain.
    """

    raw_rows = _as_float_rows(raw_gain_rows)
    groups = np.asarray(group_ids, dtype=object)
    if len(raw_rows) != groups.shape[0]:
        raise ValueError("raw_gain_rows and group_ids must have the same length")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if eligible_rows is None:
        eligible = [np.ones(row.shape, dtype=bool) for row in raw_rows]
    else:
        eligible = [np.asarray(list(row), dtype=bool) for row in eligible_rows]
        if len(eligible) != len(raw_rows) or any(
            mask.shape != row.shape for mask, row in zip(eligible, raw_rows)
        ):
            raise ValueError("eligible_rows must match raw_gain_rows")

    scales = np.full(len(raw_rows), epsilon, dtype=np.float32)
    bounded = [np.zeros_like(row) for row in raw_rows]
    normalized = [np.zeros_like(row) for row in raw_rows]
    gated = [np.zeros_like(row) for row in raw_rows]

    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        absolute_gains = np.concatenate(
            [np.abs(raw_rows[index][eligible[index]]) for index in indices if eligible[index].any()]
        ) if indices.size else np.empty(0, dtype=np.float32)
        nonzero_gains = absolute_gains[absolute_gains > epsilon]
        scale = float(np.median(nonzero_gains) + epsilon) if nonzero_gains.size else epsilon

        for index in indices:
            scales[index] = scale
            bounded[index][eligible[index]] = np.tanh(
                raw_rows[index][eligible[index]] / scale
            ).astype(np.float32)

        group_bounded = np.concatenate(
            [bounded[index][eligible[index]] for index in indices if eligible[index].any()]
        ) if indices.size else np.empty(0, dtype=np.float32)
        mean = float(group_bounded.mean()) if group_bounded.size else 0.0
        std = float(group_bounded.std()) if group_bounded.size else 0.0

        for index in indices:
            normalized[index][eligible[index]] = (
                (bounded[index][eligible[index]] - mean) / (std + epsilon)
            ).astype(np.float32)
            # Eq. 12 uses strict directional agreement.  epsilon belongs to robust
            # scaling, not to the gate itself.
            gate = eligible[index] & (raw_rows[index] * normalized[index] > 0.0)
            gated[index] = np.where(gate, normalized[index], 0.0).astype(np.float32)

    return ProcessAdvantages(
        scales=scales,
        bounded=bounded,
        normalized=normalized,
        gated=gated,
    )


def add_process_advantage(
    outcome_advantage: float, process_advantage: np.ndarray, weight: float
) -> np.ndarray:
    """Combine outcome and auxiliary process advantages (paper Eq. 14)."""

    return np.asarray(outcome_advantage + weight * process_advantage, dtype=np.float32)
