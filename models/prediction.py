"""The prediction shape. Fixed from v1 so downstream consumers never change.

v1 populates ``mean`` only and leaves the quantiles ``None`` (not faked with a
heuristic). v2's quantile model fills ``p10`` / ``p50`` / ``p90``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Prediction:
    mean: float
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None

    def as_dict(self) -> dict:
        return {"mean": self.mean, "p10": self.p10, "p50": self.p50, "p90": self.p90}


def predictions_frame(means, index=None) -> pd.DataFrame:
    """Vector of means -> a DataFrame with the full prediction schema (quantiles NaN)."""
    return pd.DataFrame(
        {"pred_mean": means, "pred_p10": pd.NA, "pred_p50": pd.NA, "pred_p90": pd.NA},
        index=index,
    )
