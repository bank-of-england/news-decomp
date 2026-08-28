"""Pandera schema for the ``decompositions`` data contract.

See ``news_decomp.md`` for the complete specification.
"""

import numpy as np
import pandera.pandas as pa
from pandera.pandas import Check, Column

REVISION_SOURCES = ["news", "reestimation", "interaction"]

decomposition_schema = pa.DataFrameSchema(
    {
        "variable": Column(str),
        "date": Column("datetime64[ns]"),
        "forecast_horizon": Column(int),
        "frequency": Column(str, Check.isin(["Q", "M"])),
        "source": Column(str),
        "vintage_date": Column("datetime64[ns]"),
        "base_vintage_date": Column("datetime64[ns]", nullable=True),
        "decomposition": Column(str, Check.isin(["level", "revision"])),
        "component": Column(str),
        "revision_source": Column(str, Check.isin(REVISION_SOURCES), nullable=True),
        "contribution": Column(float, nullable=False),
        "weight": Column(float, nullable=True),
        "news": Column(float, nullable=True),
        "forecast_metric": Column(str),
    },
    strict=True,
    checks=[
        # Keep the decomposition flag consistent with revision metadata.
        Check(
            lambda df: (
                (df["decomposition"] == "level") == df["base_vintage_date"].isna()
            ),
            error="decomposition must be 'level' iff base_vintage_date is NaT",
        ),
        Check(
            lambda df: (
                (df["decomposition"] == "revision") == df["revision_source"].notna()
            ),
            error="revision_source must be set iff decomposition is 'revision'",
        ),
        # When both factors exist, they must reproduce the contribution.
        Check(
            lambda df: (
                df[["weight", "news"]].isna().any(axis=1)
                | np.isclose(df["weight"] * df["news"], df["contribution"], atol=1e-8)
            ),
            error="contribution must equal weight * news where both are provided",
        ),
    ],
)
