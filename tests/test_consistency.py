"""Test the internal identities used by ``NewsData.report()``.

The tests check two identities for every ``(variable, source, date)`` triple:

1. **Level contributions sum to the nowcast**: stacked-bar values must sum to
    the nowcast at every vintage.

2. **Revision contributions equal consecutive nowcast differences**: revision
    values for vintage k must equal ``nowcast(k) - nowcast(k-1)``.

The fixture uses ``news_decomp.sample.simulate()``, which generates a
self-consistent synthetic dataset. Both identities must hold within the
floating-point tolerance.
"""

import pandas as pd
import pytest

import news_decomp as nd
from news_decomp.sample import SEED, simulate

TOL = 1e-8


@pytest.fixture(scope="module")
def news_data() -> nd.NewsData:
    return nd.NewsData(simulate()["decompositions"])


def _level_sum_per_vintage(
    data: nd.NewsData, variable: str, source: str, date: pd.Timestamp
) -> pd.Series:
    """Sum level contributions by vintage for one variable-source-date group."""
    lev = data._level(variable=variable, source=source)
    return (
        lev[lev["date"] == date]
        .groupby("vintage_date")["contribution"]
        .sum()
        .sort_index()
    )


def _revision_sum_per_vintage(
    data: nd.NewsData, variable: str, source: str, date: pd.Timestamp
) -> pd.Series:
    """Sum all revision contributions by vintage for one group."""
    rev = data._revision(variable=variable, source=source)
    return (
        rev[rev["date"] == date]
        .groupby("vintage_date")["contribution"]
        .sum()
        .sort_index()
    )


def _all_combos(data: nd.NewsData):
    """Yield each variable-source-date group present in the frame."""
    df = data.df
    keys = df[["variable", "source", "date"]].drop_duplicates()
    for _, row in keys.iterrows():
        yield row["variable"], row["source"], pd.Timestamp(row["date"])


def test_default_seed_matches_explicit_seed():
    """Confirm that the default simulation uses the canonical seed."""
    default = simulate()
    explicit = simulate(seed=SEED)

    pd.testing.assert_frame_equal(default["truth"], explicit["truth"])
    pd.testing.assert_frame_equal(default["releases"], explicit["releases"])


def test_last_imputation_falls_back_when_no_prior_release_exists():
    """Confirm that ``last`` fills vintages without an earlier release."""
    zero = simulate(seed=42, x_imputation="zero")["nowcasts"]
    last = simulate(seed=42, x_imputation="last")["nowcasts"]

    assert len(last) == len(zero)
    assert last[["X1_latest", "X2_latest", "y_nowcast"]].notna().all().all()


# Level contributions must equal the nowcast.


class TestLevelContributionsSumToNowcast:
    """Ensure level contributions match the nowcast at every vintage."""

    def test_all_combos(self, news_data: nd.NewsData):
        failures = []
        for variable, source, date in _all_combos(news_data):
            contrib_sum = _level_sum_per_vintage(news_data, variable, source, date)
            if contrib_sum.empty:
                continue

            evolution = news_data.nowcast_evolution(
                date=date, variable=variable, source=source
            ).set_index("vintage_date")["nowcast"]

            common = contrib_sum.index.intersection(evolution.index)
            if common.empty:
                continue

            diff = (contrib_sum.loc[common] - evolution.loc[common]).abs()
            bad = diff[diff > TOL]
            for vd, val in bad.items():
                failures.append(
                    f"variable={variable!r} source={source!r} "
                    f"date={date.date()} vintage={pd.Timestamp(vd).date()}: "
                    f"contrib_sum={contrib_sum[vd]:.6f}, "
                    f"nowcast={evolution[vd]:.6f}, diff={val:.2e}"
                )

        assert not failures, (
            f"{len(failures)} vintage(s) where level contributions ≠ nowcast:\n"
            + "\n".join(failures)
        )

    @pytest.mark.parametrize("vintage_idx", [0, -1])
    def test_first_and_last_vintage(self, news_data: nd.NewsData, vintage_idx: int):
        """Check the first and last vintage in the default data slice."""
        variable = news_data._validate_variable(None)
        source = news_data._validate_source(None)
        date = news_data._validate_date(None)

        contrib_sum = _level_sum_per_vintage(news_data, variable, source, date)
        evolution = news_data.nowcast_evolution(
            date=date, variable=variable, source=source
        ).set_index("vintage_date")["nowcast"]

        common = sorted(contrib_sum.index.intersection(evolution.index))
        assert common, "No overlapping vintages between level and evolution data."

        vd = common[vintage_idx]
        diff = abs(contrib_sum[vd] - evolution[vd])
        assert diff < TOL, (
            f"vintage={pd.Timestamp(vd).date()}: "
            f"contrib_sum={contrib_sum[vd]:.6f}, nowcast={evolution[vd]:.6f}, "
            f"diff={diff:.2e}"
        )


# Revision contributions must equal consecutive nowcast differences.


class TestRevisionContributionsEqualNowcastDiffs:
    """Ensure revisions match consecutive nowcast differences."""

    def test_all_combos(self, news_data: nd.NewsData):
        failures = []
        for variable, source, date in _all_combos(news_data):
            rev_sum = _revision_sum_per_vintage(news_data, variable, source, date)
            if rev_sum.empty:
                continue

            evolution = (
                news_data.nowcast_evolution(date=date, variable=variable, source=source)
                .set_index("vintage_date")["nowcast"]
                .sort_index()
            )
            if len(evolution) < 2:
                continue

            nowcast_diff = evolution.diff().dropna()

            common = rev_sum.index.intersection(nowcast_diff.index)
            if common.empty:
                continue

            diff = (rev_sum.loc[common] - nowcast_diff.loc[common]).abs()
            bad = diff[diff > TOL]
            for vd, val in bad.items():
                failures.append(
                    f"variable={variable!r} source={source!r} "
                    f"date={date.date()} vintage={pd.Timestamp(vd).date()}: "
                    f"rev_sum={rev_sum[vd]:.6f}, "
                    f"nowcast_diff={nowcast_diff[vd]:.6f}, diff={val:.2e}"
                )

        assert not failures, (
            f"{len(failures)} vintage(s) where revision contributions ≠ Δnowcast:\n"
            + "\n".join(failures)
        )

    def test_cumulative_revision_spans_total_change(self, news_data: nd.NewsData):
        """Ensure cumulative revisions equal the total nowcast change."""
        variable = news_data._validate_variable(None)
        source = news_data._validate_source(None)
        date = news_data._validate_date(None)

        rev_sum = _revision_sum_per_vintage(news_data, variable, source, date)
        evolution = (
            news_data.nowcast_evolution(date=date, variable=variable, source=source)
            .set_index("vintage_date")["nowcast"]
            .sort_index()
        )

        if rev_sum.empty or len(evolution) < 2:
            pytest.skip("Insufficient data for cumulative revision check.")

        total_revision = evolution.iloc[-1] - evolution.iloc[0]
        cumulative_news = rev_sum.sum()
        diff = abs(cumulative_news - total_revision)

        assert diff < TOL, (
            f"Cumulative news revisions ({cumulative_news:.6f}) ≠ "
            f"total nowcast change ({total_revision:.6f}), diff={diff:.2e}"
        )
