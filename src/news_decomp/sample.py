"""Simulate a real-time dataset for testing and examples.

The simulation uses a simple OLS process with two quarterly regressors::

    y = const + beta_x1 * X1 + beta_x2 * X2 + eps

The target ``y`` is released at the end of each quarter. The two regressors
remain quarterly quantities, but their releases arrive more often:

* ``X1`` is released **monthly**   (3 releases per quarter)
* ``X2`` is released **weekly**    (~13 releases per quarter)

Each release combines the true value with measurement noise. The noise standard
deviation **shrinks over the quarter**, from ``NOISE_STD_START`` to
``NOISE_STD_END``::

    X_release = X_true + N(0, sigma(frac))
    sigma(frac) = std_start + (std_end - std_start) * frac      # frac in (0, 1]

The truth begins in 1980, which gives OLS enough history for precise estimates.
Only the last ``N_REALTIME`` quarters receive real-time release vintages.

The OLS model is **re-estimated at every vintage** with the quarters whose
``y`` value is available. Because ``y`` arrives after a lag of
``Y_PUBLICATION_LAG``, a prior quarter can change the coefficients during the
current quarter's nowcast window. The simulation then computes a real-time
**nowcast** from the latest regressor releases and the current coefficients.
As releases become more precise, the nowcast error usually shrinks. The change
between consecutive vintages splits into **news**, **reestimation**, and an
**interaction** cross-term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Use a fixed default so examples and tests reproduce the same sample.
SEED = 0

# Data-generating process.
CONST = 1.0
BETA_X1 = 0.5
BETA_X2 = -0.3
SIGMA_Y = 0.05  # Keep residual noise small enough to show within-quarter changes.

# Real-time release schedule.
START = "1980Q1"
END = "2025Q4"
N_REALTIME = 8  # Number of recent quarters with real-time vintages.

MONTHLY_RELEASES = 3  # Number of X1 releases per quarter.
WEEKLY_RELEASES = 13  # Number of X2 releases per quarter.

NOISE_STD_START = 0.5  # Measurement-noise standard deviation at quarter start.
NOISE_STD_END = 0.001  # Measurement-noise standard deviation at quarter end.

# Publish y after this lag, then revise it at each later regressor release.
# These target revisions make the OLS coefficients change at each vintage.
Y_PUBLICATION_LAG = pd.Timedelta(days=50)
Y_REVISION_NOISE_START = 0.30  # Initial standard deviation of y revision error.
Y_REVISION_DECAY = 0.6  # Multiplicative decay in revision error at later vintages.
# Number of recent published quarters used for each estimate.
# Use None to include all available quarters.
ESTIMATION_WINDOW = None


def _noise_std(frac: np.ndarray | float) -> np.ndarray | float:
    """Return measurement-noise standard deviation for elapsed fraction."""
    return NOISE_STD_START + (NOISE_STD_END - NOISE_STD_START) * frac


def simulate(
    seed: int = SEED,
    sigma_y: float = SIGMA_Y,
    x_imputation: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate truth, releases, nowcasts, and decomposition tables.

    Returns
    -------
    dict[str, pd.DataFrame]
        A mapping with these keys:
        ``"truth"``: one row per quarter with final ``date, X1, X2, y`` values.
        ``"releases"``: a long table with ``series, reference_date,
        vintage_date, value`` columns.
        ``"nowcasts"``: one row per quarter and vintage with the latest inputs,
        coefficients, nowcast, truth, and error.
        ``"coefficients"``: the true data-generating coefficients.
        ``"fitted_coefficients"``: OLS coefficients fitted to the full truth
        sample for reference.
        ``"decompositions"``: a schema-valid table of level and revision
        contributions.
        ``"vintages"``: outturn and forecast tables for ``NowcastData``.

    Parameters
    ----------
    seed : int
        Random seed used to generate the sample data.
    sigma_y : float
        Standard deviation of the regression residual.
    x_imputation : str | None
        How to fill a regressor that has **not yet been released** in the
        nowcast quarter. This matches ``forecast_realtime``'s ``X_imputation``:

        - ``None`` (default): leave missing values in place and drop vintages
            where a regressor is unavailable.
        - ``"zero"``: replace each missing regressor with ``0.0``.
        - ``"last"``: carry forward the latest earlier-quarter release. Use
            ``0.0`` when no earlier release exists.
    """
    rng = np.random.default_rng(seed)

    quarters = pd.period_range(START, END, freq="Q")
    n = len(quarters)

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eps = sigma_y * rng.standard_normal(n)
    y = CONST + BETA_X1 * x1 + BETA_X2 * x2 + eps

    quarter_end = quarters.to_timestamp(how="end").normalize()
    truth = pd.DataFrame({"date": quarter_end, "X1": x1, "X2": x2, "y": y})

    releases = _simulate_releases(quarters[-N_REALTIME:], truth, rng)

    fitted = fit_ols(truth)
    nowcasts = _simulate_nowcasts(releases, truth, x_imputation=x_imputation)

    coefficients = pd.DataFrame(
        {"const": [CONST], "beta_x1": [BETA_X1], "beta_x2": [BETA_X2]}
    )
    data = {
        "truth": truth,
        "releases": releases,
        "nowcasts": nowcasts,
        "coefficients": coefficients,
        "fitted_coefficients": fitted,
    }
    data["decompositions"] = build_decompositions(data)
    data["vintages"] = build_vintages(data)
    return data


def _simulate_releases(
    realtime_quarters: pd.PeriodIndex,
    truth: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build the long release table for the selected real-time quarters.

    ``X1`` and ``X2`` arrive within their quarter with shrinking measurement
    noise. ``y`` arrives after the quarter ends and then receives revisions at
    later regressor-release vintages; the revision error shrinks geometrically.
    """
    truth_by_date = truth.set_index("date")
    frames: list[pd.DataFrame] = []

    for period in realtime_quarters:
        q_start = period.to_timestamp(how="start").normalize()
        ref_date = period.to_timestamp(how="end").normalize()
        x1_true = truth_by_date.at[ref_date, "X1"]
        x2_true = truth_by_date.at[ref_date, "X2"]

        # Generate monthly X1 releases with shrinking noise.
        m_dates = pd.date_range(q_start, periods=MONTHLY_RELEASES, freq="ME")
        m_frac = np.arange(1, MONTHLY_RELEASES + 1) / MONTHLY_RELEASES
        x1_obs = x1_true + rng.normal(0.0, _noise_std(m_frac))
        frames.append(
            pd.DataFrame(
                {
                    "series": "X1",
                    "reference_date": ref_date,
                    "vintage_date": m_dates,
                    "value": x1_obs,
                }
            )
        )

        # Generate weekly X2 releases with shrinking noise.
        w_dates = pd.date_range(q_start, periods=WEEKLY_RELEASES, freq="W")
        w_frac = np.arange(1, WEEKLY_RELEASES + 1) / WEEKLY_RELEASES
        x2_obs = x2_true + rng.normal(0.0, _noise_std(w_frac))
        frames.append(
            pd.DataFrame(
                {
                    "series": "X2",
                    "reference_date": ref_date,
                    "vintage_date": w_dates,
                    "value": x2_obs,
                }
            )
        )

    releases = pd.concat(frames, ignore_index=True)

    # Use every regressor release date as a possible vintage.
    vintage_schedule = np.sort(releases["vintage_date"].unique())

    # Publish y after a lag, then revise it at later regressor-release vintages.
    y_frames: list[pd.DataFrame] = []
    for period in realtime_quarters:
        ref_date = period.to_timestamp(how="end").normalize()
        y_true = truth_by_date.at[ref_date, "y"]
        first_pub = ref_date + Y_PUBLICATION_LAG
        rev_vintages = vintage_schedule[vintage_schedule >= first_pub]
        if len(rev_vintages) == 0:
            continue
        stds = Y_REVISION_NOISE_START * Y_REVISION_DECAY ** np.arange(len(rev_vintages))
        y_obs = y_true + rng.normal(0.0, stds)
        y_obs[-1] = y_true  # Set the latest vintage to the final truth.
        y_frames.append(
            pd.DataFrame(
                {
                    "series": "y",
                    "reference_date": ref_date,
                    "vintage_date": rev_vintages,
                    "value": y_obs,
                }
            )
        )

    releases = pd.concat([releases, *y_frames], ignore_index=True)
    return releases.sort_values(
        ["reference_date", "series", "vintage_date"]
    ).reset_index(drop=True)


def fit_ols(truth: pd.DataFrame) -> pd.Series:
    """Fit OLS to the final truth values and return the coefficients."""
    design = np.column_stack(
        [np.ones(len(truth)), truth["X1"].to_numpy(), truth["X2"].to_numpy()]
    )
    coef, *_ = np.linalg.lstsq(design, truth["y"].to_numpy(), rcond=None)
    return pd.Series(coef, index=["const", "beta_x1", "beta_x2"])


def _make_fitter(truth: pd.DataFrame, releases: pd.DataFrame):
    """Return a function that estimates OLS coefficients as of each vintage.

    Use the most recent ``ESTIMATION_WINDOW`` quarters whose ``y`` is available
    at the vintage. Real-time ``y`` values change at later vintages, so the fit
    can change at every step. Older history uses the final truth. Cache each
    fit by its available data.
    """
    t = truth.sort_values("date").reset_index(drop=True)
    y_rel = releases[releases["series"] == "y"][
        ["reference_date", "vintage_date", "value"]
    ]
    y_by_q = {
        ref: g.sort_values("vintage_date") for ref, g in y_rel.groupby("reference_date")
    }
    first_pub = {ref: g["vintage_date"].min() for ref, g in y_by_q.items()}
    first_realtime = min(first_pub) if first_pub else None
    cache: dict[tuple, pd.Series] = {}

    def _asof_y(ref, vintage):
        g = y_by_q[ref]
        vals = g[g["vintage_date"] <= vintage]
        return vals["value"].iloc[-1], vals["vintage_date"].iloc[-1]

    def coef_asof(vintage: pd.Timestamp) -> pd.Series | None:
        published = [ref for ref, pub in first_pub.items() if pub <= vintage]
        if published:
            last_pub = max(published)
        elif first_realtime is not None:
            last_pub = t.loc[t["date"] < first_realtime, "date"].max()
        else:
            last_pub = t["date"].max()
        if pd.isna(last_pub):
            return None

        if ESTIMATION_WINDOW is None:
            sample = t[t["date"] <= last_pub].copy()
        else:
            sample = t[t["date"] <= last_pub].tail(ESTIMATION_WINDOW).copy()

        # Replace real-time y values with their latest values at this vintage.
        key_parts = []
        y_values = sample["y"].to_numpy(copy=True)
        for i, ref in enumerate(sample["date"].to_numpy()):
            ref = pd.Timestamp(ref)
            if ref in y_by_q and first_pub[ref] <= vintage:
                val, vtg = _asof_y(ref, vintage)
                y_values[i] = val
                key_parts.append((ref, vtg))
        sample_y = y_values
        key = (last_pub, tuple(key_parts))

        if key not in cache:
            design = np.column_stack(
                [np.ones(len(sample)), sample["X1"].to_numpy(), sample["X2"].to_numpy()]
            )
            beta, *_ = np.linalg.lstsq(design, sample_y, rcond=None)
            cache[key] = pd.Series(beta, index=["const", "beta_x1", "beta_x2"])
        return cache[key]

    return coef_asof


def _simulate_nowcasts(
    releases: pd.DataFrame,
    truth: pd.DataFrame,
    x_imputation: str | None = None,
) -> pd.DataFrame:
    """Compute a real-time nowcast of ``y`` at every within-quarter vintage.

    At each vintage the model is re-estimated (see :func:`_make_fitter`) and the
    nowcast uses the most recent release of each regressor available so far::

        y_nowcast = const(v) + beta_x1(v) * X1_latest + beta_x2(v) * X2_latest

    Return the coefficients with each nowcast so the revision decomposition can
    separate data changes from coefficient changes.

    ``x_imputation`` controls a regressor that has **not yet been released** in
    the quarter: ``None`` drops the vintage, ``"zero"`` uses ``0.0``, and
    ``"last"`` carries forward the latest earlier-quarter value, using ``0.0``
    when no earlier release exists.
    """
    if x_imputation not in (None, "zero", "last"):
        raise ValueError(
            f"x_imputation must be None or one of 'zero', 'last'; got {x_imputation!r}"
        )

    coef_asof = _make_fitter(truth, releases)
    truth_by_date = truth.set_index("date")
    regressors = releases[releases["series"].isin(["X1", "X2"])]

    # Keep release history for cross-quarter "last" filling.
    prior_by_series = {
        series: grp[["reference_date", "vintage_date", "value"]]
        .sort_values("vintage_date")
        .reset_index(drop=True)
        for series, grp in regressors.groupby("series")
    }

    def _last_prior(series: str, ref_date: pd.Timestamp, vintage: pd.Timestamp):
        """Return the latest earlier-quarter release available at ``vintage``."""
        hist = prior_by_series[series]
        avail = hist[
            (hist["reference_date"] < ref_date) & (hist["vintage_date"] <= vintage)
        ]
        if avail.empty:
            return np.nan
        return avail["value"].iloc[-1]

    def _impute(series: str, latest: pd.Series, ref_date: pd.Timestamp) -> pd.Series:
        """Fill unavailable leading values according to ``x_imputation``."""
        if x_imputation is None or not latest.isna().any():
            return latest
        filled = latest.copy()
        missing = filled.index[filled.isna()]
        if x_imputation == "zero":
            filled.loc[missing] = 0.0
        else:  # "last"
            prior = [_last_prior(series, ref_date, v) for v in missing]
            filled.loc[missing] = pd.Series(prior, index=missing).fillna(0.0)
        return filled

    frames: list[pd.DataFrame] = []

    for ref_date, grp in regressors.groupby("reference_date"):
        x1 = grp[grp["series"] == "X1"].set_index("vintage_date")["value"].sort_index()
        x2 = grp[grp["series"] == "X2"].set_index("vintage_date")["value"].sort_index()

        # Evaluate both series on the union of their release dates.
        union = x1.index.union(x2.index)
        x1_latest = _impute("X1", x1.reindex(union).ffill(), ref_date)
        x2_latest = _impute("X2", x2.reindex(union).ffill(), ref_date)

        coefs = [coef_asof(v) for v in union]
        const = np.array([c["const"] if c is not None else np.nan for c in coefs])
        b1 = np.array([c["beta_x1"] if c is not None else np.nan for c in coefs])
        b2 = np.array([c["beta_x2"] if c is not None else np.nan for c in coefs])

        y_nowcast = const + b1 * x1_latest.to_numpy() + b2 * x2_latest.to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "reference_date": ref_date,
                    "vintage_date": union,
                    "X1_latest": x1_latest.to_numpy(),
                    "X2_latest": x2_latest.to_numpy(),
                    "const": const,
                    "beta_x1": b1,
                    "beta_x2": b2,
                    "y_nowcast": y_nowcast,
                    "y_true": truth_by_date.at[ref_date, "y"],
                }
            )
        )

    nowcasts = pd.concat(frames, ignore_index=True).dropna(subset=["y_nowcast"])
    nowcasts["error"] = nowcasts["y_nowcast"] - nowcasts["y_true"]
    return nowcasts.sort_values(["reference_date", "vintage_date"]).reset_index(
        drop=True
    )


# Columns required by the NowcastData outturn and forecast contracts.
_NOWCAST_OUTTURN_COLUMNS = [
    "date",
    "vintage_date",
    "variable",
    "frequency",
    "forecast_horizon",
    "value",
]
_NOWCAST_FORECAST_COLUMNS = [
    "date",
    "vintage_date",
    "variable",
    "frequency",
    "forecast_horizon",
    "value",
    "source",
]

# Columns required by the decompositions data contract.
_DECOMP_COLUMNS = [
    "variable",
    "date",
    "forecast_horizon",
    "frequency",
    "source",
    "vintage_date",
    "base_vintage_date",
    "decomposition",
    "component",
    "revision_source",
    "contribution",
    "weight",
    "news",
    "forecast_metric",
]


def build_decompositions(
    data: dict[str, pd.DataFrame],
    source: str = "ols_nowcast",
) -> pd.DataFrame:
    """Build a schema-valid ``decompositions`` table.

    The nowcast is the linear model ``y = const + b1*X1 + b2*X2`` with
    coefficients **re-estimated at every vintage**. Two kinds of rows are
    produced for the target ``y``:

        * **level** rows: one row per regressor and intercept at each vintage;
            each row records that vintage's nowcast contribution
            (``contribution_i = weight_i * X_i_latest``).
        * **revision** rows: one set between each pair of consecutive vintages.
            When data ``z`` and weights ``w`` both change, split
            ``w1*z1 - w0*z0`` for each component ``i`` as follows::

          news_i        = w0_i * (z1_i - z0_i)        # data changed, old weights
          reestimation_i = (w1_i - w0_i) * z0_i        # weights changed, old data
          interaction_i  = (w1_i - w0_i) * (z1_i - z0_i)  # cross-term

    Only ``news`` rows carry ``weight`` and ``news`` factors. The intercept
    uses the constant value 1 and contributes through re-estimation.
    """
    nowcasts = data["nowcasts"]
    rows: list[dict] = []

    def add(meta, vintage, base, decomp, comp, source_kind, contribution, weight, news):
        if contribution == 0.0:
            return
        rows.append(
            {
                **meta,
                "vintage_date": vintage,
                "base_vintage_date": base,
                "decomposition": decomp,
                "component": comp,
                "revision_source": source_kind,
                "contribution": contribution,
                "weight": weight,
                "news": news,
            }
        )

    for ref_date, grp in nowcasts.groupby("reference_date"):
        grp = grp.sort_values("vintage_date")
        meta = {
            "variable": "y",
            "date": ref_date,
            "forecast_horizon": 0,
            "frequency": "Q",
            "source": source,
            "forecast_metric": "levels",
        }

        # Record each vintage's level decomposition.
        for r in grp.itertuples(index=False):
            w = {"intercept": r.const, "X1": r.beta_x1, "X2": r.beta_x2}
            z = {"intercept": 1.0, "X1": r.X1_latest, "X2": r.X2_latest}
            for comp in ("intercept", "X1", "X2"):
                add(
                    meta,
                    r.vintage_date,
                    pd.NaT,
                    "level",
                    comp,
                    np.nan,
                    w[comp] * z[comp],
                    w[comp],
                    np.nan,
                )

        # Split each change between consecutive vintages.
        prev = None
        for r in grp.itertuples(index=False):
            if prev is not None:
                w0 = {"intercept": prev.const, "X1": prev.beta_x1, "X2": prev.beta_x2}
                w1 = {"intercept": r.const, "X1": r.beta_x1, "X2": r.beta_x2}
                z0 = {"intercept": 1.0, "X1": prev.X1_latest, "X2": prev.X2_latest}
                z1 = {"intercept": 1.0, "X1": r.X1_latest, "X2": r.X2_latest}
                for comp in ("intercept", "X1", "X2"):
                    dz = z1[comp] - z0[comp]
                    dw = w1[comp] - w0[comp]
                    # News changes the data while the old weights stay fixed.
                    add(
                        meta,
                        r.vintage_date,
                        prev.vintage_date,
                        "revision",
                        comp,
                        "news",
                        w0[comp] * dz,
                        w0[comp],
                        dz,
                    )
                    # Re-estimation changes the weights while the old data stays fixed.
                    add(
                        meta,
                        r.vintage_date,
                        prev.vintage_date,
                        "revision",
                        comp,
                        "reestimation",
                        dw * z0[comp],
                        np.nan,
                        np.nan,
                    )
                    # The interaction captures the cross-term.
                    add(
                        meta,
                        r.vintage_date,
                        prev.vintage_date,
                        "revision",
                        comp,
                        "interaction",
                        dw * dz,
                        np.nan,
                        np.nan,
                    )
            prev = r

    decompositions = pd.DataFrame(rows, columns=_DECOMP_COLUMNS)
    decompositions["date"] = pd.to_datetime(decompositions["date"])
    decompositions["vintage_date"] = pd.to_datetime(decompositions["vintage_date"])
    decompositions["base_vintage_date"] = pd.to_datetime(
        decompositions["base_vintage_date"]
    )
    decompositions["forecast_horizon"] = decompositions["forecast_horizon"].astype(int)
    for col in ("contribution", "weight", "news"):
        decompositions[col] = decompositions[col].astype(float)
    return decompositions.reset_index(drop=True)


def build_vintages(
    data: dict[str, pd.DataFrame],
    source: str = "ols_nowcast",
) -> dict[str, pd.DataFrame]:
    """Build outturn and nowcast-forecast tables for NowcastData.

    Produce two DataFrames that describe the real-time data required by
    ``forecast_evaluation.NowcastData``:

        * **outturns**: point-in-time observations of ``y``, ``X1``, and ``X2`` at
            each release vintage. Earlier history uses final truth values at the
            first release vintage. Real-time quarters have one row per release, so
            ``NowcastData._align_outturn_vintages`` can carry values forward.

    * **forecasts**: within-quarter nowcasts of ``y``, one row per
      ``(reference_date, vintage_date)`` in the simulation.  The
    ``forecast_horizon`` is the signed quarter offset from the vintage's
    quarter to the target (0 = current-quarter nowcast, -1 = one quarter
    back, and so on).

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Simulation outputs returned by :func:`simulate`.
    source : str
        Forecast source label to use in the output tables.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping with keys ``"outturns"`` and ``"forecasts"``.

    Examples
    --------

        import forecast_evaluation as fe
        now = fe.NowcastData(
            outturns_data=data["vintages"]["outturns"],
            forecasts_data=data["vintages"]["forecasts"],
        )
    """
    truth = data["truth"]
    releases = data["releases"]
    nowcasts = data["nowcasts"]

    all_vintages = np.sort(releases["vintage_date"].unique())
    first_vintage = pd.Timestamp(all_vintages[0])
    realtime_refs = set(releases["reference_date"].unique())

    # Build the outturn table.
    out_rows: list[dict] = []

    # Add final historical values at the first available vintage.
    pre_rt = truth[~truth["date"].isin(realtime_refs)]
    for _, row in pre_rt.iterrows():
        for var in ("y", "X1", "X2"):
            out_rows.append(
                {
                    "date": row["date"],
                    "vintage_date": first_vintage,
                    "variable": var,
                    "frequency": "Q",
                    "forecast_horizon": 0,
                    "value": float(row[var]),
                }
            )

    # Add each real-time release; NowcastData carries values forward between them.
    for _, row in releases.iterrows():
        var = row["series"]  # "y", "X1", or "X2"
        out_rows.append(
            {
                "date": row["reference_date"],
                "vintage_date": row["vintage_date"],
                "variable": var,
                "frequency": "Q",
                "forecast_horizon": 0,
                "value": float(row["value"]),
            }
        )

    outturns = pd.DataFrame(out_rows, columns=_NOWCAST_OUTTURN_COLUMNS)
    outturns["date"] = pd.to_datetime(outturns["date"])
    outturns["vintage_date"] = pd.to_datetime(outturns["vintage_date"])
    outturns["forecast_horizon"] = outturns["forecast_horizon"].astype(int)
    outturns["value"] = outturns["value"].astype(float)
    outturns = outturns.sort_values(["variable", "date", "vintage_date"]).reset_index(
        drop=True
    )

    # Build forecasts. The horizon is target-quarter ordinal minus
    # vintage-quarter ordinal.
    ref_ord = (
        pd.to_datetime(nowcasts["reference_date"])
        .dt.to_period("Q")
        .apply(lambda p: p.ordinal)
    )
    vtg_ord = (
        pd.to_datetime(nowcasts["vintage_date"])
        .dt.to_period("Q")
        .apply(lambda p: p.ordinal)
    )

    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(nowcasts["reference_date"].values),
            "vintage_date": pd.to_datetime(nowcasts["vintage_date"].values),
            "variable": "y",
            "frequency": "Q",
            "forecast_horizon": (ref_ord - vtg_ord).astype(int).values,
            "value": nowcasts["y_nowcast"].values.astype(float),
            "source": source,
        },
        columns=_NOWCAST_FORECAST_COLUMNS,
    )
    forecasts = forecasts.sort_values(["date", "vintage_date"]).reset_index(drop=True)

    return {"outturns": outturns, "forecasts": forecasts}


def snapshot(
    data: dict[str, pd.DataFrame], vintage_date, window: int = 12
) -> pd.DataFrame:
    """Return the data available at a given ``vintage_date``.

    Return one row per quarter for the ``window`` quarters ending in the
    vintage's quarter. Columns are ``date, X1, X2, y, y_nowcast``:

        * Completed quarters carry their released values and ``y_nowcast`` is
            ``NaN``.
        * The in-progress quarter has ``y = NaN`` and uses the latest available
            ``X1`` and ``X2`` releases to compute ``y_nowcast``.
    """
    v = pd.Timestamp(vintage_date).normalize()
    truth_by_date = data["truth"].set_index("date")
    releases = data["releases"]
    coef_asof = _make_fitter(data["truth"], releases)
    coef = coef_asof(v)
    const, b1, b2 = coef["const"], coef["beta_x1"], coef["beta_x2"]

    cur_q = v.to_period("Q")
    q_index = pd.period_range(end=cur_q, periods=window, freq="Q")
    dates = q_index.to_timestamp(how="end").normalize()

    records = []
    for d in dates:
        if d + Y_PUBLICATION_LAG <= v:  # y is available at this vintage.
            x1 = truth_by_date.at[d, "X1"]
            x2 = truth_by_date.at[d, "X2"]
            y = truth_by_date.at[d, "y"]
            y_nowcast = np.nan
        else:  # Use the latest regressors while y remains unavailable.
            sub = releases[
                (releases["reference_date"] == d) & (releases["vintage_date"] <= v)
            ]
            x1_rel = sub.loc[sub["series"] == "X1", "value"]
            x2_rel = sub.loc[sub["series"] == "X2", "value"]
            x1 = x1_rel.iloc[-1] if len(x1_rel) else np.nan
            x2 = x2_rel.iloc[-1] if len(x2_rel) else np.nan
            y = np.nan
            y_nowcast = (
                const + b1 * x1 + b2 * x2
                if not (np.isnan(x1) or np.isnan(x2))
                else np.nan
            )
        records.append({"date": d, "X1": x1, "X2": x2, "y": y, "y_nowcast": y_nowcast})

    return pd.DataFrame(records)


def _quarter_fraction(vintage: pd.Series, reference: pd.Series) -> pd.Series:
    """Return the share of the quarter elapsed at each vintage date."""
    q_start = reference.dt.to_period("Q").dt.to_timestamp(how="start")
    span = (reference - q_start).dt.days.clip(lower=1)
    elapsed = (vintage - q_start).dt.days.clip(lower=0)
    return (elapsed / span).clip(0.0, 1.0)


def plot(data: dict[str, pd.DataFrame], show: bool = True, savepath: str | None = None):
    """Plot the target and regressor releases, coloured by vintage timing.

    Three panels share a calendar-time axis for target ``y`` and regressors
    ``X1`` and ``X2``. Each release appears at its ``vintage_date`` and uses a
    colour that indicates its position within the quarter. The final truth value
    for each quarter appears as a black cross.
    """
    import matplotlib.pyplot as plt

    truth = data["truth"]
    releases = data["releases"]

    realtime_refs = releases["reference_date"].unique()
    truth_rt = truth[truth["date"].isin(realtime_refs)].sort_values("date")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    cmap = plt.get_cmap("viridis")

    # Show the quarterly target and overlay nowcasts by release timing.
    axes[0].plot(
        truth_rt["date"],
        truth_rt["y"],
        "-o",
        color="black",
        label="y (released)",
        zorder=3,
    )
    nowcasts = data.get("nowcasts")
    if nowcasts is not None and len(nowcasts):
        nc_frac = _quarter_fraction(
            nowcasts["vintage_date"], nowcasts["reference_date"]
        )
        axes[0].scatter(
            nowcasts["vintage_date"],
            nowcasts["y_nowcast"],
            c=nc_frac,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=18,
            edgecolor="none",
            label="y nowcast",
        )
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper left")

    scatter = None
    for ax, series in zip(axes[1:], ["X1", "X2"]):
        sub = releases[releases["series"] == series].copy()
        frac = _quarter_fraction(sub["vintage_date"], sub["reference_date"])
        scatter = ax.scatter(
            sub["vintage_date"],
            sub["value"],
            c=frac,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=30,
            edgecolor="none",
        )
        ax.scatter(
            truth_rt["date"],
            truth_rt[series],
            marker="x",
            color="black",
            s=60,
            label=f"{series} (truth)",
        )
        ax.set_ylabel(series)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("vintage date")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, location="right", fraction=0.04, pad=0.02)
        cbar.set_label("fraction of quarter elapsed at release")

    fig.suptitle("Real-time regressor releases vs. target (colour per vintage)")

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_vintage(
    data: dict[str, pd.DataFrame],
    vintage_date,
    window: int = 12,
    show: bool = True,
    savepath: str | None = None,
):
    """Plot the ``y``, ``X1``, and ``X2`` values known at one vintage.

    Completed quarters show released values. The in-progress quarter shows the
    latest regressors and the resulting nowcast of ``y``.
    """
    import matplotlib.pyplot as plt

    v = pd.Timestamp(vintage_date).normalize()
    snap = snapshot(data, v, window=window)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))

    axes[0].plot(snap["date"], snap["y"], "-o", color="black", label="y (released)")
    nc = snap[snap["y"].isna() & snap["y_nowcast"].notna()]
    if len(nc):
        axes[0].plot(
            nc["date"],
            nc["y_nowcast"],
            "D",
            color="tab:red",
            markersize=8,
            label="y nowcast",
        )
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper left")

    for ax, series, color in zip(axes[1:], ["X1", "X2"], ["tab:blue", "tab:green"]):
        ax.plot(snap["date"], snap[series], "-o", color=color, label=series)
        ax.set_ylabel(series)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("reference date (quarter end)")
    fig.suptitle(f"Data as known at vintage {v.date()}")
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    data = simulate()
    print("truth:", data["truth"].shape)
    print(data["truth"].tail())
    print("\nreleases:", data["releases"].shape)
    print(data["releases"].head(MONTHLY_RELEASES + WEEKLY_RELEASES + 1))
    print("\nnowcasts:", data["nowcasts"].shape)
    print(data["nowcasts"].head(MONTHLY_RELEASES + WEEKLY_RELEASES))
    print("\ntrue coefficients:")
    print(data["coefficients"].to_string(index=False))
    print("\nOLS-recovered coefficients (full sample):")
    print(data["fitted_coefficients"].to_string())

    # Compare average absolute error at the first and last vintages.
    nc = data["nowcasts"]
    by_q = nc.groupby("reference_date")
    first_err = by_q.first()["error"].abs().mean()
    last_err = by_q.last()["error"].abs().mean()
    print("\nmean nowcast |error| across real-time quarters:")
    print(f"  first (least mature) vintage: {first_err:.4f}")
    print(f"  last  (most mature)  vintage: {last_err:.4f}")

    # Validate and display the decomposition table.
    from news_decomp.schema import decomposition_schema

    decomp = data["decompositions"]
    decomposition_schema.validate(decomp)
    print("\ndecompositions:", decomp.shape, "(schema-valid)")
    print(decomp.head(5).to_string(index=False))

    # Display the outturn and forecast tables.
    vt = data["vintages"]
    print("\nvintages['outturns']:", vt["outturns"].shape)
    print(vt["outturns"].head(5).to_string(index=False))
    print("\nvintages['forecasts']:", vt["forecasts"].shape)
    print(vt["forecasts"].head(5).to_string(index=False))
    print("\nforecast_horizon distribution:")
    print(vt["forecasts"]["forecast_horizon"].value_counts().sort_index().to_string())

    plot(data)
    last_ref = nc["reference_date"].max()
    plot_vintage(data, last_ref - pd.Timedelta(days=20))
