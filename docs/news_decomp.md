# News Decomposition — Data Contract

## Purpose

`news_decomp` measures how variables contribute to forecasts without depending
on the model that generated them.

It answers two related questions:

- **Level decomposition**: what does each input contribute to one forecast?
  \[\hat y \;=\; \sum_{\text{component}} \text{contribution}\]
- **News decomposition**: what explains a forecast *revision* between two vintages?
  \[\hat y_{v_1} - \hat y_{v_0} \;=\; \sum_{\text{component}} \text{contribution}\]

The module reads one standardised table and does not import model classes.
Upstream code computes weights, surprises, and counterfactuals; this document
describes only how the consumer reads those results.

---

## The dataset: `decompositions`

This long-format table accompanies `data.forecasts`. Each row records one
additive component of either a forecast level or a forecast revision.

### Columns

| Column | Type | Null? | Description |
|---|---|---|---|
| `variable` | str | no | Target being decomposed, such as `"gdpkp"`. |
| `date` | Timestamp | no | End date of the forecast target period. |
| `forecast_horizon` | int | no | Steps from the target: `0` means nowcast; positive values mean future periods. |
| `frequency` | str | no | `"Q"` or `"M"`, matching the forecast. |
| `source` | str | no | Model or label that produced the forecast. |
| `vintage_date` | Timestamp | no | New vintage at which the decomposition was computed. |
| `base_vintage_date` | Timestamp | yes | Earlier vintage for revision rows; `NaT` for level rows. |
| `decomposition` | str | no | `"level"` for a forecast level or `"revision"` for a change between vintages. |
| `component` | str | no | Contributor name (see namespace below) |
| `revision_source` | str | yes | Revision part: `"news"`, `"reestimation"`, or `"interaction"`. Blank (`NaN`) for level rows. |
| `contribution` | float | no | Signed additive contribution. The contributions close the level or revision identity. |
| `weight` | float | yes | Linear-model weight \(w_i\) when the factorisation with `news` applies. |
| `news` | float | yes | Surprise \(x_i - E[x_i \mid \Omega_{v_0}]\) for a linear news row. |
| `forecast_metric` | str | no | Transform used for the forecast, such as `"levels"`, `"pop"`, or `"yoy"`. |

For a linear model, the contribution factorises as follows:

\[
\mathrm{contribution}_i \;=\; \underbrace{\mathrm{weight}_i}_{w_i} \;\times\; \underbrace{\mathrm{news}_i}_{x_i - E[x_i\mid\Omega_{v_0}]}
\]

`weight` and `news` explain the contribution. They distinguish a large surprise
with a small weight from a small surprise with a large weight. Both are
optional; `contribution` is the value that the additivity check sums.

### `component` and `revision_source`

These three columns classify each row:

- **`decomposition`**: `"level"` or `"revision"`, which identifies the identity that the row closes.
- **`component`**: the input or term that contributed, such as `"bls_payrolls"`, `"intercept"`, or `"residual"`.
- **`revision_source`**: the part of a revision explained by the row. Leave it blank for level rows.

Use `decomposition` as the filter. `base_vintage_date = NaT` agrees with a
level row, but the decomposition value is authoritative. All rows in a group
share the same decomposition value.

| `decomposition` | `base_vintage_date` | `revision_source` | Additive identity |
|---|---|---|---|
| `"level"` | `NaT` | blank | \(\hat y = \sum \text{contribution}\) |
| `"revision"` | old vintage date | `news` / `reestimation` / `interaction` | \(\Delta\hat y = \sum \text{contribution}\) |

For revision rows, `revision_source` splits the revision into three pieces:

| `revision_source` | What moved | `weight`/`news` present? |
|---|---|---|
| `news` | New data changed while parameters stayed fixed. | Both, for linear models. |
| `reestimation` | Parameters changed while data stayed fixed. | No. |
| `interaction` | The cross-term from simultaneous data and parameter changes. | No. |

### `component` namespace

The `component` label follows one of these patterns. The final column shows
which row types may use each pattern.

| Pattern | Meaning | Valid for |
|---|---|---|
| `"<varname>"` | Regressor or news variable, such as `"bls_payrolls"`. | Any row. |
| `"intercept"` | Constant term. | Level rows. |
| `"<varname>_lag<k>"` | Regressor or own-lag contribution `k` periods back. Use the target name for own lags, such as `"gdpkp_lag1"`. | Level rows. |
| `"residual"` | Remainder that closes the identity. | Level and `news` rows. |

`reestimation` and `interaction` rows use the `"<varname>"` pattern, or a
label chosen by the model. The label identifies the parameter block that
changed, not a data release.

### Worked example

Suppose a linear MIDAS model, `sc_midas`, forecasts quarterly US GDP growth
(`gdpkp`) for **2026-Q2**. A new payrolls release arrives, and the model runs
again:

- Old vintage `v0 = 2026-05-01`: \(\hat y_{v_0} = 1.80\)
- New vintage `v1 = 2026-05-15`: \(\hat y_{v_1} = 2.05\)
- Revision to explain: \(\Delta\hat y = 2.05 - 1.80 = +0.25\)

**1. Level decomposition of the new forecast** (`decomposition = "level"`,
with a blank `revision_source`). This answers: what makes up 2.05?

| variable | date | forecast_horizon | source | vintage_date | base_vintage_date | decomposition | component | revision_source | contribution | weight | news | forecast_metric |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gdpkp | 2026-06-30 | 0 | sc_midas | 2026-05-15 | NaT | level | `intercept` | — | 0.50 | 1.00 | — | pop |
| gdpkp | 2026-06-30 | 0 | sc_midas | 2026-05-15 | NaT | level | `gdpkp_lag1` | — | 0.70 | 0.40 | — | pop |
| gdpkp | 2026-06-30 | 0 | sc_midas | 2026-05-15 | NaT | level | `bls_payrolls` | — | 0.65 | 0.30 | — | pop |
| gdpkp | 2026-06-30 | 0 | sc_midas | 2026-05-15 | NaT | level | `ip` | — | 0.15 | 0.15 | — | pop |
| gdpkp | 2026-06-30 | 0 | sc_midas | 2026-05-15 | NaT | level | `residual` | — | 0.05 | — | — | pop |

Level invariant: \(0.50 + 0.70 + 0.65 + 0.15 + 0.05 = 2.05 = \hat y_{v_1}\) ✓

**2. News decomposition of the revision** (`decomposition = "revision"`,
`base_vintage_date = 2026-05-01`). Payrolls produced a +40k surprise, and the
parameters changed slightly. This answers: what explains the +0.25 revision?

| variable | date | source | vintage_date | base_vintage_date | decomposition | component | revision_source | contribution | weight | news | forecast_metric |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gdpkp | 2026-06-30 | sc_midas | 2026-05-15 | 2026-05-01 | revision | `bls_payrolls` | news | 0.18 | 0.0045 | 40.0 | pop |
| gdpkp | 2026-06-30 | sc_midas | 2026-05-15 | 2026-05-01 | revision | `ip` | news | −0.02 | 0.010 | −2.0 | pop |
| gdpkp | 2026-06-30 | sc_midas | 2026-05-15 | 2026-05-01 | revision | `bls_payrolls` | reestimation | 0.08 | — | — | pop |
| gdpkp | 2026-06-30 | sc_midas | 2026-05-15 | 2026-05-01 | revision | `residual` | news | 0.01 | — | — | pop |

Revision invariant: \(0.18 + (-0.02) + 0.08 + 0.01 = 0.25 = \hat y_{v_1} - \hat y_{v_0}\) ✓

Factor consistency (linear `news` rows only): \(0.0045 \times 40.0 = 0.18\) and
\(0.010 \times (-2.0) = -0.02\) ✓. The `reestimation` and `residual` rows carry no
`weight`/`news`, so the check is skipped.

This example shows the main conventions: `bls_payrolls` keeps the same label in
the level and revision tables; each group contains one decomposition type; and
the `reestimation` row carries a contribution of 0.08 without factors. It
represents the parameter-change part of the revision separately from the data
surprise.

---

## Checks performed by the consumer

For **level rows** (`decomposition == "level"`):
\[
\sum_{\text{component}} \text{contribution} \;=\; \hat y_{(\text{variable},\,\text{date},\,\text{horizon},\,\text{source},\,\text{vintage})}
\]

For **revision rows** (`decomposition == "revision"`), sum every
`revision_source`:
\[
\sum_{\text{component}} \text{contribution} \;=\; \hat y_{\text{vintage}} - \hat y_{\text{base\_vintage}}
\]

When both factors are present, the consumer also checks:
\[
\text{weight} \times \text{news} \;=\; \text{contribution}
\]

### Grouping keys

- **Forecast or level identity:** `(variable, date, forecast_horizon, source, vintage_date)`
- **Revision identity:** the same keys plus `base_vintage_date`.

---

## Answering questions with the data

Every row has an additive `contribution` and keys for `component`,
`decomposition`, and `revision_source`. Common questions therefore use simple
filters and `groupby` operations. The recipes below use only contract columns,
so they work with any conforming producer.

### "What makes up this forecast?" (level)

Filter one forecast to its level rows, then rank components by contribution
magnitude:

```python
level = df[
    (df["decomposition"] == "level")
    & (df["variable"] == "gdpkp")
    & (df["date"] == target_date)
    & (df["source"] == "sc_midas")
    & (df["vintage_date"] == vintage)
]

level.groupby("component")["contribution"].sum().sort_values(key=abs, ascending=False)
```

The contributions sum to the forecast value.

### "Which indicator moved the forecast, and by how much?" (revision)

Each revision row is a signed contribution to the change between two vintages.
Group by `component` and sum the contributions:

```python
rev = df[
    (df["decomposition"] == "revision")
    & (df["variable"] == "gdpkp")
    & (df["vintage_date"] == v1)
    & (df["base_vintage_date"] == v0)
]

rev.groupby("component")["contribution"].sum().sort_values(key=abs, ascending=False)
```

`contribution` gives the change in forecast units, and `component` identifies
the source of that change. The sum equals
\(\hat y_{v_1} - \hat y_{v_0}\).

### "…because of new data, or because the model re-estimated?"

Split the revision by `revision_source`. `news` rows represent new data;
`reestimation` and `interaction` represent parameter effects:

```python
rev.groupby(["component", "revision_source"])["contribution"].sum().unstack()
```

To attribute the move only to new data, keep `revision_source == "news"`.

### "How big was the surprise, and how sensitive was the forecast?"

For linear `news` rows, the contribution factorises as `weight × news`:

- `news` — the surprise \(x_i - E[x_i \mid \Omega_{v_0}]\), in the indicator's own units
- `weight` — the forecast's sensitivity to that indicator

```python
rev.loc[
    rev["revision_source"] == "news", ["component", "news", "weight", "contribution"]
]
```

Some components have no single scalar weight, such as a summed multi-lag
block. Those rows use `NaN` for `weight` and `news`; use their defined
`contribution` instead.

### "Does it all add up?" (check)

```python
# Level: components sum to the forecast value.
level.groupby(["variable", "date", "forecast_horizon", "source", "vintage_date"])[
    "contribution"
].sum()

# Revision: components (all revision_sources) sum to the forecast change.
rev.groupby(
    [
        "variable",
        "date",
        "forecast_horizon",
        "source",
        "vintage_date",
        "base_vintage_date",
    ]
)["contribution"].sum()
```

> **Namespaced components.** A producer may label a component as
> `"model::component"` when several sub-models contribute. Split on the
> producer's separator to aggregate by sub-model or raw indicator:
> `df["component"].str.split("::").str[0]`.

> **Identifying the release.** The contract identifies the component and the
> size of its contribution, but not the observation's reference date or value.
> Join a source-data table on `(component, vintage_date)` to recover that detail.

---

## Pandera schema (reference)

```python
import numpy as np
import pandera as pa
from pandera import Column, Check

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
        # decomposition flag must be consistent with base_vintage_date and revision_source.
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
        # weight * news must reconstruct contribution where both are provided.
        Check(
            lambda df: (
                df[["weight", "news"]].isna().any(axis=1)
                | np.isclose(df["weight"] * df["news"], df["contribution"], atol=1e-8)
            ),
            error="contribution must equal weight * news where both are provided",
        ),
    ],
)
```

---

## Scope

| In scope | Out of scope (upstream) |
|---|---|
| The `decompositions` table schema | How models compute `weight`, `news`, counterfactuals |
| Additivity & factor-consistency invariants | The `_decompose()` producer hook on `ForecastModel` |
| `component` / `revision_source` conventions | Frozen-parameter re-estimation logic |
| Consumption independent of the model (`groupby`, waterfall, checks) | Shapley allocation for nonlinear models |

The consumer uses only this contract. Any model that emits conforming rows,
linear or otherwise, can use `news_decomp` without code tied to that model.

## Purpose

This note summarises the evaluation framework for a mixed-frequency dynamic
factor model that nowcasts UK GDP in real time.

It answers two questions:

- **Model accuracy**: how close is the aggregate nowcast to realised GDP?
- **Indicator usefulness**: how much does each indicator improve the nowcast,
  and does that value come from signal or publication timing?

The framework evaluates nowcasts across pseudo-real-time vintages and separates
two sources of indicator value:

- **Intrinsic content**: value that reflects the indicator's co-movement with GDP.
- **Timing premium**: additional value that comes from an early release date.

---

## Metrics

Let \( y_\tau \) denote realised GDP growth in quarter \( \tau \), and
\( \hat y_\tau \) the nowcast for that quarter.

### Model accuracy

**Root mean squared error (RMSE)**

\[
\mathrm{RMSE}
\;=\;
\sqrt{\frac{1}{T}\sum_{\tau=1}^{T}(y_\tau - \hat y_\tau)^2}
\]

RMSE gives more weight to large misses, so unusual episodes affect it strongly.

**Mean absolute error (MAE)**

\[
\mathrm{MAE}
\;=\;
\frac{1}{T}\sum_{\tau=1}^{T}\lvert y_\tau - \hat y_\tau \rvert
\]

MAE measures the average absolute nowcast error in GDP growth points.

---

### Indicator usefulness

Let \( \Omega_t \) denote the full information set at evaluation date \( t \),
and \( \Omega_t^{(-j)} \) the same information set with indicator \( j \)
removed.

**Marginal contribution**

\[
\Delta_{j,\tau}
\;=\;
\hat y_\tau(\Omega_t) - \hat y_\tau(\Omega_t^{(-j)})
\]

This is the change in the quarter-\( \tau \) nowcast that indicator \( j \)
provides.

**Signal magnitude**

\[
V_j^{\mathrm{abs}}
\;=\;
\frac{1}{T}\sum_{\tau=1}^{T}\lvert \Delta_{j,\tau} \rvert
\]

This measures how much indicator \( j \) usually moves the nowcast.

**Directional accuracy (hit rate)**

\[
H_j
\;=\;
\frac{1}{T}\sum_{\tau=1}^{T}
\mathbf{1}
\left(
\lvert y_\tau-\hat y_\tau(\Omega_t)\rvert
<
\lvert y_\tau-\hat y_\tau(\Omega_t^{(-j)})\rvert
\right)
\]

This is the share of quarters in which indicator \( j \) moves the nowcast
closer to realised GDP.

**Average error improvement**

\[
E_j
\;=\;
\frac{1}{T}\sum_{\tau=1}^{T}
\left[
\lvert y_\tau-\hat y_\tau(\Omega_t^{(-j)})\rvert
-
\lvert y_\tau-\hat y_\tau(\Omega_t)\rvert
\right]
\]

This measures the average reduction in forecast error from indicator \( j \).

---

## Timing decomposition

Observed usefulness combines predictive content with release-calendar effects.
The framework separates them by estimating:

\[
V_{j,k}^{\mathrm{abs}}
\;=\;
\alpha_j + \beta_j n_{j,k} + \eta_{j,k}
\]

where:

- \( \alpha_j \) = **intrinsic content**
- \( \beta_j \) = **timing premium**
- \( n_{j,k} \) = number of within-quarter observations available for indicator
    \( j \) at evaluation point \( k \)

An indicator with high \( \alpha_j \) remains informative after accounting for
timing. An indicator with high \( \beta_j \) gains more value from early release.

---

## Information density

To compare indicators after accounting for publication delay, the framework
also uses information density:

\[
D_j
\;=\;
\frac{V_j^{\mathrm{abs}}}{w_j},
\qquad
D_j^*
\;=\;
\frac{\alpha_j}{w_j}
\]

where \( w_j \) is the publication delay in weeks.

This measures the nowcasting value delivered per unit of waiting time.

---

## Summary

The framework evaluates aggregate nowcast accuracy and each indicator's
marginal usefulness in real time.

Its central distinction is between **intrinsic signal** and **timing advantage**.
This helps identify indicators that matter because they contain strong GDP
information, rather than only because they arrive early.

---

## Key references

Two papers motivate this framework:

- Giannone, D., Reichlin, L., and Small, D. (2008),
    [“Nowcasting: The real-time informational content of macroeconomic data”](https://doi.org/10.1162/jpet.2008.26.4.665),
  *Journal of Monetary Economics*, 55(4), 665–676.
- Bańbura, M., and Modugno, M. (2014),
  [“Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data”](https://doi.org/10.1002/jae.2306),
  *Journal of Applied Econometrics*, 29(1), 133–160.

---

## Scope

| In scope | Out of scope |
|---|---|
| Definition of evaluation metrics | Estimation details of the upstream DFM |
| Real-time nowcast accuracy assessment | Data collection and publication calendars |
| Indicator-level contribution metrics | Parameter estimation for the model |
| Intrinsic-content vs timing-premium decomposition | Alternative nonlinear attribution methods |

This note summarises the evaluation framework and its metrics. It does not
describe model-estimation code or the detailed construction of real-time data.
