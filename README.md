# news_decomp

---

## Installation

```bash
# Install the published package.
pip install news-decomp

# Clone the repository for development.
git clone https://github.com/bank-of-england/news-decomp
cd news-decomp

# Install the package with development dependencies.
pip install -e ".[dev,docs,notebooks]"
```

---

## Quick start

Run the sample-data example:
```bash
python -m examples.example_data
```

Calculate forecast metrics and draw charts:
```bash
python -m examples.example_analysis
```

Build the nowcast report:
```bash
python -m examples.example_report
```

Or open the interactive Marimo example:
```bash
marimo edit examples/example_data_marimo.py
```

The Marimo example is also published in the documentation. Regenerate its
Markdown page after changing the app:
```bash
python docs/convert_notebooks.py
```

---

## The `decompositions` table

The package consumes one long-format table. Each row records one additive
component of either a forecast level or a forecast revision. Every row must
satisfy the schema in `src/news_decomp/schema.py`; see
[news_decomp.md](docs/news_decomp.md) for the complete data contract.

### Columns

| Column | Type | Nullable | Description |
|---|---|---|---|
| `variable` | str | no | Target variable, such as `"gdpkp"` or `"y"`. |
| `date` | Timestamp | no | End of the target period, such as `2026-06-30` for 2026-Q2. |
| `forecast_horizon` | int | no | Steps from the target: `0` means nowcast, `1` means one step ahead, and so on. |
| `frequency` | str | no | Target frequency: `"Q"` for quarterly or `"M"` for monthly. |
| `source` | str | no | Model or label that produced the forecast. |
| `vintage_date` | Timestamp | no | Date at which the decomposition was computed. |
| `base_vintage_date` | Timestamp | **yes** | Earlier vintage for revision rows; `NaT` for level rows. |
| `decomposition` | str | no | `"level"` for a forecast level or `"revision"` for a change between vintages. |
| `component` | str | no | Contributor name, such as a regressor, `"intercept"`, own lag, or `"residual"`. |
| `revision_source` | str | **yes** | Revision part: `"news"`, `"reestimation"`, or `"interaction"`. Blank (`NaN`) for level rows. |
| `contribution` | float | no | Signed additive contribution. Components sum to the level or revision. |
| `weight` | float | **yes** | Linear-model weight $w_i$ when the factorisation $\text{contribution} = w_i \times \text{news}_i$ applies. |
| `news` | float | **yes** | Surprise $x_i - \mathbb{E}[x_i \mid \Omega_{v_0}]$ for a linear news row. |
| `forecast_metric` | str | no | Transform used for the forecast, such as `"levels"`, `"pop"`, or `"yoy"`. |

## Documentation

- [News Decomposition data contract](docs/news_decomp.md): schema, identities,
  and worked examples.
- [API reference](docs/api.md): public classes and methods.
- [About](docs/about.md).
- The `examples/` directory contains runnable scripts for simulation, metrics,
  reports, and Marimo.

## Data Classification
Bank of England Data Classification: OFFICIAL BLUE