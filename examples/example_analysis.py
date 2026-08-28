"""Print forecast and indicator metrics, then draw their charts.

Run with::

    python -m examples.example_analysis
"""

import matplotlib.pyplot as plt
import pandas as pd

from news_decomp.news_decomp import NewsData
from news_decomp.sample import simulate

# 1. Generate sample data and the realised target series.
data = simulate()
news_data = NewsData(data["decompositions"])
realised = data["truth"].set_index("date")["y"]

# 2. Derive observation counts and publication delays from releases.
releases = data["releases"]
decomp = data["decompositions"]
level_rows = decomp[decomp["decomposition"] == "level"]

# Count releases available for each indicator at each evaluation vintage.
# Restrict the count to the target quarter, as required by the metric note.
n_obs_records = []
for comp in ["X1", "X2"]:
    rows = level_rows[level_rows["component"] == comp][
        ["date", "vintage_date"]
    ].drop_duplicates()
    comp_releases = releases[releases["series"] == comp]
    for row in rows.itertuples(index=False):
        n = int(
            (
                (comp_releases["reference_date"] == row.date)
                & (comp_releases["vintage_date"] <= row.vintage_date)
            ).sum()
        )
        n_obs_records.append(
            {"component": comp, "vintage_date": row.vintage_date, "n": n}
        )
n_obs = (
    pd.DataFrame(n_obs_records)
    .groupby(["component", "vintage_date"], as_index=False)["n"]
    .mean()
)

# Calculate the average delay in weeks from each target period to its latest
# release. Give same-day releases a one-day floor so density stays finite.
latest_release = (
    releases.groupby(["series", "reference_date"])["vintage_date"].max().reset_index()
)
latest_release["delay_weeks"] = (
    latest_release["reference_date"] - latest_release["vintage_date"]
).dt.days.clip(lower=1) / 7
pub_delays = (
    latest_release.groupby("series")["delay_weeks"]
    .mean()
    .rename(index={"X1": "X1", "X2": "X2"})
)
# Series and component names match in this simulation.
pub_delays.index.name = "component"

# 3. Print model accuracy.
print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)
print(f"  RMSE : {news_data.rmse(realised):.4f}")
print(f"  MAE  : {news_data.mae(realised):.4f}")

# 4. Print the consolidated indicator table.
print("\n" + "=" * 60)
print("INDICATOR USEFULNESS")
print("=" * 60)
table = news_data.indicator_table(realised, n_obs=n_obs, pub_delays=pub_delays)
# Keep indicators with timing data; intercept and residual rows do not qualify.
table = table.dropna()
print(table.to_string())
print("=" * 60)

# 5. Print the revision split between new data and parameter changes.
print("\n" + "=" * 60)
print("VINTAGE-REVISION CONTRIBUTION (new data vs revised data)")
print("=" * 60)
vrc = news_data.vintage_revision_contribution()
if not vrc.empty:
    print(vrc.to_string(index=False))
else:
    print("  (no revision rows in data)")
print("=" * 60)

# 6. Plot RMSE and MAE over time.
news_data.plot_accuracy(realised, show=False)

# 7. Plot indicator metrics over time.
news_data.plot_indicators_over_time(realised, show=False)

plt.show()
