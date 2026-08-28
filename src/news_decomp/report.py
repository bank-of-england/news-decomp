"""Reporting mixin for coordinated nowcast analysis and visualisation.

This module provides :class:`NewsReport`, which coordinates analysis and
reporting methods for :class:`~news_decomp.NewsData`.

The mixin delegates data preparation to
:class:`~news_decomp.analysis.NewsAnalysis` and visualisation to
:class:`~news_decomp.plots.NewsPlots`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class NewsReport:
    """Provide tables and a coordinated nowcast report.

    :class:`~news_decomp.NewsData` supplies the validated decomposition frame
    through ``self.df``.

    """

    # Table methods.

    def data_flow_table(
        self,
        n_dates: int | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Build a table that links model updates to their impacts.

        Parameters
        ----------
        n_dates : int | None
            Keep only the most recent ``n_dates`` target dates. ``None`` keeps
            every date.
        variable : str | None
            Target variable. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: [Model Update, Data Release, Target
            Quarter, Series, Data Revision, Impact (pp), Nowcast (%)]
        """
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        level = self.df[self.df["decomposition"] == "level"].copy()
        revision = self.df[
            (self.df["decomposition"] == "revision")
            & (self.df["revision_source"] == "news")
        ].copy()

        # Restrict both level and revision rows to the selected target dates.
        if n_dates is not None:
            dates = sorted(level["date"].unique())[-n_dates:]
            level = level[level["date"].isin(dates)]
            revision = revision[revision["date"].isin(dates)]

        # Sum level contributions to obtain the nowcast at each vintage.
        nowcast_by_vintage = (
            level.groupby(["date", "vintage_date"])["contribution"]
            .sum()
            .reset_index(name="nowcast")
        )

        # Prefer news rows because they identify the data change at each update.
        if not revision.empty:
            flow = revision[
                [
                    "date",
                    "vintage_date",
                    "base_vintage_date",
                    "component",
                    "news",
                    "contribution",
                ]
            ].copy()
            flow = flow[~flow["component"].isin(["intercept", "residual"])]
            flow = flow.merge(
                nowcast_by_vintage, on=["date", "vintage_date"], how="left"
            )
        else:
            flow = level[["date", "vintage_date", "component", "contribution"]].copy()
            flow = flow[~flow["component"].isin(["intercept", "residual"])]
            flow["base_vintage_date"] = pd.NaT
            flow["news"] = np.nan
            flow = flow.merge(
                nowcast_by_vintage, on=["date", "vintage_date"], how="left"
            )

        # Rename columns and format dates for the printed report.
        flow = flow.rename(
            columns={
                "vintage_date": "Model Update",
                "base_vintage_date": "Data Release",
                "date": "Target Quarter",
                "component": "Series",
                "news": "Data Revision",
                "contribution": "Impact (pp)",
                "nowcast": "Nowcast (%)",
            }
        )

        flow["Model Update"] = pd.to_datetime(flow["Model Update"]).dt.strftime(
            "%Y-%m-%d"
        )
        flow["Data Release"] = pd.to_datetime(flow["Data Release"]).dt.strftime(
            "%Y-%m-%d"
        )
        flow["Target Quarter"] = flow["Target Quarter"].apply(
            lambda d: str(pd.Timestamp(d).to_period("Q"))
        )

        flow = flow.sort_values(
            ["Model Update", "Impact (pp)"],
            ascending=[False, False],
        )

        cols = [
            "Model Update",
            "Data Release",
            "Target Quarter",
            "Series",
            "Data Revision",
            "Impact (pp)",
            "Nowcast (%)",
        ]
        flow = flow[[c for c in cols if c in flow.columns]]

        return flow.reset_index(drop=True)

    def summary_table(
        self,
        variable: str | None = None,
        source: str | None = None,
        target_date: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Return summary statistics for each indicator.

        Parameters
        ----------
        variable : str | None
            Target variable. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        target_date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the latest date.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            [Indicator, Total Impact (pp), Avg |Impact|, Releases, Direction]
        """
        # Resolve and validate the requested data slice.
        variable = self._validate_variable(variable)
        source = self._validate_source(source)
        target_date = self._validate_date(target_date)

        # Keep rows for the selected variable and source.
        df = self._filter(variable=variable, source=source)

        revision = df[
            (df["decomposition"] == "revision")
            & (df["date"] == target_date)
            & (df["revision_source"] == "news")
        ]

        if revision.empty:
            level = df[(df["decomposition"] == "level") & (df["date"] == target_date)]
            level = level[~level["component"].isin(["intercept", "residual"])]
            summary = (
                level.groupby("component")
                .agg(
                    total_impact=("contribution", "sum"),
                    avg_abs_impact=("contribution", lambda x: x.abs().mean()),
                    n_releases=("contribution", "count"),
                )
                .reset_index()
            )
        else:
            revision = revision[~revision["component"].isin(["intercept", "residual"])]
            summary = (
                revision.groupby("component")
                .agg(
                    total_impact=("contribution", "sum"),
                    avg_abs_impact=("contribution", lambda x: x.abs().mean()),
                    n_releases=("contribution", "count"),
                )
                .reset_index()
            )

        summary["direction"] = summary["total_impact"].apply(
            lambda x: "up" if x > 0.001 else ("down" if x < -0.001 else "flat")
        )

        summary = summary.rename(
            columns={
                "component": "Indicator",
                "total_impact": "Total Impact (pp)",
                "avg_abs_impact": "Avg |Impact|",
                "n_releases": "Releases",
                "direction": "Direction",
            }
        )

        return summary.sort_values("Total Impact (pp)", key=abs, ascending=False)

    # Report generation.

    def report(
        self,
        variable: str | None = None,
        source: str | None = None,
        target_date: pd.Timestamp | str | None = None,
        show: bool = True,
        figsize: tuple[float, float] = (12, 12),
    ) -> tuple[Any, dict[str, Any]]:
        """Generate the full nowcast report.

        The figure has two rows:

        - Row 1 shows level contributions and the nowcast path.
        - Row 2 shows raw revision contributions and the net change.

        In each row, the stacked bars sum to the overlaid line by construction.

        Matplotlib uses its active style unless the caller has configured a
        different style.

        Parameters
        ----------
        variable : str | None
            Target variable. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        target_date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the latest date.
        show : bool
            Display the plots immediately when ``True``.
        figsize : tuple[float, float]
            Overall figure size in inches.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            The figure and a mapping of panel names to axes.
        """
        # Resolve and validate the requested data slice.
        variable = self._validate_variable(variable)
        source = self._validate_source(source)
        target_date = self._validate_date(target_date)

        import matplotlib.pyplot as plt

        fig, (ax_nowcast, ax_rev_contrib) = plt.subplots(2, 1, figsize=figsize)

        # Use one vintage order so each revision bar sits below its forecast.
        lev = self._level(variable=variable, source=source)
        lev = lev[lev["date"] == target_date]
        x_order = sorted(pd.Timestamp(v) for v in lev["vintage_date"].unique())

        # Row 1: level contributions with the nowcast line overlaid.
        self.plot_nowcast_contributions(
            date=target_date,
            variable=variable,
            source=source,
            ax=ax_nowcast,
            show=False,
            x_order=x_order,
        )

        # Row 2: raw revision contributions with the net change overlaid.
        self.plot_raw_revision_contributions(
            date=target_date,
            variable=variable,
            source=source,
            ax=ax_rev_contrib,
            show=False,
            x_order=x_order,
        )

        # The bar plots use integer x positions; datetime limits would misalign
        # the bars.

        # Print the headline nowcast summary.
        target_q = pd.Timestamp(target_date).to_period("Q")
        print("\n" + "=" * 90)
        print(f"NOWCAST SUMMARY - {target_q}")
        print("=" * 90)

        evolution = self.nowcast_evolution(
            date=target_date,
            variable=variable,
            source=source,
        )
        if not evolution.empty:
            latest = evolution.iloc[-1]
            first = evolution.iloc[0]
            print(f"  Current Nowcast : {latest['nowcast']:.2f}%")
            print(f"  Initial Nowcast : {first['nowcast']:.2f}%")
            print(f"  Total Revision  : {latest['nowcast'] - first['nowcast']:+.2f} pp")
            print(f"  Vintages        : {len(evolution)}")

        # Print indicator statistics.
        print("\n" + "-" * 90)
        print("INDICATOR SUMMARY")
        print("-" * 90)
        summary = self.summary_table(
            variable=variable, source=source, target_date=target_date
        )
        if not summary.empty:
            print(summary.to_string(index=False))
        else:
            print("  (no indicator data)")

        # Print the data flow table.
        print("\n" + "-" * 90)
        print("DATA FLOW TABLE")
        print("-" * 90)
        flow_table = self.data_flow_table(n_dates=10, variable=variable, source=source)
        if not flow_table.empty:
            print(flow_table.to_string(index=False))
        else:
            print("  (no data flow records)")

        print("=" * 90 + "\n")

        # Add space between panels and reserve room for legends.
        fig.tight_layout(pad=3.0, h_pad=8.0, w_pad=2.0)
        fig.subplots_adjust(right=0.75)  # Leave 25% on right for legends

        if show:
            plt.show()

        return fig, {
            "nowcast_contributions": ax_nowcast,
            "revision_contributions": ax_rev_contrib,
        }
