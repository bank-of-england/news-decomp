"""Plotting methods for :class:`~news_decomp.NewsData`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _norm_pdf(x: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """Return the standard normal probability density at ``x``."""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi))


def _norm_ppf(p: float) -> float:
    """Return the standard normal percent point function at ``p``.

    Use the Abramowitz and Stegun rational approximation, accurate to about
    ``1e-5``.
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")

    if p < 0.5:
        return -_norm_ppf(1 - p)

    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)


def _discretize_ticks(
    positions: np.ndarray, labels: list[str], max_ticks: int = 10
) -> tuple[np.ndarray, list[str]]:
    """Choose regularly spaced axis ticks when labels would overlap.

    When many dates exist, show every Nth date and always retain the first and
    last labels.

    Parameters
    ----------
    positions : np.ndarray
        Tick positions (indices).
    labels : list[str]
        Tick labels corresponding to positions.
    max_ticks : int
        Maximum number of ticks to display.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Discretized positions and labels.
    """
    n = len(positions)
    if n <= max_ticks:
        return positions, labels

    # Choose an interval that keeps the label count near the requested limit.
    interval = max(1, int(np.ceil(n / max_ticks)))

    # Retain the endpoints so the displayed range remains clear.
    indices = list(range(0, n, interval))
    if n - 1 not in indices:
        indices.append(n - 1)

    positions_disc = positions[indices]
    labels_disc = [labels[i] for i in indices]

    return positions_disc, labels_disc


class NewsPlots:
    """Provide plots for a ``NewsData`` instance.

    The methods use the validated ``decompositions`` DataFrame in ``self.df``
    and the analysis methods supplied by ``NewsData``.
    """

    def plot_accuracy(
        self,
        realised: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
        show: bool = True,
    ):
        """Plot RMSE and MAE for each vintage date."""
        import matplotlib.pyplot as plt

        acc = self.accuracy_over_time(realised, variable=variable, source=source)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(acc["vintage_date"], acc["rmse"], "-o", label="RMSE")
        ax.plot(acc["vintage_date"], acc["mae"], "-s", label="MAE")
        ax.set_xlabel("Vintage date")
        ax.set_ylabel("Error")
        ax.set_title("Forecast accuracy over time")
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_contributions(
        self,
        variable: str | None = None,
        source: str | None = None,
        vintage_date=None,
        show: bool = True,
    ):
        """Plot level contributions by component as stacked bars."""
        import matplotlib.pyplot as plt

        lev = self._level(variable=variable, source=source)
        if vintage_date is not None:
            lev = lev[lev["vintage_date"] == pd.Timestamp(vintage_date)]

        pivot = lev.pivot_table(
            index="date", columns="component", values="contribution", aggfunc="sum"
        )
        pivot.index = pivot.index.strftime("%Y-%m-%d")

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot.bar(stacked=True, ax=ax)
        ax.set_ylabel("Contribution")
        ax.set_title("Level decomposition — contributions by component")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_signal_magnitude(
        self,
        variable: str | None = None,
        source: str | None = None,
        show: bool = True,
    ):
        """Plot each component's signal magnitude as a horizontal bar."""
        import matplotlib.pyplot as plt

        sig = self.signal_magnitude(variable=variable, source=source).sort_values()

        fig, ax = plt.subplots(figsize=(8, max(4, len(sig) * 0.4)))
        sig.plot.barh(ax=ax)
        ax.set_xlabel("Signal magnitude (mean |contribution|)")
        ax.set_title("Indicator signal magnitude")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_hit_rate(
        self,
        realised: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
        show: bool = True,
    ):
        """Plot each component's directional accuracy as a horizontal bar."""
        import matplotlib.pyplot as plt

        hr = self.hit_rate(realised, variable=variable, source=source).sort_values()

        fig, ax = plt.subplots(figsize=(8, max(4, len(hr) * 0.4)))
        hr.plot.barh(ax=ax, color="teal")
        ax.axvline(50, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Hit rate (%)")
        ax.set_title("Directional accuracy by component")
        ax.set_xlim(0, 100)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_error_improvement(
        self,
        realised: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
        show: bool = True,
    ):
        """Plot each component's average error improvement horizontally."""
        import matplotlib.pyplot as plt

        ei = self.error_improvement(
            realised, variable=variable, source=source
        ).sort_values()

        fig, ax = plt.subplots(figsize=(8, max(4, len(ei) * 0.4)))
        colors = ["green" if v > 0 else "red" for v in ei.values]
        ei.plot.barh(ax=ax, color=colors)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Error improvement")
        ax.set_title("Average error improvement by component")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_timing_decomposition(
        self,
        n_obs: pd.DataFrame,
        variable: str | None = None,
        source: str | None = None,
        show: bool = True,
    ):
        """Plot intrinsic content against the timing premium."""
        import matplotlib.pyplot as plt

        td = self.timing_decomposition(n_obs, variable=variable, source=source)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(td["beta"], td["alpha"], s=60)
        for _, row in td.iterrows():
            ax.annotate(
                row["component"],
                (row["beta"], row["alpha"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Timing premium (β)")
        ax.set_ylabel("Intrinsic content (α)")
        ax.set_title("Timing decomposition")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_indicators_over_time(
        self,
        realised: pd.DataFrame | pd.Series,
        min_periods: int = 4,
        variable: str | None = None,
        source: str | None = None,
        show: bool = True,
    ):
        """Plot indicator metrics as they change across vintage dates.

        The figure contains one subplot each for signal magnitude, directional
        accuracy, and error improvement. Each component has one line.
        """
        import matplotlib.pyplot as plt

        hist = self.indicator_table_over_time(
            realised, min_periods=min_periods, variable=variable, source=source
        )
        if hist.empty:
            print("Not enough data for historical indicator plot.")
            return None, None

        metrics = ["Signal magnitude", "Directional accuracy", "Error improvement"]
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 9))

        components = sorted(hist["component"].unique())
        for ax, metric in zip(axes, metrics):
            for comp in components:
                sub = hist[hist["component"] == comp]
                ax.plot(
                    sub["vintage_date"], sub[metric], "-o", markersize=3, label=comp
                )
            ax.set_ylabel(metric)
            ax.set_title(metric)
            if metric == "Directional accuracy":
                ax.axhline(50, color="grey", linestyle="--", linewidth=0.8)

        axes[-1].set_xlabel("Vintage date")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    # Nowcast plots.

    def plot_nowcast_evolution(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> tuple[Any, Any]:
        """Plot the nowcast path across vintages for one target quarter.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the most recent date.
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.

        Returns
        -------
        tuple[Any, Any]
        """
        import matplotlib.pyplot as plt

        evolution = self.nowcast_evolution(date=date, variable=variable, source=source)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Use integer positions so uneven date gaps do not distort the spacing.
        x = np.arange(len(evolution))
        vintage_labels = [
            pd.Timestamp(v).strftime("%d/%m/%y") for v in evolution["vintage_date"]
        ]

        # Draw the nowcast path.
        ax.plot(x, evolution["nowcast"], linewidth=2, marker="o", markersize=6)

        # Shade the area under the path.
        ax.fill_between(x, evolution["nowcast"], alpha=0.15)

        # Label each data point with its vintage date.
        ax.set_xticks(x)
        ax.set_xticklabels(vintage_labels, rotation=0, ha="center", fontsize=9)

        # Label the axes and identify the target quarter.
        ax.set_xlabel("Vintage Date", fontsize=11)
        ax.set_ylabel("GDP Growth (%)", fontsize=11)

        if date is not None:
            target_q = pd.Timestamp(date).to_period("Q")
        else:
            target_q = (
                evolution["vintage_date"].max().to_period("Q")
                if not evolution.empty
                else "N/A"
            )

        ax.set_title(
            f"Nowcast Evolution — {target_q}",
            fontsize=14,
            fontweight="bold",
        )

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_contributions_by_vintage(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> tuple[Any, Any]:
        """Plot component contributions across vintages for one target quarter.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the most recent date.
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.

        Returns
        -------
        tuple[Any, Any]
        """
        import matplotlib.pyplot as plt

        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        lev = self._level(variable=variable, source=source)
        lev = lev[lev["date"] == date]
        lev = lev[~lev["component"].isin(["intercept", "residual"])]

        pivot = (
            lev.groupby(["vintage_date", "component"])["contribution"]
            .sum()
            .reset_index()
            .pivot(index="vintage_date", columns="component", values="contribution")
            .fillna(0)
            .sort_index()
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        components = pivot.columns.tolist()
        x = np.arange(len(pivot))
        width = 0.6

        bottom_pos = np.zeros(len(pivot))
        bottom_neg = np.zeros(len(pivot))

        prop_cycler = plt.rcParams["axes.prop_cycle"]
        colors_cycle = prop_cycler.by_key()["color"] if prop_cycler else []

        for i, comp in enumerate(components):
            values = pivot[comp].values
            color = colors_cycle[i % len(colors_cycle)] if colors_cycle else None

            pos_vals = np.where(values >= 0, values, 0)
            neg_vals = np.where(values < 0, values, 0)

            if pos_vals.any():
                ax.bar(
                    x,
                    pos_vals,
                    width,
                    bottom=bottom_pos,
                    label=comp,
                    color=color,
                    edgecolor="white",
                )
                bottom_pos += pos_vals
            if neg_vals.any():
                ax.bar(
                    x,
                    neg_vals,
                    width,
                    bottom=bottom_neg,
                    label=comp if not pos_vals.any() else "",
                    color=color,
                    edgecolor="white",
                )
                bottom_neg += neg_vals

        ax.axhline(0, color="black", linewidth=0.8)
        vintage_labels = [pd.Timestamp(v).strftime("%d/%m/%y") for v in pivot.index]
        ax.set_xticks(x)
        ax.set_xticklabels(vintage_labels, rotation=0, ha="center", fontsize=9)

        target_q = pd.Timestamp(date).to_period("Q")
        ax.set_xlabel("Vintage Date", fontsize=11)
        ax.set_ylabel("Contribution (pp)", fontsize=11)
        ax.set_title(
            f"Indicator Contributions — {target_q}",
            fontsize=14,
            fontweight="bold",
        )
        # Keep the legend outside the plot so it does not cover the bars.
        ax.legend(
            title="Component",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=9,
            frameon=True,
        )

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_revision_evolution(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> tuple[Any, Any]:
        """Plot cumulative news revisions across vintages for one target quarter.

        Revision-space analog of :meth:`plot_nowcast_evolution`.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the most recent date.
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.

        Returns
        -------
        tuple[Any, Any]
        """
        import matplotlib.pyplot as plt

        evolution = self.revision_evolution(date=date, variable=variable, source=source)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        if not evolution.empty:
            ax.plot(
                evolution["vintage_date"],
                evolution["cumulative_revision"],
                linewidth=2,
                marker="o",
                markersize=6,
            )
            ax.fill_between(
                evolution["vintage_date"],
                evolution["cumulative_revision"],
                alpha=0.15,
            )

        ax.axhline(0, color="black", linewidth=0.8)

        target_q = (
            pd.Timestamp(date).to_period("Q")
            if date is not None
            else (
                evolution["vintage_date"].max().to_period("Q")
                if not evolution.empty
                else "N/A"
            )
        )
        ax.set_xlabel("Vintage Date", fontsize=11)
        ax.set_ylabel("Cumulative Revision (pp)", fontsize=11)
        ax.set_title(
            f"Cumulative News Revision — {target_q}",
            fontsize=14,
            fontweight="bold",
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_revision_contributions_by_vintage(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> tuple[Any, Any]:
        """Plot per-component news revisions across vintages for one quarter.

        Revision-space analog of :meth:`plot_contributions_by_vintage`.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the most recent date.
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.

        Returns
        -------
        tuple[Any, Any]
        """
        import matplotlib.pyplot as plt

        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        rev = self._revision(variable=variable, source=source)
        rev = rev[
            (rev["date"] == date)
            & (rev["revision_source"] == "news")
            & (~rev["component"].isin(["intercept", "residual"]))
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        if rev.empty:
            target_q = pd.Timestamp(date).to_period("Q")
            ax.set_title(
                f"Revision Contributions — {target_q}",
                fontsize=14,
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.5,
                "No revision data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            if show:
                plt.show()
            return fig, ax

        pivot = (
            rev.groupby(["vintage_date", "component"])["contribution"]
            .sum()
            .reset_index()
            .pivot(index="vintage_date", columns="component", values="contribution")
            .fillna(0)
            .sort_index()
        )

        components = pivot.columns.tolist()
        x = np.arange(len(pivot))
        width = 0.6
        bottom_pos = np.zeros(len(pivot))
        bottom_neg = np.zeros(len(pivot))

        prop_cycler = plt.rcParams["axes.prop_cycle"]
        colors_cycle = prop_cycler.by_key()["color"] if prop_cycler else []

        for i, comp in enumerate(components):
            values = pivot[comp].values
            color = colors_cycle[i % len(colors_cycle)] if colors_cycle else None
            pos_vals = np.where(values >= 0, values, 0)
            neg_vals = np.where(values < 0, values, 0)
            if pos_vals.any():
                ax.bar(
                    x,
                    pos_vals,
                    width,
                    bottom=bottom_pos,
                    label=comp,
                    color=color,
                    edgecolor="white",
                )
                bottom_pos += pos_vals
            if neg_vals.any():
                ax.bar(
                    x,
                    neg_vals,
                    width,
                    bottom=bottom_neg,
                    label=comp if not pos_vals.any() else "",
                    color=color,
                    edgecolor="white",
                )
                bottom_neg += neg_vals

        ax.axhline(0, color="black", linewidth=0.8)
        vintage_labels = [pd.Timestamp(v).strftime("%d/%m/%y") for v in pivot.index]
        ax.set_xticks(x)
        ax.set_xticklabels(vintage_labels, rotation=0, ha="center", fontsize=9)

        target_q = pd.Timestamp(date).to_period("Q")
        ax.set_xlabel("Vintage Date", fontsize=11)
        ax.set_ylabel("News Revision (pp)", fontsize=11)
        ax.set_title(
            f"Revision Contributions — {target_q}",
            fontsize=14,
            fontweight="bold",
        )
        # Keep the legend outside the plot so it does not cover the bars.
        ax.legend(
            title="Component",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=9,
            frameon=True,
        )
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_raw_revision_contributions(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 5),
        x_order: Any = None,
    ) -> tuple[Any, Any]:
        """Plot raw nowcast revisions by indicator with the net change overlaid.

        The bars use first differences of the ``level`` contributions from
        :meth:`raw_revision_contributions`. Their sum equals the net change in
        the nowcast at each vintage.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the most recent date.
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.
        x_order : Any
            Optional vintage order used to align multiple plots.

        Returns
        -------
        tuple[Any, Any]
        """
        date = self._validate_date(date)
        contrib = self.raw_revision_contributions(date, variable, source)

        pivot = None
        if not contrib.empty:
            pivot = (
                contrib.pivot(
                    index="vintage_date", columns="component", values="contribution"
                )
                .fillna(0.0)
                .sort_index()
            )
            # Hide unchanged vintages unless an explicit order must preserve
            # their positions for another panel.
            if x_order is None:
                pivot = pivot[pivot.abs().sum(axis=1) > 1e-12]

        target_q = pd.Timestamp(date).to_period("Q")
        return self._plot_stacked_vintage_bars(
            pivot,
            ax=ax,
            title=f"Raw Revision Contributions — {target_q}",
            ylabel="Revision (pp)",
            net_label="net revision",
            figsize=figsize,
            show=show,
            x_order=x_order,
        )

    def plot_nowcast_contributions(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 5),
        x_order: Any = None,
    ) -> tuple[Any, Any]:
        """Plot level contributions across vintages with the nowcast overlaid.

        This combines the line from :meth:`plot_nowcast_evolution` with the
        bars from :meth:`plot_contributions_by_vintage`. The bars sum to the
        nowcast at every vintage.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the most recent date.
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.
        x_order : Any
            Optional vintage order used to align multiple plots.

        Returns
        -------
        tuple[Any, Any]
        """
        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        lev = self._level(variable=variable, source=source)
        lev = lev[lev["date"] == date]

        pivot = None
        if not lev.empty:
            pivot = (
                lev.groupby(["vintage_date", "component"])["contribution"]
                .sum()
                .unstack("component")
                .fillna(0.0)
                .sort_index()
            )

        target_q = pd.Timestamp(date).to_period("Q")
        return self._plot_stacked_vintage_bars(
            pivot,
            ax=ax,
            title=f"Nowcast Contributions — {target_q}",
            ylabel="Contribution (pp)",
            net_label="nowcast",
            figsize=figsize,
            show=show,
            x_order=x_order,
        )

    def _plot_stacked_vintage_bars(
        self,
        pivot,
        ax,
        *,
        title: str,
        ylabel: str,
        net_label: str,
        figsize: tuple[float, float],
        show: bool,
        x_order=None,
    ):
        """Render signed stacked bars for each vintage with a net line.

        ``pivot`` is a ``vintage_date × component`` DataFrame of contributions,
        or ``None``/empty. The line shows each row sum, so it reconciles to the
        bars. When ``x_order`` is provided, reindex the pivot so multiple
        panels share x positions.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        if pivot is None or pivot.empty:
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            if show:
                plt.show()
            return fig, ax

        if x_order is not None:
            pivot = pivot.reindex([pd.Timestamp(v) for v in x_order])
            # Keep missing vintages in the layout without drawing their values.
            present = pivot.notna().any(axis=1).to_numpy()
            pivot = pivot.fillna(0.0)
        else:
            present = np.ones(len(pivot), dtype=bool)

        x = np.arange(len(pivot))
        width = 0.6
        bottom_pos = np.zeros(len(pivot))
        bottom_neg = np.zeros(len(pivot))

        prop_cycler = plt.rcParams["axes.prop_cycle"]
        colors_cycle = prop_cycler.by_key()["color"] if prop_cycler else []

        for i, comp in enumerate(pivot.columns):
            values = pivot[comp].to_numpy()
            color = colors_cycle[i % len(colors_cycle)] if colors_cycle else None
            pos_vals = np.where(values >= 0, values, 0.0)
            neg_vals = np.where(values < 0, values, 0.0)
            if pos_vals.any():
                ax.bar(
                    x,
                    pos_vals,
                    width,
                    bottom=bottom_pos,
                    label=comp,
                    color=color,
                    edgecolor="white",
                    alpha=0.7,
                )
                bottom_pos += pos_vals
            if neg_vals.any():
                ax.bar(
                    x,
                    neg_vals,
                    width,
                    bottom=bottom_neg,
                    label=comp if not pos_vals.any() else "",
                    color=color,
                    edgecolor="white",
                    alpha=0.7,
                )
                bottom_neg += neg_vals

        line_vals = pivot.sum(axis=1).to_numpy().astype(float)
        line_vals[~present] = np.nan
        ax.plot(
            x,
            line_vals,
            "k-o",
            linewidth=1.5,
            label=net_label,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        vintage_labels = [pd.Timestamp(v).strftime("%d/%m/%y") for v in pivot.index]
        ax.set_xticks(x)
        ax.set_xticklabels(vintage_labels, rotation=0, ha="center", fontsize=9)
        ax.set_xlabel("Vintage Date", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(
            title="Component",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=9,
            frameon=True,
        )
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_revision_by_source(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> tuple[Any, Any]:
        """Plot revision contributions grouped by revision source.

        Show news, reestimation, and interaction contributions at each vintage.
        Their sum equals the nowcast change between consecutive vintages.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the most recent date.
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.

        Returns
        -------
        tuple[Any, Any]
        """
        import matplotlib.pyplot as plt

        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        rev = self._revision(variable=variable, source=source)
        rev = rev[rev["date"] == date]

        # Put one revision source in each column for stacked plotting.
        pivot = (
            rev.groupby(["vintage_date", "revision_source"])["contribution"]
            .sum()
            .unstack(fill_value=0.0)
            .sort_index()
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        if pivot.empty:
            ax.set_title("No revision data", fontsize=14)
            if show:
                plt.show()
            return fig, ax

        sources = pivot.columns.tolist()
        x = np.arange(len(pivot))
        width = 0.6

        bottom_pos = np.zeros(len(pivot))
        bottom_neg = np.zeros(len(pivot))

        prop_cycler = plt.rcParams["axes.prop_cycle"]
        colors_cycle = prop_cycler.by_key()["color"] if prop_cycler else []

        for i, src in enumerate(sources):
            values = pivot[src].values
            color = colors_cycle[i % len(colors_cycle)] if colors_cycle else None

            pos_vals = np.where(values >= 0, values, 0)
            neg_vals = np.where(values < 0, values, 0)

            if pos_vals.any():
                ax.bar(
                    x,
                    pos_vals,
                    width,
                    bottom=bottom_pos,
                    label=src,
                    color=color,
                    edgecolor="white",
                )
                bottom_pos += pos_vals
            if neg_vals.any():
                ax.bar(
                    x,
                    neg_vals,
                    width,
                    bottom=bottom_neg,
                    label=src if not pos_vals.any() else "",
                    color=color,
                    edgecolor="white",
                )
                bottom_neg += neg_vals

        ax.axhline(0, color="black", linewidth=0.8)
        vintage_labels = [pd.Timestamp(v).strftime("%d/%m/%y") for v in pivot.index]
        ax.set_xticks(x)
        ax.set_xticklabels(vintage_labels, rotation=0, ha="center", fontsize=9)

        target_q = pd.Timestamp(date).to_period("Q")
        ax.set_xlabel("Vintage Date", fontsize=11)
        ax.set_ylabel("Revision (pp)", fontsize=11)
        ax.set_title(
            f"Revision by Source — {target_q}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(
            title="Source",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=9,
            frameon=True,
        )

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_revision_impacts(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (10, 5),
    ) -> tuple[Any, Any]:
        """Plot each component's cumulative impact on the nowcast revision.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            Target quarter. ``None`` selects the most recent date.
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.

        Returns
        -------
        tuple[Any, Any]
        """
        import matplotlib.pyplot as plt

        impacts = self.cumulative_revision_impacts(
            date=date, variable=variable, source=source
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Colour positive and negative impacts differently.
        impacts_sorted = impacts.sort_values("cumulative_impact")

        colors = [
            "green" if x > 0 else "red" for x in impacts_sorted["cumulative_impact"]
        ]

        ax.barh(
            impacts_sorted["component"],
            impacts_sorted["cumulative_impact"],
            color=colors,
        )

        ax.set_xlabel("Cumulative Impact (pp)", fontsize=11)
        ax.set_ylabel("Component", fontsize=11)

        if date is not None:
            target_q = pd.Timestamp(date).to_period("Q")
        else:
            target_q = "Latest"

        ax.set_title(
            f"Cumulative Revision Impacts — {target_q}",
            fontsize=14,
            fontweight="bold",
        )

        ax.axvline(0, color="black", linewidth=0.8)

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_contribution_decomposition(
        self,
        variable: str | None = None,
        source: str | None = None,
        ax: Any = None,
        show: bool = True,
        figsize: tuple[float, float] = (12, 6),
        n_dates: int = 10,
    ) -> tuple[Any, Any]:
        """Plot level contributions by component across target dates.

        Parameters
        ----------
        variable : str | None
            Variable to plot. ``None`` selects the first variable.
        source : str | None
            Model source. ``None`` selects the first source.
        ax : Any
            Matplotlib axes to use. ``None`` creates a new figure.
        show : bool
            Display the plot immediately when ``True``.
        figsize : tuple[float, float]
            Figure size when creating a new figure.
        n_dates : int
            Number of recent target dates to show.

        Returns
        -------
        tuple[Any, Any]
        """
        import matplotlib.pyplot as plt

        lev = self._level(variable=variable, source=source)

        # Keep the requested number of recent target dates.
        dates = sorted(lev["date"].unique())[-n_dates:]
        lev = lev[lev["date"].isin(dates)]

        # Sum contributions by target date and component.
        pivot = (
            lev.groupby(["date", "component"])["contribution"]
            .sum()
            .reset_index()
            .pivot(index="date", columns="component", values="contribution")
            .fillna(0)
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Draw signed stacked bars.
        components = pivot.columns.tolist()
        x = np.arange(len(pivot))
        width = 0.6

        # Stack positive and negative values on their respective sides.
        bottom_pos = np.zeros(len(pivot))
        bottom_neg = np.zeros(len(pivot))

        # Reuse the active Matplotlib colour cycle.
        prop_cycler = plt.rcParams["axes.prop_cycle"]
        colors_cycle = prop_cycler.by_key()["color"] if prop_cycler else []

        for i, comp in enumerate(components):
            values = pivot[comp].values
            color = colors_cycle[i % len(colors_cycle)] if colors_cycle else None

            # Positive values stack upward.
            pos_vals = np.where(values >= 0, values, 0)
            # Negative values stack downward.
            neg_vals = np.where(values < 0, values, 0)

            if pos_vals.any():
                ax.bar(
                    x,
                    pos_vals,
                    width,
                    bottom=bottom_pos,
                    label=comp,
                    color=color,
                    edgecolor="white",
                )
                bottom_pos += pos_vals

            if neg_vals.any():
                ax.bar(
                    x,
                    neg_vals,
                    width,
                    bottom=bottom_neg,
                    label=comp if not pos_vals.any() else "",
                    color=color,
                    edgecolor="white",
                )
                bottom_neg += neg_vals

        # Configure labels and the target-quarter axis.
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        date_labels = [str(pd.Timestamp(d).to_period("Q")) for d in pivot.index]
        ax.set_xticklabels(date_labels, rotation=0, ha="center", fontsize=10)

        ax.set_xlabel("Target Quarter", fontsize=11)
        ax.set_ylabel("Contribution (pp)", fontsize=11)
        ax.set_title(
            "Nowcast Decomposition by Indicator",
            fontsize=14,
            fontweight="bold",
        )

        ax.legend(title="Component", loc="upper left", fontsize=9)

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax
