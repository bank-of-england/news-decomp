"""Analysis methods for :class:`~news_decomp.NewsData`."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd


class NewsAnalysis:
    """Provide forecast, accuracy, timing, and revision metrics.

    The methods use ``self.df``, the validated ``decompositions`` DataFrame
    that :class:`~news_decomp.NewsData` assigns during construction.
    """

    # Internal helpers.

    _GROUP_KEYS: ClassVar[list[str]] = [
        "variable",
        "date",
        "forecast_horizon",
        "source",
        "vintage_date",
    ]

    def _filter(
        self,
        decomposition: str | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        df = self.df
        if decomposition is not None:
            df = df[df["decomposition"] == decomposition]
        if variable is not None:
            df = df[df["variable"] == variable]
        if source is not None:
            df = df[df["source"] == source]
        return df

    def _level(self, **kw: str | None) -> pd.DataFrame:
        return self._filter(decomposition="level", **kw)

    def _revision(self, **kw: str | None) -> pd.DataFrame:
        return self._filter(decomposition="revision", **kw)

    @staticmethod
    def _align_realised(
        forecasts: pd.DataFrame,
        realised: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame:
        """Join forecast values to realised values by date and variable.

        ``realised`` may be a Series indexed by ``date`` or a DataFrame with
        columns ``[variable, date, value]``.
        """
        if isinstance(realised, pd.Series):
            realised = realised.rename("value").reset_index()
            realised.columns = ["date", "value"]
        merge_on = ["date"]
        if "variable" in realised.columns:
            merge_on.append("variable")
        return forecasts.merge(realised, on=merge_on, how="inner")

    # Forecast aggregation.

    def forecasts(
        self,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Sum level contributions into one forecast for each group.

        The result has columns ``[variable, date, forecast_horizon, source,
        vintage_date, forecast]``.
        """
        lev = self._level(variable=variable, source=source)
        return (
            lev.groupby(self._GROUP_KEYS, as_index=False)["contribution"]
            .sum()
            .rename(columns={"contribution": "forecast"})
        )

    # Model accuracy.

    def rmse(
        self,
        realised: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
    ) -> float:
        r"""Return the root mean squared error across matched forecasts.

        $$\mathrm{RMSE} = \sqrt{\frac{1}{T}\sum(y_\tau - \hat y_\tau)^2}$$
        """
        merged = self._align_realised(
            self.forecasts(variable=variable, source=source), realised
        )
        return float(np.sqrt(((merged["value"] - merged["forecast"]) ** 2).mean()))

    def mae(
        self,
        realised: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
    ) -> float:
        r"""Return the mean absolute error across matched forecasts.

        $$\mathrm{MAE} = \frac{1}{T}\sum |y_\tau - \hat y_\tau|$$
        """
        merged = self._align_realised(
            self.forecasts(variable=variable, source=source), realised
        )
        return float((merged["value"] - merged["forecast"]).abs().mean())

    def accuracy_over_time(
        self,
        realised: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Compute RMSE and MAE separately for each vintage.

        The result has columns ``[vintage_date, rmse, mae]``.
        """
        merged = self._align_realised(
            self.forecasts(variable=variable, source=source), realised
        )
        merged["error"] = merged["value"] - merged["forecast"]

        def _agg(g: pd.DataFrame) -> pd.Series:
            return pd.Series(
                {
                    "rmse": np.sqrt((g["error"] ** 2).mean()),
                    "mae": g["error"].abs().mean(),
                }
            )

        return merged.groupby("vintage_date").apply(_agg).reset_index()

    # Indicator usefulness.

    def marginal_contributions(
        self,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Return each component's marginal contribution $\Delta_{j,\tau}$.

        For level rows, the marginal contribution is the ``contribution``
        value.
        """
        return self._level(variable=variable, source=source)[
            self._GROUP_KEYS + ["component", "contribution"]
        ].copy()

    def signal_magnitude(
        self,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.Series:
        r"""Measure the mean absolute contribution of each component.

        $$V_j^{\mathrm{abs}} = \frac{1}{T}\sum_\tau |\Delta_{j,\tau}|$$
        """
        mc = self.marginal_contributions(variable=variable, source=source)
        return mc.groupby("component")["contribution"].apply(lambda s: s.abs().mean())

    def hit_rate(
        self,
        realised: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.Series:
        r"""Measure how often each component moves the forecast closer to truth.

        $$H_j = \frac{1}{T}\sum_\tau \mathbf{1}(|y - \hat y| < |y - \hat y^{(-j)}|) \times 100$$
        """
        lev = self._level(variable=variable, source=source)
        fcst = self.forecasts(variable=variable, source=source)
        merged = self._align_realised(fcst, realised)

        comp = lev[self._GROUP_KEYS + ["component", "contribution"]].merge(
            merged[self._GROUP_KEYS + ["forecast", "value"]],
            on=self._GROUP_KEYS,
        )
        comp["err_full"] = (comp["value"] - comp["forecast"]).abs()
        comp["err_without"] = (
            comp["value"] - (comp["forecast"] - comp["contribution"])
        ).abs()
        comp["hit"] = (comp["err_full"] < comp["err_without"]).astype(int)
        return comp.groupby("component")["hit"].mean() * 100

    def error_improvement(
        self,
        realised: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.Series:
        r"""Measure each component's average reduction in absolute error.

        $$E_j = \frac{1}{T}\sum_\tau [|y - \hat y^{(-j)}| - |y - \hat y|]$$

        Positive values mean component *j* improves the forecast on average.
        """
        lev = self._level(variable=variable, source=source)
        fcst = self.forecasts(variable=variable, source=source)
        merged = self._align_realised(fcst, realised)

        comp = lev[self._GROUP_KEYS + ["component", "contribution"]].merge(
            merged[self._GROUP_KEYS + ["forecast", "value"]],
            on=self._GROUP_KEYS,
        )
        comp["err_full"] = (comp["value"] - comp["forecast"]).abs()
        comp["err_without"] = (
            comp["value"] - (comp["forecast"] - comp["contribution"])
        ).abs()
        comp["improvement"] = comp["err_without"] - comp["err_full"]
        return comp.groupby("component")["improvement"].mean()

    # Timing decomposition.

    def timing_decomposition(
        self,
        n_obs: pd.DataFrame,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Separate signal content from the effect of observation timing.

        $$V_{j,k}^{\mathrm{abs}} = \alpha_j + \beta_j\, n_{j,k} + \eta_{j,k}$$

        Parameters
        ----------
        n_obs : pd.DataFrame
            DataFrame with columns ``[component, vintage_date, n]`` giving the
            number of within-quarter observations available for each indicator
            at each evaluation point.
        variable : str | None
            Target variable to include. ``None`` includes all variables.
        source : str | None
            Model source to include. ``None`` includes all sources.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``[component, alpha, beta]``. ``alpha``
            measures intrinsic content; ``beta`` measures the timing premium.
        """
        lev = self._level(variable=variable, source=source)
        v_jk = (
            lev.groupby(["component", "vintage_date"])["contribution"]
            .apply(lambda s: s.abs().mean())
            .reset_index(name="v_abs")
        )
        merged = v_jk.merge(n_obs, on=["component", "vintage_date"], how="inner")

        results = []
        for comp, grp in merged.groupby("component"):
            if len(grp) < 2:
                results.append(
                    {
                        "component": comp,
                        "alpha": float(grp["v_abs"].mean()),
                        "beta": 0.0,
                    }
                )
                continue
            x = grp["n"].values.astype(float)
            y = grp["v_abs"].values.astype(float)
            X = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            results.append({"component": comp, "alpha": coef[0], "beta": coef[1]})

        return pd.DataFrame(results)

    # Information density.

    def information_density(
        self,
        pub_delays: pd.Series | dict[str, float],
        n_obs: pd.DataFrame | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Measure each indicator's signal per week of publication delay.

        $$D_j = V_j^{\mathrm{abs}} / w_j$$

        Parameters
        ----------
        pub_delays : pd.Series | dict[str, float]
            Mapping ``component → w_j``, where ``w_j`` is the publication
            delay in weeks.
        n_obs : pd.DataFrame | None
            If provided, also computes ``D_j^* = \alpha_j / w_j`` via
            :meth:`timing_decomposition`.
        variable : str | None
            Target variable to include. ``None`` includes all variables.
        source : str | None
            Model source to include. ``None`` includes all sources.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``[component, v_abs, w, density]`` and
            optionally ``[alpha, density_star]``.
        """
        if isinstance(pub_delays, dict):
            pub_delays = pd.Series(pub_delays, name="w")
        pub_delays = pub_delays.rename("w")

        sig = self.signal_magnitude(variable=variable, source=source).rename("v_abs")
        delays_df = pub_delays.reset_index()
        delays_df.columns = ["component", "w"]
        out = sig.reset_index().merge(delays_df, on="component", how="inner")
        out["density"] = out["v_abs"] / out["w"]

        if n_obs is not None:
            td = self.timing_decomposition(n_obs, variable=variable, source=source)
            out = out.merge(td[["component", "alpha"]], on="component", how="left")
            out["density_star"] = out["alpha"] / out["w"]

        return out

    # Real-time and revision metrics.

    def revision_predictability(
        self,
        realised_vintages: dict[str, pd.DataFrame | pd.Series],
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Measure how often each component explains a realised revision.

        $$P_j^{(v)} = \frac{1}{T}\sum_\tau \mathbf{1}(|r_\tau^{(v)} - \Delta_{j,\tau}| < |r_\tau^{(v)}|) \times 100$$

        Parameters
        ----------
        realised_vintages : dict[str, pd.DataFrame | pd.Series]
            Ordered mapping from vintage labels to realised Series or
            DataFrames. The method compares consecutive vintages.
        variable : str | None
            Target variable to include. ``None`` includes all variables.
        source : str | None
            Model source to include. ``None`` includes all sources.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``[component, <revision_label>, ...]``.
        """
        labels = list(realised_vintages.keys())
        lev = self._level(variable=variable, source=source)
        mc = lev[self._GROUP_KEYS + ["component", "contribution"]]

        results = {}
        for i in range(1, len(labels)):
            prev_label, curr_label = labels[i - 1], labels[i]
            rev_label = f"{prev_label}→{curr_label}"

            prev_r = realised_vintages[prev_label]
            curr_r = realised_vintages[curr_label]
            if isinstance(prev_r, pd.Series):
                prev_r = prev_r.rename("value_prev").reset_index()
                prev_r.columns = ["date", "value_prev"]
            else:
                prev_r = prev_r.rename(columns={"value": "value_prev"})
            if isinstance(curr_r, pd.Series):
                curr_r = curr_r.rename("value_curr").reset_index()
                curr_r.columns = ["date", "value_curr"]
            else:
                curr_r = curr_r.rename(columns={"value": "value_curr"})

            merge_on = ["date"]
            revisions = prev_r.merge(curr_r, on=merge_on)
            revisions["revision"] = revisions["value_curr"] - revisions["value_prev"]

            comp = mc.merge(revisions[["date", "revision"]], on="date", how="inner")
            comp["closer"] = (
                (comp["revision"] - comp["contribution"]).abs() < comp["revision"].abs()
            ).astype(int)
            results[rev_label] = comp.groupby("component")["closer"].mean() * 100

        return pd.DataFrame(results)

    def news_vs_noise_r2(
        self,
        realised_first: pd.DataFrame | pd.Series,
        realised_final: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.Series:
        r"""Return each indicator's partial R² for target revisions.

        Regresses $r_\tau = y^{(V)}_\tau - y^{(0)}_\tau$ on $y^{(0)}_\tau$
        and $\Delta_{j,\tau}$, returning the partial R² for each component.

        Parameters
        ----------
        realised_first : pd.DataFrame | pd.Series
            First-release values of the target.
        realised_final : pd.DataFrame | pd.Series
            Final-revised values of the target.
        variable : str | None
            Target variable to include. ``None`` includes all variables.
        source : str | None
            Model source to include. ``None`` includes all sources.

        Returns
        -------
        pd.Series
            Series mapping each component to its partial R².
        """
        if isinstance(realised_first, pd.Series):
            realised_first = realised_first.rename("y0").reset_index()
            realised_first.columns = ["date", "y0"]
        else:
            realised_first = realised_first.rename(columns={"value": "y0"})
        if isinstance(realised_final, pd.Series):
            realised_final = realised_final.rename("yV").reset_index()
            realised_final.columns = ["date", "yV"]
        else:
            realised_final = realised_final.rename(columns={"value": "yV"})

        revisions = realised_first.merge(realised_final, on="date")
        revisions["revision"] = revisions["yV"] - revisions["y0"]

        lev = self._level(variable=variable, source=source)
        mc = lev[self._GROUP_KEYS + ["component", "contribution"]]

        merged = mc.merge(revisions[["date", "y0", "revision"]], on="date", how="inner")

        results = {}
        for comp, grp in merged.groupby("component"):
            if len(grp) < 3:
                results[comp] = np.nan
                continue
            r = grp["revision"].values
            y0 = grp["y0"].values
            delta = grp["contribution"].values

            # Fit the baseline model with the initial target value.
            X_base = np.column_stack([np.ones_like(y0), y0])
            coef_base, *_ = np.linalg.lstsq(X_base, r, rcond=None)
            ss_res_base = ((r - X_base @ coef_base) ** 2).sum()

            # Add the component contribution to measure its incremental fit.
            X_full = np.column_stack([np.ones_like(y0), y0, delta])
            coef_full, *_ = np.linalg.lstsq(X_full, r, rcond=None)
            ss_res_full = ((r - X_full @ coef_full) ** 2).sum()

            ss_tot = ((r - r.mean()) ** 2).sum()
            if ss_tot < 1e-12:
                results[comp] = np.nan
            else:
                partial_r2 = (ss_res_base - ss_res_full) / ss_res_base
                results[comp] = max(0.0, partial_r2)

        return pd.Series(results, name="partial_r2")

    def realtime_error_improvement(
        self,
        realised_final: pd.DataFrame | pd.Series,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.Series:
        r"""Measure error improvement against the final target vintage.

        $$E_j^{\mathrm{rt}} = \frac{1}{T}\sum_\tau [|y^{(V)}_\tau - \hat y^{(-j)}| - |y^{(V)}_\tau - \hat y|]$$

        Identical to :meth:`error_improvement` but explicitly named for
        when ``realised_final`` is the final-revised target.
        """
        return self.error_improvement(realised_final, variable=variable, source=source)

    def vintage_revision_contribution(
        self,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Split revision rows into new-data and parameter-change effects.

        Uses ``revision_source`` to split:
        - $\Delta_{j,\tau}^{(v)}$ = ``"news"`` rows (new data arrived)
        - $\Gamma_{j,\tau}^{(v)}$ = ``"reestimation"`` + ``"interaction"`` rows

        Parameters
        ----------
        variable : str | None
            Target variable to include. ``None`` includes all variables.
        source : str | None
            Model source to include. ``None`` includes all sources.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns
            ``[component, new_data_contribution, revised_data_contribution]``.
        """
        rev = self._revision(variable=variable, source=source)
        if rev.empty:
            return pd.DataFrame(
                columns=[
                    "component",
                    "new_data_contribution",
                    "revised_data_contribution",
                ]
            )

        new_data = (
            rev[rev["revision_source"] == "news"]
            .groupby("component")["contribution"]
            .mean()
            .rename("new_data_contribution")
        )
        revised_data = (
            rev[rev["revision_source"].isin(["reestimation", "interaction"])]
            .groupby("component")["contribution"]
            .mean()
            .rename("revised_data_contribution")
        )
        return (
            pd.DataFrame(
                {
                    "new_data_contribution": new_data,
                    "revised_data_contribution": revised_data,
                }
            )
            .fillna(0.0)
            .reset_index()
        )

    # Nowcast analysis.

    def nowcast_evolution(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Return the nowcast at each vintage for one target date.

        $$\hat y_{t,k} = \sum_j \Delta_{j,t,k}$$

        The result has columns ``[vintage_date, nowcast]``.
        """
        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        lev = self._level(variable=variable, source=source)
        evolution = (
            lev[lev["date"] == date]
            .groupby("vintage_date")["contribution"]
            .sum()
            .reset_index(name="nowcast")
            .sort_values("vintage_date")
        )
        return evolution

    def raw_revision_contributions(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Return each component's contribution to the raw nowcast revision.

        Unlike :meth:`revision_impacts` (which uses the ``revision`` rows'
        ``news``/``reestimation``/``interaction`` split, and does not sum to
        the change in the nowcast), this is built purely from the ``level``
        decomposition. Each component's contribution to the revision between
        consecutive vintages is the first difference of its level
        contribution,

        $$\rho_{j,t,k} = \Delta_{j,t,k} - \Delta_{j,t,k-1},$$

        so the components sum **exactly** to the change in the nowcast,
        $\sum_j \rho_{j,t,k} = \hat y_{t,k} - \hat y_{t,k-1}$ -- i.e. the
        first difference of :meth:`nowcast_evolution`. It reconciles by
        construction and always matches the direction of the nowcast.

        The result has columns ``[base_vintage_date, vintage_date, component,
        contribution]``.
        """
        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        lev = self._level(variable=variable, source=source)
        lev = lev[lev["date"] == date]
        if lev.empty:
            return pd.DataFrame(
                columns=[
                    "base_vintage_date",
                    "vintage_date",
                    "component",
                    "contribution",
                ]
            )

        # Arrange one component per column so each row represents one vintage.
        wide = (
            lev.groupby(["vintage_date", "component"])["contribution"]
            .sum()
            .unstack("component")
            .fillna(0.0)
            .sort_index()
        )

        # First differences give component revisions; the first vintage has no
        # predecessor and is therefore dropped.
        diffs = wide.diff().iloc[1:]
        diffs.insert(0, "base_vintage_date", wide.index[:-1])

        out = diffs.reset_index().melt(
            id_vars=["vintage_date", "base_vintage_date"],
            var_name="component",
            value_name="contribution",
        )
        return (
            out[["base_vintage_date", "vintage_date", "component", "contribution"]]
            .sort_values(["vintage_date", "component"])
            .reset_index(drop=True)
        )

    def revision_evolution(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Return the cumulative news revision at each vintage.

        $$R_{t,k} = \sum_{k'\leq k} \sum_j \delta_{j,t,k'}$$

        The result has columns ``[vintage_date, cumulative_revision]``.
        """
        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        rev = self._revision(variable=variable, source=source)
        rev = rev[
            (rev["date"] == date)
            & (rev["revision_source"] == "news")
            & (~rev["component"].isin(["intercept", "residual"]))
        ]

        if rev.empty:
            return pd.DataFrame(columns=["vintage_date", "cumulative_revision"])

        per_vintage = rev.groupby("vintage_date")["contribution"].sum().sort_index()
        cumulative = per_vintage.cumsum().reset_index(name="cumulative_revision")
        return cumulative

    def revision_impacts(
        self,
        date: pd.Timestamp | str | None = None,
        vintage_date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Return each component's impact on forecast revisions.

        If ``vintage_date`` is provided, return the revision into that vintage.
        Otherwise, aggregate all revisions.

        The result has columns ``[component, impact, revision_source]``.
        """
        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        rev = self._revision(variable=variable, source=source)
        rev = rev[rev["date"] == date]

        if vintage_date is not None:
            rev = rev[rev["vintage_date"] == pd.Timestamp(vintage_date)]

        # Use news contributions when they exist because they represent data releases.
        news = rev[rev["revision_source"] == "news"]

        if news.empty:
            impacts = (
                rev.groupby(["component", "revision_source"])["contribution"]
                .sum()
                .reset_index(name="impact")
            )
        else:
            impacts = (
                news.groupby(["component", "revision_source"])["contribution"]
                .sum()
                .reset_index(name="impact")
            )

        return impacts.sort_values("impact", key=abs, ascending=False)

    def cumulative_revision_impacts(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        r"""Return each component's cumulative impact across vintages.

        The result has columns ``[component, cumulative_impact]``.
        """
        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        rev = self._revision(variable=variable, source=source)
        rev = rev[(rev["date"] == date) & (rev["revision_source"] == "news")]

        if rev.empty:
            # Use the latest level contributions when no news rows exist.
            lev = self._level(variable=variable, source=source)
            lev = lev[lev["date"] == date]
            last_vintage = lev["vintage_date"].max()
            lev = lev[lev["vintage_date"] == last_vintage]
            impacts = (
                lev.groupby("component")["contribution"]
                .sum()
                .reset_index(name="cumulative_impact")
            )
        else:
            impacts = (
                rev.groupby("component")["contribution"]
                .sum()
                .reset_index(name="cumulative_impact")
            )

        # Exclude the intercept and residual from component impacts.
        impacts = impacts[~impacts["component"].isin(["intercept", "residual"])]
        return impacts.sort_values("cumulative_impact", key=abs, ascending=True)

    def release_table_data(
        self,
        date: pd.Timestamp | str | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Build a table of release impacts for each vintage.

        The result has columns ``[vintage_date, component, news, weight,
        contribution, cumulative_nowcast]``.
        """
        date = self._validate_date(date)
        variable = self._validate_variable(variable)
        source = self._validate_source(source)

        lev = self._level(variable=variable, source=source)
        lev = lev[lev["date"] == date]

        rev = self._revision(variable=variable, source=source)
        rev = rev[(rev["date"] == date) & (rev["revision_source"] == "news")]

        # Get nowcast at each vintage
        nowcast_by_vintage = (
            lev.groupby("vintage_date")["contribution"]
            .sum()
            .reset_index(name="nowcast")
        )

        # Get per-vintage, per-component news impacts
        if not rev.empty:
            impacts = rev[
                ["vintage_date", "component", "news", "weight", "contribution"]
            ].copy()
            impacts = impacts[~impacts["component"].isin(["intercept", "residual"])]
            impacts = impacts.merge(nowcast_by_vintage, on="vintage_date", how="left")
            impacts = impacts.rename(columns={"nowcast": "cumulative_nowcast"})
        else:
            impacts = lev[["vintage_date", "component", "contribution"]].copy()
            impacts = impacts[~impacts["component"].isin(["intercept", "residual"])]
            impacts["news"] = np.nan
            impacts["weight"] = np.nan
            impacts = impacts.merge(nowcast_by_vintage, on="vintage_date", how="left")
            impacts = impacts.rename(columns={"nowcast": "cumulative_nowcast"})

        return impacts.sort_values(["vintage_date", "component"])

    # Consolidated indicator table.

    def indicator_table(
        self,
        realised: pd.DataFrame | pd.Series,
        n_obs: pd.DataFrame | None = None,
        pub_delays: pd.Series | dict[str, float] | None = None,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Build one table containing the available indicator metrics.

        The result always contains columns for signal magnitude, directional
        accuracy, error improvement, intrinsic content, timing premium, and
        information density. Missing inputs produce ``NaN`` values.

        Parameters
        ----------
        realised : pd.DataFrame | pd.Series
            Realised target values.
        n_obs : pd.DataFrame | None
            If provided, compute intrinsic content and timing premium.
        pub_delays : pd.Series | dict[str, float] | None
            If provided with ``n_obs``, compute information density.
        variable : str | None
            Target variable to include. If None, includes all variables.
        source : str | None
            Model source to include. If None, includes all sources.

        Returns
        -------
        pd.DataFrame
            One row per component with the requested indicator metrics.
        """
        table = pd.DataFrame(
            {
                "Signal magnitude": self.signal_magnitude(
                    variable=variable, source=source
                ),
                "Directional accuracy": self.hit_rate(
                    realised, variable=variable, source=source
                ),
                "Error improvement": self.error_improvement(
                    realised, variable=variable, source=source
                ),
            }
        )

        if n_obs is not None:
            td = self.timing_decomposition(n_obs, variable=variable, source=source)
            td_indexed = td.set_index("component")
            table["Intrinsic content"] = td_indexed["alpha"]
            table["Timing premium"] = td_indexed["beta"]

            if pub_delays is not None:
                info = self.information_density(
                    pub_delays, n_obs=n_obs, variable=variable, source=source
                )
                info_indexed = info.set_index("component")
                table["Information density"] = info_indexed["density"]
                if "density_star" in info_indexed.columns:
                    table["Information density*"] = info_indexed["density_star"]

        # Ensure all columns are always present (NaN if inputs not provided)
        for col in [
            "Intrinsic content",
            "Timing premium",
            "Information density",
            "Information density*",
        ]:
            if col not in table.columns:
                table[col] = np.nan

        table.index.name = "component"
        return table

    # Historical indicator metrics.

    def indicator_table_over_time(
        self,
        realised: pd.DataFrame | pd.Series,
        min_periods: int = 4,
        variable: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Compute indicator metrics on an expanding vintage-date window.

        For each vintage date *v*, use only level rows with
        ``vintage_date <= v``. This shows how each indicator's metrics change
        as more out-of-sample quarters accumulate.

        Parameters
        ----------
        realised : pd.DataFrame | pd.Series
            Realised target values.
        min_periods : int
            Minimum number of forecast observations before computing
            metrics (avoids noisy early estimates).
        variable : str | None
            Target variable to include. ``None`` includes all variables.
        source : str | None
            Model source to include. ``None`` includes all sources.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns
            ``[vintage_date, component, Signal magnitude, Directional
            accuracy, Error improvement]``.
        """
        lev = self._level(variable=variable, source=source)
        vintages = sorted(lev["vintage_date"].unique())

        records = []
        for i, v in enumerate(vintages):
            subset = lev[lev["vintage_date"] <= v]
            # Require enough distinct forecast groups for a stable estimate.
            n_groups = subset.groupby(self._GROUP_KEYS).ngroups
            if n_groups < min_periods:
                continue

            # Reuse the forecast aggregation for this historical slice.
            tmp = _TemporalSlice(subset)
            fcst = tmp.forecasts()
            merged = self._align_realised(fcst, realised)
            if len(merged) < min_periods:
                continue

            # Signal magnitude per component
            mc = subset[self._GROUP_KEYS + ["component", "contribution"]]
            sig = mc.groupby("component")["contribution"].apply(
                lambda s: s.abs().mean()
            )

            # Hit rate and error improvement
            comp = mc.merge(
                merged[self._GROUP_KEYS + ["forecast", "value"]],
                on=self._GROUP_KEYS,
            )
            comp["err_full"] = (comp["value"] - comp["forecast"]).abs()
            comp["err_without"] = (
                comp["value"] - (comp["forecast"] - comp["contribution"])
            ).abs()
            comp["hit"] = (comp["err_full"] < comp["err_without"]).astype(int)
            comp["improvement"] = comp["err_without"] - comp["err_full"]

            hr = comp.groupby("component")["hit"].mean() * 100
            ei = comp.groupby("component")["improvement"].mean()

            for component in sig.index:
                records.append(
                    {
                        "vintage_date": v,
                        "component": component,
                        "Signal magnitude": sig.get(component, np.nan),
                        "Directional accuracy": hr.get(component, np.nan),
                        "Error improvement": ei.get(component, np.nan),
                    }
                )

        return pd.DataFrame(records)


class _TemporalSlice:
    """Compute forecasts from a subset of level rows."""

    _GROUP_KEYS = NewsAnalysis._GROUP_KEYS

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def forecasts(self) -> pd.DataFrame:
        return (
            self.df.groupby(self._GROUP_KEYS, as_index=False)["contribution"]
            .sum()
            .rename(columns={"contribution": "forecast"})
        )
