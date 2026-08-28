"""The :class:`NewsData` interface for the ``decompositions`` data contract."""

import pandas as pd

from news_decomp.analysis import NewsAnalysis
from news_decomp.plots import NewsPlots
from news_decomp.report import NewsReport
from news_decomp.schema import decomposition_schema


class NewsData(NewsAnalysis, NewsPlots, NewsReport):
    """Analyse and visualise a validated ``decompositions`` table.

    Parameters
    ----------
    df : pd.DataFrame
        A long-format ``decompositions`` frame. See ``news_decomp.md`` for
        the data contract.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = self.validate(df)

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Validate ``df`` against the ``decompositions`` schema.

        Return the validated frame. Raise ``pandera.errors.SchemaError`` when
        a row violates the schema.
        """
        return decomposition_schema.validate(df)

    def _validate_date(self, date: pd.Timestamp | str | None) -> pd.Timestamp:
        """Validate a target date and return it as a Timestamp.

        Parameters
        ----------
        date : pd.Timestamp | str | None
            A Timestamp, string, or ``None``. ``None`` selects the latest
            target date in the data.

        Returns
        -------
        pd.Timestamp
            The validated target date.

        Raises
        ------
        ValueError
            If date is provided but not found in the decompositions data.
        """
        if date is None:
            return self.df["date"].max()

        date_ts = pd.Timestamp(date)

        # Reject dates that do not occur in the decomposition table.
        available_dates = self.df["date"].unique()
        if date_ts not in available_dates:
            raise ValueError(
                f"target_date {date_ts} not found in decomposition data. "
                f"Available dates: {sorted(available_dates)}"
            )

        return date_ts

    def _validate_variable(self, variable: str | None) -> str:
        """Validate a variable name and return the selected value.

        Parameters
        ----------
        variable : str | None
            A variable name or ``None``. ``None`` selects the first variable.

        Returns
        -------
        str
            The validated variable name.

        Raises
        ------
        ValueError
            If variable is provided but not found in the decompositions data.
        """
        if variable is None:
            return self.df["variable"].iloc[0]

        available_variables = self.df["variable"].unique()
        if variable not in available_variables:
            raise ValueError(
                f"variable '{variable}' not found in decomposition data. "
                f"Available variables: {sorted(available_variables)}"
            )

        return variable

    def _validate_source(self, source: str | None) -> str:
        """Validate a source name and return the selected value.

        Parameters
        ----------
        source : str | None
            A source name or ``None``. ``None`` selects the first source.

        Returns
        -------
        str
            The validated source name.

        Raises
        ------
        ValueError
            If source is provided but not found in the decompositions data.
        """
        if source is None:
            return self.df["source"].iloc[0]

        available_sources = self.df["source"].unique()
        if source not in available_sources:
            raise ValueError(
                f"source '{source}' not found in decomposition data. "
                f"Available sources: {sorted(available_sources)}"
            )

        return source

    def summary(self) -> None:
        """Print a formatted summary of the decomposition table.

        The summary lists variables, sources, components, metrics, decomposition
        kinds, and revision sources. It also prints frequency, date ranges,
        vintage ranges, horizon ranges, and row counts for each variable-source
        pair.
        """
        df = self.df
        revision = df[df["decomposition"] == "revision"]

        def _names(col, frame=df):
            return sorted(frame[col].dropna().unique().tolist())

        print("\n" + "=" * 40)
        print("NEWS DECOMPOSITION SUMMARY")
        print("=" * 40)
        print(f"  Rows: {len(df)}")

        variables = _names("variable")
        sources = _names("source")
        components = _names("component")
        metrics = _names("forecast_metric")
        print(f"\n  Variables  ({len(variables)}): {', '.join(variables)}")
        print(f"  Sources    ({len(sources)}): {', '.join(sources)}")
        print(f"  Components ({len(components)}): {', '.join(components)}")
        print(f"  Metrics    ({len(metrics)}): {', '.join(metrics)}")

        decomp_counts = df["decomposition"].value_counts().to_dict()
        print(
            "\n  Decomposition kinds: "
            + ", ".join(f"{k}={v}" for k, v in decomp_counts.items())
        )
        if len(revision):
            rev_counts = revision["revision_source"].value_counts().to_dict()
            print(
                "  Revision sources:    "
                + ", ".join(f"{k}={v}" for k, v in rev_counts.items())
            )

        self._print_group_table()
        print("=" * 40 + "\n")

    def _print_group_table(self) -> None:
        """Print dates, vintages, and row counts for each variable-source pair."""
        df = self.df
        revision_keys = [
            "variable",
            "source",
            "forecast_horizon",
            "date",
            "vintage_date",
            "base_vintage_date",
        ]

        separator = "  " + "-" * 30
        print("\n  [BY VARIABLE x SOURCE]")

        for (variable, source), grp in df.groupby(["variable", "source"]):
            rev = grp[grp["decomposition"] == "revision"]
            n_levels = int((grp["decomposition"] == "level").sum())
            n_revisions = rev.drop_duplicates(revision_keys).shape[0]
            print(separator)
            print(f"  {variable}  |  {source}")
            print(f"    Frequency   : {grp['frequency'].iloc[0]}")
            print(
                f"    Target dates: {grp['date'].min():%Y-%m-%d}"
                f" to {grp['date'].max():%Y-%m-%d}"
                f"  ({grp['date'].nunique()} quarters)"
            )
            print(
                f"    Vintages    : {grp['vintage_date'].min():%Y-%m-%d}"
                f" to {grp['vintage_date'].max():%Y-%m-%d}"
                f"  ({grp['vintage_date'].nunique()} vintages)"
            )
            hor_min = grp["forecast_horizon"].min()
            hor_max = grp["forecast_horizon"].max()
            hor = str(hor_min) if hor_min == hor_max else f"{hor_min}-{hor_max}"
            print(f"    Horizon     : {hor}")
            print(f"    Level rows  : {n_levels}")
            print(f"    Revisions   : {n_revisions}")
            if len(rev):
                rev_counts = rev["revision_source"].value_counts()
                for src, cnt in rev_counts.items():
                    print(f"      {src:<16}: {cnt} rows")

        print(separator)
