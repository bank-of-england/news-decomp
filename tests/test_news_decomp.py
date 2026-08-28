"""Test that ``NewsData`` accepts the simulated sample data."""

import news_decomp as nd
from news_decomp.sample import simulate


def test_newsdecomp_from_sample():
    """Create ``NewsData`` from the schema-valid sample data."""
    data = simulate()

    news_data = nd.NewsData(data["decompositions"])

    assert news_data.df is not None
    assert len(news_data.df) == len(data["decompositions"])


def test_summary_prints_dimensions(capsys):
    """Confirm that ``summary`` prints the table dimensions and revisions."""
    news_data = nd.NewsData(simulate()["decompositions"])

    news_data.summary()

    out = capsys.readouterr().out
    assert "NEWS DECOMPOSITION SUMMARY" in out
    assert "Variables  (1): y" in out
    assert "ols_nowcast" in out
    for component in ("intercept", "X1", "X2"):
        assert component in out
    # Re-estimation at each vintage produces all three revision sources.
    for source in ("news", "reestimation", "interaction"):
        assert source in out
