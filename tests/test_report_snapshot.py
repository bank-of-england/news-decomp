"""Test nowcast report output and labels with Syrupy."""

import io
import sys

import matplotlib
import pandas as pd

# Use a non-interactive backend during tests.
matplotlib.use("Agg")

from news_decomp import NewsData
from news_decomp.sample import simulate


def test_data_flow_table_labels_news_as_data_revision():
    """Use a data-revision label instead of an outturn-level label."""
    data = simulate()
    flow = NewsData(data["decompositions"]).data_flow_table()

    assert "Data Revision" in flow.columns
    assert "Outturn" not in flow.columns

    row = flow.iloc[0]
    decomposition = data["decompositions"]
    expected = decomposition[
        (decomposition["vintage_date"] == pd.Timestamp(row["Model Update"]))
        & (decomposition["date"].dt.to_period("Q").astype(str) == row["Target Quarter"])
        & (decomposition["component"] == row["Series"])
        & (decomposition["revision_source"] == "news")
    ]["news"].iloc[0]
    assert row["Data Revision"] == expected


def test_report_snapshot(snapshot):
    """Confirm that report output matches the stored snapshot."""
    # Generate the report.
    data = simulate()
    news_data = NewsData(data["decompositions"])

    # Capture console output for comparison.
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        news_data.report(show=False)
    finally:
        sys.stdout = old_stdout

    # Read the captured output.
    actual = captured_output.getvalue()

    # Syrupy handles snapshot file access and comparison.
    assert actual == snapshot


def test_report_structure():
    """Confirm that the report contains its required sections."""
    data = simulate()
    news_data = NewsData(data["decompositions"])

    # Capture console output
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        news_data.report(show=False)
    finally:
        sys.stdout = old_stdout

    output = captured_output.getvalue()

    # Check for required sections
    required_sections = [
        "NOWCAST SUMMARY",
        "Current Nowcast",
        "Initial Nowcast",
        "Total Revision",
        "Vintages",
        "DATA FLOW TABLE",
        "Model Update",
        "Data Release",
        "Target Quarter",
    ]

    for section in required_sections:
        assert section in output, f"Missing required section: {section}"


def test_report_summary_stats():
    """Confirm that summary statistics and data-flow rows are present."""
    data = simulate()
    news_data = NewsData(data["decompositions"])

    # Capture console output for inspection.
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        news_data.report(show=False)
    finally:
        sys.stdout = old_stdout

    output = captured_output.getvalue()

    # Check that the report printed a current nowcast.
    lines = output.split("\n")
    nowcast_line = [line for line in lines if "Current Nowcast" in line]
    assert len(nowcast_line) > 0, "Current Nowcast not found"

    # Check that the data-flow table contains dated rows.
    flow_table_lines = [line for line in lines if "2025" in line or "2024" in line]
    assert len(flow_table_lines) > 0, "Data flow table appears empty"
