"""
Tests for `append_summary_rows`, the mean/median/std footer every summary CSV carries.

The failure it guards against: appending the `median` row and then taking `mean()` over the
frame folded the median into the mean (and both into `std`), so `dataset_summary.csv` reported
a different mean from the LaTeX table written in the same run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from analysis.compare import format_latex_table

from analysis import append_summary_rows


def _frame() -> pd.DataFrame:
    # Object dtype, as the pre-indexed frames the subcommands build are.
    frame = pd.DataFrame(columns=["a", "b"], index=["t1", "t2", "t3", "skipped"])
    frame.loc["t1"] = [1.0, 10.0]
    frame.loc["t2"] = [2.0, 20.0]
    frame.loc["t3"] = [9.0, 90.0]
    return frame


def test_aggregates_cover_trajectory_rows_only() -> None:
    summary = append_summary_rows(_frame())

    assert summary.loc["mean", "a"] == pytest.approx(4.0)
    assert summary.loc["median", "a"] == pytest.approx(2.0)
    assert summary.loc["std", "a"] == pytest.approx(np.std([1.0, 2.0, 9.0]))
    assert summary.loc["mean", "b"] == pytest.approx(40.0)


def test_csv_footer_agrees_with_latex_footer() -> None:
    frame = _frame()
    results = [
        (name, {"a": float(row["a"]), "b": float(row["b"])})
        for name, row in frame.drop(index="skipped").iterrows()
    ]
    latex = format_latex_table(results, "t", columns=[("a", "A"), ("b", "B")])
    summary = append_summary_rows(frame)

    for row_name in ("mean", "median", "std"):
        expected = (
            f"{row_name} & {summary.loc[row_name, 'a']:.2f} & {summary.loc[row_name, 'b']:.2f}"
        )
        assert expected in latex


def test_empty_frame_is_left_alone() -> None:
    assert append_summary_rows(pd.DataFrame(columns=["a"])).empty
