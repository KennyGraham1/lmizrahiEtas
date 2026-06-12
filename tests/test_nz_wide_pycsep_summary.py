import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from run_nz_wide_pycsep_analysis import summarize_result


def result_with_quantile(quantile):
    return SimpleNamespace(
        quantile=quantile,
        status="normal",
        test_distribution=np.array([0.0, 1.0]),
        observed_statistic=0.5,
    )


def test_two_sided_result_rejects_empty_lower_tail():
    summary = summarize_result(result_with_quantile((0.0, 1.0)), False)

    assert summary["consistent"] is False


def test_two_sided_result_requires_both_tails():
    assert summarize_result(result_with_quantile((0.025, 0.975)), False)[
        "consistent"
    ]
    assert not summarize_result(result_with_quantile((0.0245, 0.9755)), False)[
        "consistent"
    ]


def test_one_sided_lower_result_uses_upper_quantile():
    assert summarize_result(result_with_quantile((0.99, 0.03)), True)["consistent"]
    assert not summarize_result(result_with_quantile((0.99, 0.02)), True)[
        "consistent"
    ]
