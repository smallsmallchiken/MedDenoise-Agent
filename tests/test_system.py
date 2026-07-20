"""系统单元测试: pytest tests/"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from meddenoise import data as med_data
from meddenoise.agent.coordinator import Coordinator
from meddenoise.agent.planner import rule_based_plan
from meddenoise.tools import analysis, denoise, enhance, metrics


@pytest.fixture(scope="module")
def phantom():
    return med_data.load_sample_images(size=128)["ct_phantom"]


@pytest.fixture(scope="module")
def noisy(phantom):
    return med_data.add_gaussian_noise(phantom, sigma=25)


def test_noise_estimation(phantom, noisy):
    sigma = analysis.estimate_noise_sigma(noisy)
    assert 15 < sigma < 35


def test_analysis_keys(noisy):
    result = analysis.analyze_image(noisy, "ct_test.png")
    assert result["modality"] == "CT"
    assert result["noise_level"] in ("low", "medium", "high")


@pytest.mark.parametrize("method", list(denoise.DENOISERS))
def test_denoisers_improve_psnr(phantom, noisy, method):
    output = denoise.denoise(noisy, method, sigma=25)
    assert output.shape == noisy.shape
    assert metrics.psnr(phantom, output) > metrics.psnr(phantom, noisy)


def test_vt_bm3d_beats_gaussian(phantom, noisy):
    sa = denoise.denoise(noisy, "vt-bm3d", 25)
    gauss = denoise.denoise(noisy, "gaussian", 25)
    assert metrics.psnr(phantom, sa) > metrics.psnr(phantom, gauss)


def test_enhancers_output_range(noisy):
    for method in enhance.ENHANCERS:
        out = enhance.enhance(noisy, method)
        assert out.min() >= 0 and out.max() <= 1


def test_rule_planner():
    from meddenoise.tools import dncnn

    plan = rule_based_plan({"estimated_sigma": 30, "edge_density": 0.2,
                            "modality": "CT", "dynamic_range": 1.0})
    expected = "dncnn" if dncnn.is_available() else "vt-bm3d"
    assert plan[0]["args"]["method"] == expected
    assert plan[-1]["tool"] == "evaluate_quality"

    plan_strong = rule_based_plan({"estimated_sigma": 40, "edge_density": 0.2,
                                   "modality": "CT", "dynamic_range": 1.0})
    assert plan_strong[0]["args"]["method"] == "vt-bm3d"


def test_coordinator_end_to_end(phantom, noisy):
    coordinator = Coordinator(verbose=False)
    result = coordinator.process(noisy, phantom, "ct_phantom.png")
    assert result["evaluation"]["psnr"] > metrics.psnr(phantom, noisy)
    assert len(result["trace"]) >= 2
