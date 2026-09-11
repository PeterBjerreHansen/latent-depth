import math
import json
from pathlib import Path

import pytest
import torch

from config import ExperimentConfig, SweepConfig
from rhm.theory import loss_upper_bounds, sample_complexities
from sweep import fixed_exposure_budget
from training import input_block_size, objective_inputs


def test_next_token_indexing():
    cfg = ExperimentConfig.from_dict(
        {
            "rhm": {"v": 16, "n": 16, "m": 4, "s": 2, "L": 3},
            "objective": {"mode": "next_token"},
        }
    )
    tokens = torch.arange(8).unsqueeze(0)
    x, y = objective_inputs(tokens)
    assert x.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert y.tolist() == [[1, 2, 3, 4, 5, 6, 7]]
    assert input_block_size(cfg) == 8


def test_last_token_mode_is_rejected():
    with pytest.raises(ValueError, match="next_token"):
        ExperimentConfig.from_dict(
            {
                "rhm": {"v": 32, "n": 32, "m": 8, "s": 2, "L": 3},
                "objective": {"mode": "last_token"},
            }
        )


def test_objective_inputs_rejects_short_sequences():
    with pytest.raises(ValueError, match="at least two"):
        objective_inputs(torch.zeros((1, 1), dtype=torch.long))


def test_characteristic_sample_complexities():
    values = sample_complexities(3, v=32, m=8, s=2)
    expected = [
        32 * 8 / 0.75,
        32 * 8**3 / 0.75,
        32 * 8**5 / 0.75,
    ]
    for got, want in zip(values, expected):
        assert math.isclose(got, want, rel_tol=1e-12)


def test_loss_bounds_are_monotone_improving():
    bounds = loss_upper_bounds(3, v=32, m=8, s=2)
    assert math.isclose(bounds[0], math.log(32))
    assert all(a > b for a, b in zip(bounds, bounds[1:]))


def test_sweep_accepts_fixed_exposure_budget():
    sweep = SweepConfig.from_json("configs/next_token_sweep.json")
    assert sweep.samples_per_example is None
    sweep = SweepConfig.from_json("configs/next_token_sweep_fixed_exposure.json")
    assert sweep.samples_per_example == 8.0


def test_fixed_exposure_budget_uses_complete_dataset_passes():
    assert fixed_exposure_budget(
        train_size=32, batch_size=64, samples_per_example=8
    ) == (8, 1)
    assert fixed_exposure_budget(
        train_size=100, batch_size=64, samples_per_example=8
    ) == (16, 2)
    with pytest.raises(ValueError, match="whole number"):
        fixed_exposure_budget(train_size=32, batch_size=64, samples_per_example=1.5)


def test_sweep_rejects_fractional_exposure_target(tmp_path: Path):
    config = json.loads(Path("configs/next_token_sweep.json").read_text(encoding="utf-8"))
    config["samples_per_example"] = 1.5
    path = tmp_path / "fractional.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="whole number"):
        SweepConfig.from_json(path)
