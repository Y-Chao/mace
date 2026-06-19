"""Tests for the forces cosine-similarity loss, its CLI weight args, and the
force cosine-similarity tracking metric."""

import numpy as np
import pytest
import torch

from mace.data import AtomicData, Configuration
from mace.modules import (
    WeightedEnergyForcesCosineLoss,
    WeightedEnergyForcesLoss,
    WeightedHuberEnergyForcesCosineLoss,
)
from mace.modules.loss import mean_cosine_distance_forces
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.scripts_utils import get_loss_fn
from mace.tools.utils import compute_cosine_similarity_forces


@pytest.fixture(name="config")
def _config():
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, -2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        properties={
            "forces": np.array(
                [
                    [0.0, -1.3, 0.0],
                    [1.0, 0.2, 0.0],
                    [0.0, 1.1, 0.3],
                ]
            ),
            "energy": -1.5,
            "stress": np.array([1.0, 0.0, 0.5, 0.0, -1.0, 0.0]),
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
            "stress": 1.0,
        },
    )


@pytest.fixture(name="table")
def _table():
    return AtomicNumberTable([1, 8])


@pytest.fixture(autouse=True)
def _set_torch_default_dtype():
    torch.set_default_dtype(torch.float64)


@pytest.fixture(name="batch")
def _batch(config, table):
    data = AtomicData.from_config(config, z_table=table, cutoff=3.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data, data],
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )
    return next(iter(data_loader))


# ---------------------------------------------------------------------------
# Cosine helper / loss correctness
# ---------------------------------------------------------------------------


def test_cosine_distance_zero_when_aligned(batch):
    pred = {"energy": batch.energy, "forces": batch.forces}
    assert mean_cosine_distance_forces(batch, pred).item() == pytest.approx(0.0)


def test_cosine_distance_positive_when_misaligned(batch):
    # Rotate the predicted forces away from the reference -> 1 - cos > 0.
    pred = {"energy": batch.energy, "forces": batch.forces + 0.5}
    assert mean_cosine_distance_forces(batch, pred).item() > 0.0


def test_cosine_distance_maximal_when_antiparallel(batch):
    # Anti-parallel forces -> cos = -1 -> per-atom distance = 2.
    pred = {"energy": batch.energy, "forces": -batch.forces}
    # config-weighted mean of (1 - cos) with unit weights == 2.0
    assert mean_cosine_distance_forces(batch, pred).item() == pytest.approx(2.0)


def test_cosine_loss_reduces_to_baseline_when_zero_weight(batch):
    baseline = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=10.0)
    cosine = WeightedEnergyForcesCosineLoss(
        energy_weight=1.0, forces_weight=10.0, cosine_weight=0.0
    )
    # Use a non-trivial prediction so both energy and forces terms are nonzero.
    pred = {
        "energy": batch.energy + 0.3,
        "forces": batch.forces + 0.2,
        "stress": batch.stress,
    }
    assert cosine(batch, pred).item() == pytest.approx(baseline(batch, pred).item())


def test_huber_cosine_loss_runs_and_zero_weight_drops_cosine(batch):
    loss_with = WeightedHuberEnergyForcesCosineLoss(
        energy_weight=1.0, forces_weight=1.0, cosine_weight=5.0, huber_delta=0.01
    )
    loss_without = WeightedHuberEnergyForcesCosineLoss(
        energy_weight=1.0, forces_weight=1.0, cosine_weight=0.0, huber_delta=0.01
    )
    pred = {"energy": batch.energy + 0.3, "forces": batch.forces + 0.2}
    # With a misaligned prediction the cosine term is strictly positive, so the
    # weighted loss must exceed the one that drops the cosine term.
    assert loss_with(batch, pred).item() > loss_without(batch, pred).item()


# ---------------------------------------------------------------------------
# Optional stress term
# ---------------------------------------------------------------------------


def test_stress_term_off_by_default(batch):
    # stress_weight defaults to 0.0 -> stress prediction is ignored entirely.
    no_stress = WeightedEnergyForcesCosineLoss(
        energy_weight=1.0, forces_weight=1.0, cosine_weight=1.0
    )
    pred = {
        "energy": batch.energy,
        "forces": batch.forces,
        "stress": batch.stress + 5.0,  # large stress error, must be ignored
    }
    # Only the cosine term contributes here (E and F are exact); stress weight 0.
    assert no_stress(batch, pred).item() == pytest.approx(0.0)


def test_stress_term_contributes_when_weighted(batch):
    loss_fn = WeightedEnergyForcesCosineLoss(
        energy_weight=1.0, forces_weight=1.0, cosine_weight=0.0, stress_weight=10.0
    )
    pred_exact = {
        "energy": batch.energy,
        "forces": batch.forces,
        "stress": batch.stress,
    }
    pred_wrong = {
        "energy": batch.energy,
        "forces": batch.forces,
        "stress": batch.stress + 1.0,
    }
    assert loss_fn(batch, pred_exact).item() == pytest.approx(0.0)
    assert loss_fn(batch, pred_wrong).item() > 0.0


def test_huber_cosine_stress_term(batch):
    loss_fn = WeightedHuberEnergyForcesCosineLoss(
        energy_weight=1.0,
        forces_weight=1.0,
        cosine_weight=0.0,
        stress_weight=10.0,
        huber_delta=0.01,
    )
    pred_exact = {
        "energy": batch.energy,
        "forces": batch.forces,
        "stress": batch.stress,
    }
    pred_wrong = {
        "energy": batch.energy,
        "forces": batch.forces,
        "stress": batch.stress + 1.0,
    }
    assert loss_fn(batch, pred_exact).item() == pytest.approx(0.0)
    assert loss_fn(batch, pred_wrong).item() > 0.0


def test_get_loss_fn_passes_stress_weight():
    parser = build_default_arg_parser()
    args = parser.parse_args(
        ["--name", "test", "--loss", "cosine", "--stress_weight", "8.0"]
    )
    loss_fn = get_loss_fn(args, dipole_only=False, compute_dipole=False)
    assert isinstance(loss_fn, WeightedEnergyForcesCosineLoss)
    assert loss_fn.stress_weight.item() == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Weight value -> loss change
# ---------------------------------------------------------------------------


def test_loss_scales_with_cosine_weight(batch):
    pred = {"energy": batch.energy, "forces": batch.forces + 0.4}

    # Pure energy/forces part (cosine weight 0) is the constant baseline.
    base = WeightedEnergyForcesCosineLoss(
        energy_weight=1.0, forces_weight=1.0, cosine_weight=0.0
    )(batch, pred).item()

    cos_term = mean_cosine_distance_forces(batch, pred).item()
    assert cos_term > 0.0

    for w in (1.0, 2.0, 5.0):
        loss_w = WeightedEnergyForcesCosineLoss(
            energy_weight=1.0, forces_weight=1.0, cosine_weight=w
        )(batch, pred).item()
        # loss(w) == base + w * cos_term  (the cosine term enters linearly)
        assert loss_w == pytest.approx(base + w * cos_term)

    # Monotonically increasing in the cosine weight for a misaligned prediction.
    losses = [
        WeightedEnergyForcesCosineLoss(
            energy_weight=1.0, forces_weight=1.0, cosine_weight=w
        )(batch, pred).item()
        for w in (0.0, 1.0, 2.0, 5.0)
    ]
    assert losses == sorted(losses)
    assert losses[0] < losses[-1]


# ---------------------------------------------------------------------------
# CLI args for the cosine weights
# ---------------------------------------------------------------------------


def test_cosine_weight_args_defaults():
    parser = build_default_arg_parser()
    args = parser.parse_args(["--name", "test"])
    assert args.cosine_weight == 1.0
    assert args.swa_cosine_weight == 1.0


def test_cosine_weight_args_parse_custom_values():
    parser = build_default_arg_parser()
    args = parser.parse_args(
        [
            "--name",
            "test",
            "--loss",
            "cosine",
            "--cosine_weight",
            "3.5",
            "--swa_cosine_weight",
            "7.0",
        ]
    )
    assert args.loss == "cosine"
    assert args.cosine_weight == pytest.approx(3.5)
    assert args.swa_cosine_weight == pytest.approx(7.0)


def test_swa_cosine_weight_alias():
    parser = build_default_arg_parser()
    args = parser.parse_args(
        ["--name", "test", "--stage_two_cosine_weight", "9.0"]
    )
    assert args.swa_cosine_weight == pytest.approx(9.0)


@pytest.mark.parametrize("loss_name", ["cosine", "huber_cosine"])
def test_loss_choices_accept_cosine(loss_name):
    parser = build_default_arg_parser()
    args = parser.parse_args(["--name", "test", "--loss", loss_name])
    assert args.loss == loss_name


# ---------------------------------------------------------------------------
# Factory wires the CLI weight into the constructed loss
# ---------------------------------------------------------------------------


def test_get_loss_fn_passes_cosine_weight():
    parser = build_default_arg_parser()
    args = parser.parse_args(
        ["--name", "test", "--loss", "cosine", "--cosine_weight", "4.0"]
    )
    loss_fn = get_loss_fn(args, dipole_only=False, compute_dipole=False)
    assert isinstance(loss_fn, WeightedEnergyForcesCosineLoss)
    assert loss_fn.cosine_weight.item() == pytest.approx(4.0)


def test_get_loss_fn_huber_cosine():
    parser = build_default_arg_parser()
    args = parser.parse_args(
        ["--name", "test", "--loss", "huber_cosine", "--cosine_weight", "2.0"]
    )
    loss_fn = get_loss_fn(args, dipole_only=False, compute_dipole=False)
    assert isinstance(loss_fn, WeightedHuberEnergyForcesCosineLoss)
    assert loss_fn.cosine_weight.item() == pytest.approx(2.0)


def test_loss_repr_includes_cosine_weight():
    loss_fn = WeightedEnergyForcesCosineLoss(cosine_weight=2.5)
    assert "cosine_weight=2.500" in repr(loss_fn)


# ---------------------------------------------------------------------------
# Force cosine-similarity tracking metric
# ---------------------------------------------------------------------------


def test_metric_identical_forces_is_one():
    f = np.array([[0.0, -1.3, 0.0], [1.0, 0.2, 0.0], [0.0, 1.1, 0.3]])
    assert compute_cosine_similarity_forces(f, f) == pytest.approx(1.0)


def test_metric_antiparallel_forces_is_minus_one():
    f = np.array([[0.0, -1.3, 0.0], [1.0, 0.2, 0.0], [0.0, 1.1, 0.3]])
    assert compute_cosine_similarity_forces(-f, f) == pytest.approx(-1.0)
