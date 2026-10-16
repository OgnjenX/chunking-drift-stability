"""EXP001 numerical, measurement, and intervention invariants."""

import numpy as np
import pytest

from chunkdrift.analyze import paired
from chunkdrift.data import HELDOUT, LEARNED, NEW, OLD, corrupt, encode
from chunkdrift.metrics import accuracy, change, probe
from chunkdrift.model import Model


@pytest.mark.parametrize("kind", ["flat", "dense", "competitive"])
@pytest.mark.parametrize("lesion", [None, "feedback_off", "chunk_lesion", "lower_lesion"])
def test_exact_gradients(kind, lesion):
    """Verify exact gradients."""
    m = Model(kind, 3, hidden=5, chunks=4)
    if kind != "flat" and lesion:
        m.intervene(lesion, 3, 0.5)
    x, t = corrupt(["AB", "CDAB"], np.random.default_rng(1))
    _, grads = m.loss_grad(x, t)
    for name, w in m.w.items():
        for idx in np.ndindex(w.shape):
            old, eps = w[idx], 1e-5
            w[idx] = old + eps
            plus = m.loss_grad(x, t)[0]
            w[idx] = old - eps
            minus = m.loss_grad(x, t)[0]
            w[idx] = old
            assert grads[name][idx] == pytest.approx((plus - minus) / (2 * eps), abs=1e-7)


def test_data_boundaries_and_order():
    """Verify data boundaries and order."""
    assert set(NEW).isdisjoint(HELDOUT)
    assert len(set(NEW + HELDOUT)) == 16
    x, _ = encode(["AB", "BA"])
    assert not np.array_equal(x[0], x[1])
    x, t = corrupt(["AB"], np.random.default_rng(5), missing=1)
    assert np.count_nonzero(x) == 3
    assert t.tolist() == [[0, 1, 8, 8]]


def test_capacity_noise_and_initialization_matching():
    """Verify capacity noise and initialization matching."""
    a, b, f = Model("competitive", 4), Model("dense", 4), Model("flat", 4)
    assert a.parameter_count == b.parameter_count == 5760
    assert f.parameter_count == 3456
    for k, weight in a.w.items():
        np.testing.assert_array_equal(weight, b.w[k])
    aa, bb, ff = a.copy(), b.copy(), f.copy()
    for m in (a, b, f):
        m.noise(7, 1, 0.8, {k: 1 for k in m.w})
    for k, weight in a.w.items():
        np.testing.assert_allclose(weight - aa.w[k], b.w[k] - bb.w[k])
    for k, weight in f.w.items():
        np.testing.assert_allclose(a.w[k] - aa.w[k], weight - ff.w[k])


def test_noise_is_unconstrained_and_can_destroy_function():
    """Verify noise is unconstrained and can destroy function."""
    m = Model("flat", 4, hidden=36)
    m.w["E"] = np.eye(36)
    m.w["D"] = np.eye(36) * 4
    x, t = encode(LEARNED)
    assert accuracy(m, x, t)["token"] == 1
    m.noise(1, 1, 100, {k: 1 for k in m.w})
    assert accuracy(m, x, t)["token"] < 0.5


def test_no_noise_no_updates_is_identical_and_evaluation_does_not_mutate():
    """Verify no noise no updates is identical and evaluation does not mutate."""
    m = Model("competitive", 2)
    copy = m.copy()
    x, t = encode(OLD)
    accuracy(m, x, t)
    m.noise(2, 1, 0, {k: 1 for k in m.w})
    for key, weight in m.w.items():
        np.testing.assert_array_equal(copy.w[key], weight)
    cosine = change(m.forward(x)["u"], copy.forward(x)["u"])["cosine"]
    assert cosine is not None
    assert cosine < 1e-14


def test_geometry_separates_rotation_from_coordinate_change():
    """Verify geometry separates rotation from coordinate change."""
    a = np.random.default_rng(3).normal(size=(12, 6))
    b = -a
    assert change(a, b)["cosine"] == pytest.approx(2)
    assert change(a, b)["one_minus_cka"] == pytest.approx(0)


def test_feedback_ablation_keeps_bottom_up_amplitude():
    """Verify feedback ablation keeps bottom up amplitude."""
    m = Model("competitive", 1)
    m.intervene("feedback_off", 1)
    x, _ = encode(OLD)
    a = m.forward(x)
    np.testing.assert_allclose(a["h"], 0.5 * a["u"])


def test_determinism_and_branch_independence():
    """Verify determinism and branch independence."""
    a = Model("competitive", 1)
    b = a.copy()
    x, t = corrupt(OLD, np.random.default_rng(7))
    for _ in range(3):
        a.update(x, t, 0.15)
        b.update(x, t, 0.15)
    for k, weight in a.w.items():
        np.testing.assert_array_equal(weight, b.w[k])
    b.w["E"][0, 0] += 1
    assert a.w["E"][0, 0] != b.w["E"][0, 0]


def test_analysis_uses_paired_seed_units():
    """Verify analysis uses paired seed units."""
    result = paired([1, 3, 5, 7], [0, 2, 4, 6], 100)
    assert result["mean"] == 1
    assert result["ci95"] == [1, 1]


def test_masked_target_metric_excludes_visible_answers():
    """Verify masked target metric excludes visible answers."""
    m = Model("flat", 1, hidden=36)
    m.w["E"] = np.eye(36)
    m.w["D"] = np.eye(36) * 4
    result = probe(m, ["EFGH"], 1, 32)
    assert result["token"] == 0.75
    assert result["masked_token"] == 0
