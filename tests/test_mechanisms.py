import numpy as np

from srclearner.controls import learned_oracle
from srclearner.domain import World
from srclearner.mechanisms import Learner, fuzzy_art_choice_match_learn, instar, working_memory


def test_instar_analytic_and_gate():
    w = np.array([0.5, 0.5])
    x = np.array([3.0, 1.0])
    expected = x / 4 + (w - x / 4) * np.exp(-0.2 * 4)
    np.testing.assert_allclose(instar(w, x), expected)
    np.testing.assert_array_equal(instar(w, x, gate=0), w)
    z = w.copy()
    for _ in range(20):
        z = instar(z, x)
    np.testing.assert_allclose(z, x / 4, atol=1e-6)
    assert np.isclose(z.sum(), 1)


def test_item_order_and_ratio_maintenance():
    a = working_memory([2, 4])
    b = working_memory([2, 4, 7])
    assert a[2] > a[4] and b[2] > b[4] > b[7]
    assert np.isclose(a[2] / a[4], b[2] / b[4])
    assert not np.allclose(a, working_memory([4, 2]))


def test_category_learning_and_old_retention():
    m = Learner(vigilance=0.985, feedback=0)
    for _ in range(30):
        m.train([0, 2], 0)
        m.train([1, 3], 1)
    assert m.predict([0, 2]) == 0 and m.predict([1, 3]) == 1
    old = m.w.copy()
    for _ in range(30):
        m.train([0, 2, 8], 0)
    assert m.predict([0, 2]) == 0 and m.predict([1, 3]) == 1
    assert np.linalg.norm(m.w - old) > 0 and m.movement > 0


def test_overlap_competition_and_lesion():
    m = Learner(feedback=0)
    for _ in range(20):
        m.train([0, 2, 3], 1)
        m.train([0, 2, 4], 2)
    _, r = m.response([0, 2, 3])
    assert np.count_nonzero(r) == 1
    p, r = m.response([0, 2, 3], lesion=True)
    np.testing.assert_allclose(p, np.ones(4) / 4)
    assert not r.any()


def test_feedback_changes_matching():
    m = Learner(feedback=0.5)
    for _ in range(10):
        m.train([0, 1], 0)
    x = working_memory([0, 2])
    _, s = m._matches(2, x)
    m.mode = "no_feedback"
    _, s0 = m._matches(2, x)
    assert not np.allclose(s, s0)


def test_frozen_filters_really_fixed():
    m = Learner(mode="frozen", capacity=12)
    old = m.w.copy()
    for _ in range(3):
        m.train([0, 1, 2], 1)
    np.testing.assert_array_equal(m.w, old)
    assert m.counts.sum() > 0 and m.movement == 0


def test_fuzzy_art_reference():
    x = np.array([0.8, 0.2, 0.2, 0.8])
    w = np.ones(4)
    choice, match, updated = fuzzy_art_choice_match_learn(x, w)
    assert np.isclose(choice, 2 / 4.01) and match == 1
    np.testing.assert_array_equal(updated, x)


def test_world_split_balance_oracle_and_no_label_api():
    w = World(211)
    train = w.sample(320)
    test = w.sample(160, heldout=True)
    assert learned_oracle(train, test) == 1
    assert set(t.context for t in train).isdisjoint(t.context for t in test)
    assert set(x for t in test for x in t.sequence) <= set(x for t in train for x in t.sequence)
    assert set(t.sequence for t in train).isdisjoint(t.sequence for t in test)
    assert np.bincount([t.outcome for t in test]).tolist() == [40] * 4
    assert len({t.tag for t in train}) == 8


def test_predictive_matching_reset():
    m = Learner(feedback=0)
    for _ in range(10):
        m.train([0, 1], 0)
    m.train([0, 1], 1)
    assert m.resets > 0


def test_capacity_overflow_is_counted():
    m = Learner(capacity=1)
    m.train([0, 1, 2], 0)
    assert m.overflows > 0
