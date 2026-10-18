"""
Checks that client selection follows paper Algorithm 2 and that each client's
privacy is accounted for exactly as the training mechanism (Algorithm 1) spends it.

Run from src/:  python -m pytest tests
Uses small synthetic client data, so CREMA-D is not needed.
"""

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from client import Client  # noqa: E402
from model import build_model  # noqa: E402
from utils.css import css_select, random_selection  # noqa: E402
from utils.moments_accountant import MomentsAccountant  # noqa: E402


# --- Algorithm 2: client selection strategy ------------------------------------

SIZES = [44, 40, 44, 43, 12, 90, 44, 61, 8, 44, 30, 75, 44, 52, 20, 44]


@pytest.mark.parametrize("k", [2, 5, 8, 16, 50])
def test_css_selects_k_distinct_clients(k):
    sel = css_select(SIZES, k, random.Random(0))
    assert len(sel) == min(k, len(SIZES))
    assert len(set(sel)) == len(sel), "the two halves must not overlap"


@pytest.mark.parametrize("k", [4, 7, 8, 12])
def test_css_first_half_is_the_largest_clients(k):
    rng = random.Random(1)
    for _ in range(200):
        sel = css_select(SIZES, k, rng)
        m = k // 2
        top, rest = sel[:m], sel[m:]
        others = [i for i in range(len(SIZES)) if i not in top]
        # every size-selected client is at least as large as every other client
        assert min(SIZES[i] for i in top) >= max(SIZES[i] for i in others)
        assert not set(top) & set(rest)


def test_css_breaks_ties_randomly_not_by_client_id():
    # six clients tie at size 44 but only 3 top slots remain after the larger ones
    rng = random.Random(2)
    k = 2 * 6   # M = 6: clients 5 (90), 11 (75), 7 (61) + 3 of the six 44s
    tied = {i for i, s in enumerate(SIZES) if s == 44}
    seen = set()
    for _ in range(300):
        top = css_select(SIZES, k, rng)[:6]
        assert {5, 11, 7} <= set(top)
        seen |= set(top) & tied
    assert seen == tied, "every tied client must sometimes win a top slot"


def test_css_random_half_is_uniform_over_the_rest():
    rng = random.Random(3)
    k, n = 8, len(SIZES)
    counts = np.zeros(n)
    trials = 20000
    for _ in range(trials):
        for i in css_select(SIZES, k, rng)[k // 2:]:
            counts[i] += 1
    top4 = {5, 11, 7, 13}   # the four strictly largest clients, never in the random half
    pool = [i for i in range(n) if i not in top4]
    assert counts[list(top4)].sum() == 0
    expected = trials * (k - k // 2) / len(pool)
    assert np.allclose(counts[pool], expected, rtol=0.08)


def test_random_selection_is_uniform():
    rng = random.Random(4)
    counts = np.zeros(len(SIZES))
    for _ in range(20000):
        for i in random_selection(SIZES, 4, rng):
            counts[i] += 1
    assert np.allclose(counts, 20000 * 4 / len(SIZES), rtol=0.08)


# --- Algorithm 1 mechanism and per-client accounting ---------------------------

@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    for cid, n in enumerate([8, 44, 120]):
        for split, m in [("train", n), ("val", max(1, n // 4))]:
            np.save(tmp_path / f"client_{cid}_x_{split}.npy",
                    rng.normal(size=(m, config.INPUT_DIM)).astype(np.float32))
            np.save(tmp_path / f"client_{cid}_y_{split}.npy",
                    rng.integers(0, config.NUM_CLASSES, m).astype(np.int64))
    return str(tmp_path)


def make_client(cid, data_dir, **kw):
    return Client(cid, build_model(), sigma=kw.pop("sigma", 3.0), delta=1e-5,
                  data_dir=data_dir, **kw)


def weights(c):
    return [p.detach().clone() for p in c.model.parameters()]


def test_one_noisy_step_per_round_by_default(data_dir):
    c = make_client(1, data_dir)
    assert c.steps_per_round == 1
    assert c.q == pytest.approx(config.BATCH_SIZE / 44)
    assert make_client(1, data_dir, local_epoch=True).steps_per_round == round(44 / config.BATCH_SIZE)
    assert make_client(2, data_dir, lot_frac=0.25).q == 0.25


def test_small_client_gets_no_amplification(data_dir):
    assert make_client(0, data_dir).q == 1.0   # n=8 < B: every record in every step


def test_epsilon_charged_only_for_rounds_participated(data_dir):
    torch.manual_seed(0)
    c = make_client(2, data_dir, sigma=2.0)
    for rnd in [1, 4, 9]:
        c.noisy_gradient(weights(c), clip=2.0, rnd=rnd)
    ref = MomentsAccountant(2.0, 1e-5)
    ref.step(q=c.q, num_steps=3 * c.steps_per_round)
    assert c.rounds_participated == 3
    assert c.epsilon() == pytest.approx(ref.get_epsilon(), rel=1e-12)
    assert [r for r, _ in c.eps_history] == [1, 4, 9]
    assert all(a < b for (_, a), (_, b) in zip(c.eps_history, c.eps_history[1:]))


def test_poisson_sampling_rate_matches_q(data_dir):
    torch.manual_seed(1)
    c = make_client(2, data_dir, lot_frac=0.25)
    draws = torch.stack([(torch.rand(c.n_train) < c.q).float() for _ in range(20000)])
    assert draws.mean().item() == pytest.approx(0.25, abs=0.005)


def test_per_sample_clipping_bounds_each_gradient(data_dir):
    torch.manual_seed(2)
    c = make_client(2, data_dir)
    ps = c._per_sample_grads(c.x_train, c.y_train)
    flat = torch.cat([g.reshape(c.n_train, -1) for g in ps], 1)
    clip = 0.5
    scale = (clip / flat.norm(dim=1)).clamp(max=1.0)
    assert (flat * scale[:, None]).norm(dim=1).max().item() <= clip + 1e-5


def test_noise_std_on_released_sum_is_sigma_times_c(data_dir):
    torch.manual_seed(3)
    sigma, clip = 3.0, 2.0
    c = make_client(2, data_dir, sigma=sigma)
    c.q = 0.0   # empty lot: the release is pure noise / expected lot size
    noise = torch.cat([g.flatten() for g in c._noisy_lot_grad(clip)]) * c.lot_size
    assert noise.std().item() == pytest.approx(sigma * clip, rel=0.01)


# --- the accountant itself -------------------------------------------------------

def test_accountant_matches_abadi_et_al_reference():
    a = MomentsAccountant(4.0, 1e-5)
    a.step(q=0.01, num_steps=10000)
    assert a.get_epsilon() == pytest.approx(1.26, abs=0.01)


@pytest.mark.parametrize("sigma,q,steps", [(1.0, 20 / 44, 50), (3.0, 20 / 44, 150),
                                           (3.0, 0.25, 600), (1.0, 1.0, 150)])
def test_accountant_is_a_valid_upper_bound_close_to_opacus(sigma, q, steps):
    opacus = pytest.importorskip("opacus.accountants")
    ours = MomentsAccountant(sigma, 1e-5)
    ours.step(q=q, num_steps=steps)
    rdp = opacus.RDPAccountant()
    rdp.history = [(sigma, q, steps)]
    theirs = rdp.get_epsilon(1e-5)
    assert ours.get_epsilon() >= theirs * 0.999, "never report less privacy loss than Opacus"
    assert ours.get_epsilon() <= theirs * 1.15


# --- client-level privacy (protects a client's whole dataset) --------------------

def test_client_level_has_no_subsampling_amplification(data_dir):
    c = make_client(2, data_dir, privacy_unit="client")
    assert c.acct_q == 1.0
    assert make_client(2, data_dir).acct_q == pytest.approx(config.BATCH_SIZE / 120)


def test_client_level_clips_whole_batch_gradient_and_adds_sigma_c_noise(data_dir):
    torch.manual_seed(4)
    sigma, clip = 3.0, 0.5
    c = make_client(2, data_dir, sigma=sigma, privacy_unit="client")
    c.sigma = 0.0   # noise-free: the release is the clipped batch gradient
    g = c._noisy_client_grad(clip)
    assert torch.sqrt(sum((gi ** 2).sum() for gi in g)).item() <= clip + 1e-5
    c.sigma = sigma
    torch.manual_seed(5)
    noisy = torch.cat([gi.flatten() for gi in c._noisy_client_grad(clip)])
    assert noisy.std().item() == pytest.approx(sigma * clip, rel=0.02)


def test_client_level_larger_clients_take_more_steps_and_spend_more(data_dir):
    torch.manual_seed(6)
    small = make_client(0, data_dir, sigma=2.0, privacy_unit="client", local_epoch=True)   # n=8
    large = make_client(2, data_dir, sigma=2.0, privacy_unit="client", local_epoch=True)   # n=120
    assert large.steps_per_round > small.steps_per_round
    for c in (small, large):
        for rnd in range(1, 4):
            c.noisy_gradient(weights(c), clip=2.0, rnd=rnd)
    assert large.epsilon() > small.epsilon()
    ref = MomentsAccountant(2.0, 1e-5)
    ref.step(q=1.0, num_steps=3 * large.steps_per_round)
    assert large.epsilon() == pytest.approx(ref.get_epsilon(), rel=1e-12)


# --- speaker-independent folds -------------------------------------------------

def _synthetic_corpus(n_speakers=20, per_speaker=30, seed=0):
    rng = np.random.default_rng(seed)
    speakers = np.repeat([f"s{i:02d}" for i in range(n_speakers)], per_speaker)
    feats = rng.normal(5.0, 3.0, size=(len(speakers), 6)).astype(np.float32)
    labels = rng.integers(0, 4, len(speakers)).astype(np.int64)
    return feats, labels, speakers


def test_folds_hold_out_each_speaker_exactly_once_and_never_train_on_it():
    from prepare_data import speaker_folds
    feats, labels, speakers = _synthetic_corpus()
    tested = []
    for f, parts, names, test_idx in speaker_folds(speakers, 5, None, np.random.default_rng(0)):
        test_spk = set(speakers[test_idx])
        train_spk = set(speakers[np.concatenate(parts)])
        assert not test_spk & train_spk, "a test speaker must never be a training client"
        assert set(names) == train_spk
        assert all(len(set(speakers[p])) == 1 for p in parts), "one speaker per client"
        tested += sorted(test_spk)
    assert sorted(tested) == sorted(set(speakers)), "every speaker is tested exactly once"


def test_skewed_folds_give_clearly_different_client_sizes_and_full_test_speakers():
    from prepare_data import speaker_folds
    feats, labels, speakers = _synthetic_corpus(n_speakers=40, per_speaker=40)
    sizes_by_speaker = {}
    for f, parts, names, test_idx in speaker_folds(speakers, 5, 0.15, np.random.default_rng(1)):
        sizes = [len(p) for p in parts]
        assert max(sizes) >= 4 * min(sizes)
        assert len(test_idx) % 40 == 0, "test speakers keep all their recordings"
        for name, n in zip(names, sizes):
            sizes_by_speaker.setdefault(name, set()).add(n)
    assert all(len(v) == 1 for v in sizes_by_speaker.values()), "a speaker's share is fixed across folds"


def test_normalization_uses_training_clients_only(tmp_path):
    from prepare_data import save_partitions, speaker_folds
    feats, labels, speakers = _synthetic_corpus()
    f, parts, names, test_idx = next(speaker_folds(speakers, 5, None, np.random.default_rng(2)))
    feats = feats.copy()
    feats[test_idx] += 100.0   # if test data leaked into the statistics, train would shift
    save_partitions(feats, labels, parts, str(tmp_path), test_idx=test_idx, client_names=names)
    train = np.concatenate([np.load(tmp_path / f"client_{c}_x_train.npy") for c in range(len(parts))])
    assert np.allclose(train.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(train.std(axis=0), 1.0, atol=1e-3)
    assert np.load(tmp_path / "test_x.npy").mean() > 10.0
