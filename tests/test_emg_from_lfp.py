"""Tests for emg_from_lfp.

The exact per-window estimator is checked against a transparent scalar reference
(``scipy.stats.pearsonr`` per window per pair); the global estimator against a
direct numpy reference. Plus return-type, defaults, and input-validation tests.
"""

import itertools

import numpy as np
import pytest
import scipy.stats

import emg_from_lfp
from emg_from_lfp import _core

WP, WS = (300, 600), (275, 625)
BAND_KW = dict(wp=WP, ws=WS, gpass=1, gstop=60)


# --------------------------------------------------------------------------
# References + data
# --------------------------------------------------------------------------
def ref_per_window(data, data_sf, target_sf, window_size):
    """Transparent scalar reference for the per-window estimator."""
    n_chans, n_samps = data.shape
    window_n_samps = int(window_size * data_sf)
    centers = (np.arange(0, n_samps / data_sf, 1 / target_sf) * data_sf).astype(int)
    out = np.zeros((1, len(centers)))
    pairs = [(i, j) for i, j in itertools.product(range(n_chans), repeat=2) if i < j]
    for i, j in pairs:
        for s, c in enumerate(centers):
            lo = max(0, int(c - window_n_samps / 2))
            hi = min(n_samps, int(c + window_n_samps / 2))
            out[0, s] += scipy.stats.pearsonr(data[i, lo:hi + 1], data[j, lo:hi + 1])[0]
    return out / len(pairs)


def ref_global(data, data_sf, target_sf, window_size):
    """Transparent reference for the global (amplitude-weighted) estimator."""
    data = data.astype(np.float64)
    n_chans, n_samps = data.shape
    z = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    centers = (np.arange(0, n_samps / data_sf, 1 / target_sf) * data_sf).astype(int)
    window_n_samps = int(window_size * data_sf)
    pairs = [(i, j) for i, j in itertools.product(range(n_chans), repeat=2) if i < j]
    out = np.zeros((1, len(centers)))
    for s, c in enumerate(centers):
        lo = max(0, int(c - window_n_samps / 2))
        hi = min(n_samps, int(c + window_n_samps / 2))
        seg = z[:, lo:hi + 1]
        out[0, s] = np.mean([np.mean(seg[i] * seg[j]) for i, j in pairs])
    return out


def make_data(n_chans, n_samps, seed=0, shared=0.5):
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_samps)
    data = shared * common[None, :] + (1 - shared) * rng.standard_normal((n_chans, n_samps))
    return data.astype(np.float32)


GRID = [
    (n_chans, n_samps, data_sf, target_sf, window_size)
    for n_chans in (2, 3, 5)
    for n_samps in (3000, 12000)
    for data_sf in (500, 1500)
    for target_sf in (20, 50)
    for window_size in (0.5, 30.0)  # window_n_samps both < and > n_samps
]


# --------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_chans,n_samps,data_sf,target_sf,window_size", GRID)
def test_per_window_matches_reference(n_chans, n_samps, data_sf, target_sf, window_size):
    data = make_data(n_chans, n_samps)
    got = _core._compute_av_corr(data, data_sf, target_sf, window_size)
    ref = ref_per_window(data, data_sf, target_sf, window_size)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8, equal_nan=True)


@pytest.mark.parametrize("n_chans,n_samps,data_sf,target_sf,window_size", GRID)
def test_global_matches_reference(n_chans, n_samps, data_sf, target_sf, window_size):
    data = make_data(n_chans, n_samps)
    got = _core._compute_global_corr(data, data_sf, target_sf, window_size)
    ref = ref_global(data, data_sf, target_sf, window_size)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-10, equal_nan=True)


def test_constant_channel_nan_parity():
    data = make_data(3, 12000)
    data[1, : 12000 // 2] = 7.0  # constant over first half -> zero-variance windows
    got = _core._compute_av_corr(data, 1500, 20, 0.5)
    ref = ref_per_window(data, 1500, 20, 0.5)
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    finite = ~np.isnan(ref)
    np.testing.assert_allclose(got[finite], ref[finite], rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------
# Public API: return types, defaults
# --------------------------------------------------------------------------
def test_both_returns_dict():
    data = make_data(3, int(1500 * 5))
    res = emg_from_lfp.compute(data, 1500.0, method="both", **BAND_KW)
    assert isinstance(res, dict) and set(res) == {"per_window", "global"}
    assert res["per_window"].shape == res["global"].shape


def test_single_returns_array():
    data = make_data(3, int(1500 * 5))
    res = emg_from_lfp.compute(data, 1500.0, method="per_window", **BAND_KW)
    assert isinstance(res, np.ndarray) and res.shape[0] == 1


def test_defaults_run():
    # compute(lfp, sf) should work out-of-the-box on the package defaults.
    data = make_data(3, int(1500 * 5))
    res = emg_from_lfp.compute(data, 1500.0)
    assert set(res) == {"per_window", "global"}
    assert emg_from_lfp.DEFAULTS["method"] == "both"


def test_unknown_method_raises():
    data = make_data(3, 3000)
    with pytest.raises(ValueError, match="Unknown EMG method"):
        emg_from_lfp.compute(data, 1500.0, method="nope")


# --------------------------------------------------------------------------
# Input validation (informative errors)
# --------------------------------------------------------------------------
def test_nyquist_violation_raises():
    data = make_data(3, 3000)
    # sf=1000 -> Nyquist 500 Hz; default ws upper edge 625 Hz is above it.
    with pytest.raises(ValueError, match="Nyquist"):
        emg_from_lfp.compute(data, 1000.0)


def test_nonpositive_sf_raises():
    data = make_data(3, 3000)
    with pytest.raises(ValueError, match="positive"):
        emg_from_lfp.compute(data, 0.0)


def test_nonpositive_band_edge_raises():
    data = make_data(3, 3000)
    with pytest.raises(ValueError, match="positive"):
        emg_from_lfp.compute(data, 1500.0, wp=(-10, 600), ws=(275, 625))
