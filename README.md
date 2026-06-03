# emg_from_lfp

Derive a **synthetic EMG** from LFP (or any multichannel field-potential
recording) by measuring the correlation of high-frequency activity across
spatially separated channels in sliding windows. When channels share a common
signal — e.g. EMG contamination during movement — their high-frequency activity
co-varies, so this correlation tracks muscle tone without a dedicated EMG
electrode. Widely used as a movement/arousal proxy for sleep scoring.

This is a self-contained port of Buzsaki's `bz_EMGFromLFP.m`, based on Erik
Schomburg's method (Schomburg et al., *Neuron* 2014). Dependencies are minimal
(`numpy`, `scipy`, `numba`).

## Install

```bash
pip install git+https://github.com/CSC-UW/emg_from_lfp.git
```

## Usage

```python
import numpy as np
from emg_from_lfp import compute, DEFAULTS

# lfp: (n_channels, n_samples), sf: sampling rate in Hz (must be > 2 * max(ws)).
both = compute(lfp, sf)                      # default method="both" -> dict
pw = compute(lfp, sf, method="per_window")   # exact per-window correlation
gl = compute(lfp, sf, method="global")       # fast amplitude-weighted approx.

print(DEFAULTS)  # target_sf=20 Hz, window_size=25 s, band 300-600 Hz, ...
```

### Methods

- **`per_window`** — exact mean pairwise Pearson correlation, re-normalized
  within each window. Amplitude-independent, bounded to `[-1, 1]`. Computed with
  an incremental sliding-window numba kernel (visits each sample O(1) times).
- **`global`** — faster global-normalization approximation (a single boxcar
  moving average). Tracks the shape/ranking of `per_window` closely but is
  **amplitude-weighted** and not bounded to `[-1, 1]`.
- **`both`** (default) — returns both as a dict; the band-pass filter (the
  dominant cost) is computed once and shared.

A `(near-)constant channel within a window yields `NaN` for that window,
matching `scipy.stats.pearsonr`.

## Citation

If you use this method, please cite:

> Schomburg, E. W., et al. (2014). Theta phase segregation of input-specific
> gamma patterns in entorhinal-hippocampal networks. *Neuron*, 84(2), 470-485.

Original MATLAB implementation: `bz_EMGFromLFP.m` (Buzsaki lab, buzcode).
Original Python port: Tom Bugnon (2020).
