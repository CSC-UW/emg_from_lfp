# emg_from_lfp

Derive a **synthetic EMG** from LFP (or any multichannel field-potential
recording) by measuring the correlation of high-frequency activity across
spatially separated channels in sliding windows. When channels share a common
signal — e.g. EMG contamination during movement — their high-frequency activity
co-varies, so this correlation tracks muscle tone without a dedicated EMG
electrode. Useful as a movement/arousal proxy for sleep scoring.

This package originated as a self-contained port of the Buzsaki lab's
[`bz_EMGFromLFP.m`](https://github.com/buzsakilab/buzcode/blob/master/detectors/bz_EMGFromLFP.m), based on Erik Schomburg's method.
Further optimizations and extensions of the method have been added.

## Citation

If you use this method, please cite:

> Schomburg, E. W., et al. (2014). Theta phase segregation of input-specific
> gamma patterns in entorhinal-hippocampal networks. *Neuron*, 84(2), 470-485.

## Install

```bash
pip install emg-from-lfp
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
  within each window. Amplitude-independent, bounded to `[-1, 1]`. Numerically
  identical to the published method and reference implementation, but computed with
  an incremental sliding-window numba kernel (visits each sample O(1) times) for
  huge memory & compute gains.
- **`global`** — faster global-normalization approximation (a single boxcar
  moving average). Tracks the shape/ranking of `per_window` closely but is
  **amplitude-weighted** and not bounded to `[-1, 1]`. Sometimes this is desirable,
  since the amplitude-weighting may carry information about the intensity of movement.
- **`both`** (default) — returns both as a dict; the band-pass filter (the
  dominant cost) is computed once and shared.

A (near-)constant channel within a window yields `NaN` for that window,
matching `scipy.stats.pearsonr`.
