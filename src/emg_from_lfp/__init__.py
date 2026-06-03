"""Derive a synthetic EMG from LFP via correlation of high-frequency activity.

Public API:
    compute(lfp, sf, **kwargs) -> array | dict   # see DEFAULTS for parameters
    DEFAULTS                                       # default parameter dict
"""

from ._core import DEFAULTS, compute

__version__ = "0.1.0"
__all__ = ["compute", "DEFAULTS"]
