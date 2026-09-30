"""The Despike section: points far off the median around them become gaps (a
Hampel filter), leaving every other point as it is."""

import warnings

import numpy as np

MAD_TO_SIGMA = 1.4826  # the MAD of normal noise times this is its standard deviation


def describe(window, threshold):
    """Short text for the legend and toggle: 'despiked 5σ/21'."""
    return f"despiked {threshold:g}σ/{window}"


def file_part(window, threshold):
    """Filename piece: 'ds5-21'."""
    return f"ds{threshold:g}-{window}"


def spikes(y, window, threshold):
    """Whether each point is a spike: more than `threshold` times the local
    spread from the median of the `window` rows around it, in row order (a
    Hampel filter).

    The spread is the window's median absolute deviation from that median,
    scaled to a standard deviation, so the spikes themselves hardly move it.
    It's at least half the line's typical spread, so a quiet stretch (or a
    parked field read to the same digits) isn't all spikes. Past the ends the
    window holds fewer points. NaN is never a spike."""
    if window < 3:
        raise ValueError("the window needs at least 3 points")
    if threshold <= 0:
        raise ValueError("the threshold must be above 0")
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        raise ValueError(f"a window of {window} points is longer than the line ({len(y)})")
    half = window // 2
    padded = np.concatenate([np.full(half, np.nan), y, np.full(window - 1 - half, np.nan)])
    rows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():  # all-NaN windows (inside a removed range) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(rows, axis=1)
        spread = MAD_TO_SIGMA * np.nanmedian(np.abs(rows - median[:, None]), axis=1)
    typical = np.nanmedian(spread[spread > 0]) if np.any(spread > 0) else 0.0
    spread = np.fmax(spread, typical / 2)
    with np.errstate(invalid="ignore"):
        return np.abs(y - median) > threshold * spread


def despike(x, y, window, threshold):
    """(x, y with the spikes turned to NaN in both, how many there were).
    NaN rather than dropped, like a removed cut: drawn as gaps, and left out
    of fits and x-unit windows."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    hit = spikes(y, window, threshold)
    return np.where(hit, np.nan, x), np.where(hit, np.nan, y), int(hit.sum())
