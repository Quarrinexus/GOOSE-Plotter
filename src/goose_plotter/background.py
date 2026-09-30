"""The Background section: a polynomial in x fitted to a line, shown or subtracted."""

import warnings

import numpy as np
from numpy.polynomial import Chebyshev

# Mode stored on a Line -> name shown in the Background box. "" is off.
MODES = {"": "Off", "fit": "Show fit", "subtract": "Subtract"}


def describe(mode, degree):
    """Short text for the legend and toggle: '− deg 10', 'fit deg 10'."""
    return f"{'fit' if mode == 'fit' else '−'} deg {degree}"


def file_part(mode, degree):
    """Filename piece: 'bg10' for a subtracted fit, 'fit10' for the fit itself."""
    return f"{'fit' if mode == 'fit' else 'bg'}{degree}"


def apply(x, y, mode, degree):
    """`y` with the fit subtracted, or the fit itself, for mode 'subtract' or 'fit'.

    The fit is a degree-`degree` least-squares polynomial in x over the whole
    line; to fit part of it, cut it to that part first (Splicing). Points that
    aren't finite are left out of the fit and stay gaps."""
    fitted = fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), degree)
    return fitted if mode == "fit" else y - fitted


def fit(x, y, degree):
    """The fitted polynomial at each finite x, NaN elsewhere."""
    if degree < 0:
        raise ValueError("the degree can't be negative")
    inside = np.isfinite(x)
    used = inside & np.isfinite(y)
    if not used.any():
        raise ValueError("there are no points to fit")
    distinct = len(np.unique(x[used]))
    if distinct <= degree:
        raise ValueError(f"a degree-{degree} fit needs more than {degree} different "
                         f"x values; the line has {distinct}")
    if distinct == 1:  # degree 0 on a single x (a parked field): just the mean
        return np.where(inside, y[used].mean(), np.nan)
    # Chebyshev.fit maps x onto [-1, 1] first, so high degrees stay well
    # conditioned even over a narrow range like 1/B = 0.036-0.067.
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        try:
            polynomial = Chebyshev.fit(x[used], y[used], degree)
        except np.exceptions.RankWarning:
            raise ValueError(f"degree {degree} is too high for the line's "
                             f"points; try a lower one") from None
    return np.where(inside, polynomial(np.where(inside, x, 0.0)), np.nan)
