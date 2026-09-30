"""The Fit section: a polynomial in x, or a function of the user's own,
fitted to a line, shown or subtracted."""

import math
import re
import warnings

import numpy as np
from numpy.polynomial import Chebyshev

from goose_plotter.axis_functions import FUNCTIONS, slug

# Mode stored on a Line -> name shown in the Fit box. "" is off.
MODES = {"": "Off", "fit": "Show fit", "subtract": "Subtract"}


def describe(mode, degree, function="", start=""):
    """Short text for the legend and toggle: '− deg 5', 'fit deg 5',
    '− A * sin(B * x) + C'."""
    shape = f"deg {degree}" if degree is not None else function or "custom"
    return f"{'fit' if mode == 'fit' else '−'} {shape}"


def file_part(mode, degree, function="", start=""):
    """Filename piece: 'bg10' for a subtracted fit, 'fit10' for the fit itself,
    'bg_A_-times-_x_+_B' for a function's ('A * x + B')."""
    prefix = "fit" if mode == "fit" else "bg"
    if degree is not None:
        return f"{prefix}{degree}"
    return f"{prefix}_{slug(function) or 'custom'}"


def apply(x, y, mode, degree, function="", start=""):
    """(`y` with the fit subtracted, or the fit itself, for mode 'subtract' or
    'fit'; the fitted parameters as ((name, value), ...), () for a polynomial).

    The fit is a degree-`degree` least-squares polynomial in x, or with a
    `function` that function (see `custom_fit`), over the whole line; to fit
    part of it, cut it to that part first (Splicing). Points that aren't
    finite are left out of the fit and stay gaps."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if degree is None:  # Advanced Fitting, even with no function typed yet
        fitted, values = custom_fit(x, y, function, start)
    else:
        fitted, values = fit(x, y, degree), {}
    return (fitted if mode == "fit" else y - fitted), tuple(values.items())


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


# --- a function of the user's own ---------------------------------------------

def polynomial_text(degree):
    """What `fit` fits, as a function to type: 'A*x**2 + B*x + C' for degree 2."""
    names = [chr(65 + i) if i < 26 else chr(64 + i // 26) + chr(65 + i % 26)
             for i in range(max(degree, 0) + 1)]
    powers = range(max(degree, 0), -1, -1)
    return " + ".join(name + ("" if n == 0 else "*x" if n == 1 else f"*x**{n}")
                      for name, n in zip(names, powers))


def parse(expr):
    """(compiled function of x, its parameters in order of first use) for a
    function like 'A * sin(B * x) + C': capital-letter names are the unknowns.
    Products need their '*' ('A sin(x)' is refused); '^' is a power."""
    if not expr.strip():
        raise ValueError("type a function of x to fit, e.g. A * sin(B * x) + C")
    try:
        code = compile(expr.replace("^", "**"), "<fit function>", "eval")
    except SyntaxError:
        raise ValueError(f"can't read the function '{expr}'; write * for every product, "
                         f"e.g. A * sin(B * x)") from None
    names = list(dict.fromkeys(code.co_names))
    params = [n for n in names if n[0].isupper()]
    unknown = [n for n in names if n not in params and n != "x" and n not in FUNCTIONS]
    if unknown:
        raise ValueError(f"'{', '.join(unknown)}' in the fit function isn't x, a function "
                         f"or a parameter (parameters start with a capital letter)")
    if not params:
        raise ValueError("the fit function has nothing to fit: give it parameters "
                         "starting with a capital letter, e.g. A * sin(B * x) + C")
    return code, params


def starts(text, params):
    """Start values for `params` from text like 'B=314, C=0'; 1 for any not given."""
    values = dict.fromkeys(params, 1.0)
    for part in re.split(r"[,;]", text or ""):
        if not part.strip():
            continue
        name, sep, number = part.partition("=")
        name = name.strip()
        if not sep or name not in values:
            raise ValueError(f"'{part.strip()}' isn't a start value like "
                             f"{params[0]}=1 for one of {', '.join(params)}")
        try:
            values[name] = float(number)
        except ValueError:
            raise ValueError(f"'{number.strip()}' isn't a number, in '{part.strip()}'") from None
        if not math.isfinite(values[name]):
            raise ValueError(f"the start value of {name} isn't finite")
    return values


def _evaluate(code, params, values, x):
    with np.errstate(all="ignore"):
        out = eval(code, {"__builtins__": {}, **FUNCTIONS},
                   {"x": x, **dict(zip(params, values))})
    return np.broadcast_to(np.asarray(out, dtype=float), x.shape)


def custom_fit(x, y, expr, start_text=""):
    """(the fitted function at each finite x, NaN elsewhere; {parameter: value}),
    a least-squares fit of `expr` to the finite points, by Levenberg–Marquardt
    from the start values (1 unless `start_text` gives them)."""
    code, params = parse(expr)
    p = np.array(list(starts(start_text, params).values()))
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    inside = np.isfinite(x)
    used = inside & np.isfinite(y)
    if used.sum() <= len(params):
        raise ValueError(f"fitting {len(params)} parameters needs more than "
                         f"{len(params)} points; the line has {used.sum()}")
    xu, yu = x[used], y[used]
    try:
        values = _levenberg_marquardt(lambda q: _evaluate(code, params, q, xu) - yu, p)
    except NameError as err:  # a name used as a function that isn't one
        raise ValueError(str(err)) from None
    except TypeError as err:  # e.g. calling a parameter: 'A(x)'
        raise ValueError(f"can't evaluate the fit function: {err}") from None
    fitted = np.full(x.shape, np.nan)
    fitted[inside] = _evaluate(code, params, values, x[inside])
    return fitted, dict(zip(params, (float(v) for v in values)))


def _levenberg_marquardt(residuals, p, iterations=200):
    """Parameters minimising the sum of squared `residuals(p)`."""
    r = residuals(p)
    if not np.all(np.isfinite(r)):
        raise ValueError("the fit function isn't finite at the start values; "
                         "give start values where it is")
    cost, damping = r @ r, 1e-3
    for _ in range(iterations):
        jac = np.empty((len(r), len(p)))
        for i in range(len(p)):  # forward differences
            step = np.sqrt(np.finfo(float).eps) * max(abs(p[i]), 1.0)
            moved = p.copy()
            moved[i] += step
            jac[:, i] = (residuals(moved) - r) / step
        if not np.all(np.isfinite(jac)):
            raise ValueError("the fit function stopped being finite while fitting; "
                             "try other start values")
        # Columns scaled to norm 1 and solved as least squares, not through
        # JᵀJ, whose squared condition loses a polynomial's high powers.
        norms = np.linalg.norm(jac, axis=0)
        norms[norms == 0] = 1.0
        scaled = jac / norms
        while True:
            try:
                stacked = np.vstack([scaled, np.sqrt(damping) * np.eye(len(p))])
                wanted = np.concatenate([-r, np.zeros(len(p))])
                delta = np.linalg.lstsq(stacked, wanted, rcond=None)[0] / norms
            except np.linalg.LinAlgError:
                delta = None
            if delta is not None:
                trial = p + delta
                r_trial = residuals(trial)
                cost_trial = r_trial @ r_trial if np.all(np.isfinite(r_trial)) else np.inf
                if cost_trial < cost:
                    break
            damping *= 10
            if damping > 1e16:
                return p  # nowhere downhill: a minimum, as far as it can tell
        done = abs(cost - cost_trial) <= 1e-12 * cost or np.all(
            np.abs(delta) <= 1e-10 * (np.abs(p) + 1e-10))
        p, r, cost = trial, r_trial, cost_trial
        damping = max(damping / 10, 1e-20)  # down to Gauss-Newton, lstsq keeps it stable
        if done:
            break
    return p
