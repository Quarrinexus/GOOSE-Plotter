"""The panels and their lines as plain JSON-ready data, for session files and undo."""

import math
from dataclasses import fields

from goose_plotter import derivative, measure, smoothing, spectrum, splicing
from goose_plotter.model import (
    GRID_AXES,
    GRID_STYLES,
    GRIDS,
    KINDS,
    LEGENDS,
    MARKERS,
    STYLES,
    SYNC,
    Layout,
    Line,
    Link,
    Panel,
    Step,
)

# Session files carry KEY: VERSION, and `dump`'s output (files and undo
# snapshots alike) "format": FORMAT. Only this format is read: a session
# saved by an older plotter is refused, not converted.
VERSION = 11
KEY = "goose_plotter_session"
FORMAT = 12
# Not saved: what the last draw found, and what the controls show.
SKIP = {"shown", "error", "found", "step", "picked", "lines", "selected", "source"}
# Settings that must be one of a menu's keys; anything else gets the default.
# Per class: a Panel's window is the FFT's.
CHOICES = {Line: {"style": STYLES, "marker": MARKERS},
           Step: {"kind": KINDS, "cut": {"remove": splicing.MODES["remove"]},
                  "fft_window": spectrum.WINDOWS, "pad": spectrum.PADDING,
                  "d_order": derivative.ORDERS,
                  "method": {k: v for k, v in smoothing.METHODS.items() if k}},
           Panel: {"legend": LEGENDS, "grid": GRIDS, "grid_axis": GRID_AXES,
                   "grid_style": GRID_STYLES},
           Link: {}}


def _steps(saved):
    """`Line.steps` from a session: a Step from each object that is one."""
    if not isinstance(saved, list):
        return ()
    return tuple(_build(Step, data) for data in saved
                 if isinstance(data, dict) and data.get("kind") in KINDS)


def dump_steps(steps):
    """Steps as plain data, as a session holds them (for recipes too)."""
    return [_plain(s) for s in steps]


def load_steps(saved):
    """Steps from `dump_steps`'s data, dropping anything that isn't one."""
    return _steps(saved)


# Fields holding a list, and what makes the saved one fit, per class as for CHOICES.
TIDY = {Line: {"steps": _steps},
        Step: {"cuts": splicing.tidy},
        Panel: {"region": measure.tidy_region,
                "points": measure.tidy_points},
        Link: {}}

# Numbers that only make sense above 0, per class as for CHOICES; the
# controls refuse the rest too.
POSITIVE = {Line: {"width", "marker_size"},
            Step: {"spike_window", "threshold", "window", "span", "f_max", "d_window"},
            Panel: {"peak_count"}, Link: set()}


def cell_key(cell):
    return f"{cell[0]},{cell[1]}"


def key_cell(key):
    r, c = key.split(",")
    return int(r), int(c)


def _plain(obj):
    return {f.name: getattr(obj, f.name) for f in fields(obj) if f.name not in SKIP}


def dump(panels, layout, links):
    """The layout, every panel's settings and lines (with a derived panel's
    source cell while it has one), and the links, in a fixed order so equal
    states dump equal (undo compares them)."""
    out = {}
    for cell, p in sorted(panels.items()):
        data = _plain(p)
        data["lines"] = [{**_plain(l), "steps": dump_steps(l.steps)} for l in p.lines]
        if p.source is not None:
            data["source"] = cell_key(p.source)
        out[cell_key(cell)] = data
    saved_links = [{"a": a, "b": b, "sync": link.sync, "frozen": link.frozen}
                   for (a, b), link in sorted((tuple(sorted(pair)), link)
                                              for pair, link in links.items())]
    return {"format": FORMAT, "layout": layout.plain(), "panels": out, "links": saved_links}


def _allowed(f, value):
    """Whether `value` fits field `f`'s type (from its annotation, e.g. 'float | None')."""
    kind = str(f.type)
    if value is None:
        return "None" in kind
    if isinstance(value, bool):
        return "bool" in kind
    if isinstance(value, (int, float)):
        if not math.isfinite(value):  # JSON's NaN and Infinity would break drawing
            return False
        return "float" in kind or ("int" in kind and isinstance(value, int))
    return isinstance(value, str) and "str" in kind


def _build(cls, data, **extra):
    """A `cls` from `data`, keeping only known fields of the right type."""
    if not isinstance(data, dict):
        raise ValueError(f"expected an object for a {cls.__name__.lower()}")  # noqa: TRY004
    kwargs = {}
    for f in fields(cls):
        if f.name in SKIP or f.name not in data:
            continue
        if f.name in TIDY[cls]:  # JSON's lists, checked item by item
            kwargs[f.name] = TIDY[cls][f.name](data[f.name])
            continue
        if not _allowed(f, data[f.name]):
            continue
        value = data[f.name]
        if f.name in CHOICES[cls] and value not in CHOICES[cls][f.name]:
            continue
        if f.name in POSITIVE[cls] and value is not None and value <= 0:
            continue
        kwargs[f.name] = value
    return cls(**kwargs, **extra)


def load(state):
    """(layout, panels, links) from `dump`'s output, links being {frozenset
    of two panel ids: Link}. Raises ValueError, changing nothing, for anything
    that isn't a session in FORMAT, an older plotter's included."""
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("it isn't a session in this plotter's format")
    try:
        layout = Layout.from_plain(state["layout"])
        saved = {key_cell(k): v for k, v in state["panels"].items()}
    except (KeyError, TypeError, AttributeError) as err:
        raise ValueError(f"not a session ({err})") from None
    grid = layout.cells()
    panels = {}
    for cell in grid:
        data = saved.get(cell)
        if data is None:
            panels[cell] = Panel()
            continue
        lines = [_build(Line, l) for l in data.get("lines") or []] or [Line()]
        panels[cell] = _build(Panel, data, lines=lines, source=_source(data, grid))
    seen = set()
    for p in panels.values():  # ids must tell panels apart; a file edited by hand may not
        if p.id in seen:
            p.id = Panel().id
        seen.add(p.id)
    return layout, panels, _links(state.get("links"), panels)


def _links(saved, panels):
    """{frozenset of ids: Link} from `dump`'s list, keeping only links between
    two different panels that are there, once each."""
    ids = {p.id for p in panels.values()}
    links = {}
    for data in saved if isinstance(saved, list) else []:
        if not isinstance(data, dict):
            continue
        pair = frozenset((data.get("a"), data.get("b")))
        if len(pair) == 2 and pair <= ids and pair not in links:
            link = _build(Link, data)
            link.sync = " ".join(k for k in SYNC if k in link.sync.split())
            links[pair] = link
    return links


def _source(data, grid):
    """The cell in `data`'s "source", if it's one of the grid's."""
    try:
        source = key_cell(data["source"])
    except (TypeError, KeyError, ValueError, AttributeError):
        return None
    return source if source in grid else None
