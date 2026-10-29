"""The panels and their lines as plain JSON-ready data, for session files and undo."""

from dataclasses import fields
import math

from goose_plotter import background, measure, smoothing, spectrum, splicing
from goose_plotter.model import (GRID_AXES, GRID_STYLES, GRIDS, KINDS, LEGENDS, MARKERS,
                                 OPERATIONS, STYLES, SYNC, Line, Link, Panel, Step,
                                 tidy_data_view)
from goose_plotter.widgets import MAX_GRID

# 2: derived panels have their own lines, in their data panel's link group.
# 3: a title, axis label or legend name of None is the automatic one, "" none.
# 4: links are between pairs of panels, each with its own ticks and freeze.
# 5: lines can be cut (Splicing), and links can share the cut.
# 6: a line's cut is a list of ranges, all kept or all removed.
# 7: the background fits the whole line; its old x range is a cut.
# 8: lines can be despiked, and links can share it.
# 9: a line's processing is a list of steps, in any order (format 10).
VERSION = 9
KEY = "goose_plotter_session"  # the session file's marker, holding VERSION
# `dump`'s own marker, so `load` reads undo snapshots and files alike; its
# absence means the layout of version 1, where a derived panel shared its
# data panel's lines and every panel carried an operation. Before 3, "" was
# the automatic text; before 4, links were groups, each panel with its ticks;
# before 5, there was no cut to share; in 5, a line had one cut range;
# before 7, a line's background had an x range of its own; before 8, there
# was no despiking to share; before 9, a line's style could be "auto"; before
# 10, a line had a fixed cut, despiking, fit and smoothing instead of steps.
FORMAT = 10
OLD_FORMATS = (2, 3, 4, 5, 6, 7, 8, 9)  # read and brought up to date; others are version 1
TEXTS = {Panel: ("title", "x_label", "y_label"), Line: ("label",)}

# Not saved: what the last draw found, and what the controls show.
SKIP = {"shown", "error", "found", "step", "lines", "selected", "source"}
# Settings that must be one of a menu's keys; anything else gets the default.
# Per class: a Panel's window is the FFT's.
CHOICES = {Line: {"style": STYLES, "marker": MARKERS},
           Step: {"kind": KINDS, "cut": {k: v for k, v in splicing.MODES.items() if k},
                  "method": {k: v for k, v in smoothing.METHODS.items() if k}},
           Panel: {"legend": LEGENDS, "grid": GRIDS, "grid_axis": GRID_AXES,
                   "grid_style": GRID_STYLES, "operation": OPERATIONS, "window": spectrum.WINDOWS,
                   "cut": splicing.MODES,
                   "pad": spectrum.PADDING},
           Link: {}}


def _steps(saved):
    """`Line.steps` from a session: a Step from each object that is one."""
    if not isinstance(saved, list):
        return ()
    return tuple(_build(Step, data) for data in saved
                 if isinstance(data, dict) and data.get("kind") in KINDS)


# Fields holding a list, and what makes the saved one fit, per class as for CHOICES.
TIDY = {Line: {"steps": _steps},
        Step: {"cuts": splicing.tidy},
        Panel: {"data_view": tidy_data_view, "cuts": splicing.tidy, "region": measure.tidy_region,
                "points": measure.tidy_points},
        Link: {}}

# Numbers that only make sense above 0, per class as for CHOICES; the
# controls refuse the rest too.
POSITIVE = {Line: {"width", "marker_size"},
            Step: {"spike_window", "threshold", "window", "span"},
            Panel: {"f_max", "derivative_window", "peak_count"}, Link: set()}


def cell_key(cell):
    return f"{cell[0]},{cell[1]}"


def key_cell(key):
    r, c = key.split(",")
    return int(r), int(c)


def _plain(obj):
    return {f.name: getattr(obj, f.name) for f in fields(obj) if f.name not in SKIP}


def dump(panels, rows, cols, links):
    """The layout, every panel's settings and lines (with a derived panel's
    source cell while it has one), and the links, in a fixed order so equal
    states dump equal (undo compares them)."""
    out = {}
    for cell, p in sorted(panels.items()):
        data = _plain(p)
        data["lines"] = [{**_plain(l), "steps": [_plain(s) for s in l.steps]} for l in p.lines]
        if p.source is not None:
            data["source"] = cell_key(p.source)
        out[cell_key(cell)] = data
    saved_links = [{"a": a, "b": b, "sync": link.sync, "frozen": link.frozen}
                   for (a, b), link in sorted((tuple(sorted(pair)), link)
                                              for pair, link in links.items())]
    return {"format": FORMAT, "rows": rows, "cols": cols, "panels": out, "links": saved_links}


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
        raise ValueError(f"expected an object for a {cls.__name__.lower()}")
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


def _line(data, fmt=FORMAT):
    """A Line from `dump`'s data; before format 10 its processing was fixed
    fields, made into steps by `_old_steps`. Before 9 its style could be "auto"."""
    if isinstance(data, dict) and data.get("style") == "auto":
        # The old automatic style: dashed for a shown fit, else solid.
        data = {**data, "style": "--" if data.get("background") == "fit" else "-"}
    if fmt != FORMAT and isinstance(data, dict):
        data = {**data, "steps": None}
        line = _build(Line, data)
        line.steps = _old_steps(data)
        return line
    return _build(Line, data)


def _old(data, name, kinds, default, ok=lambda v: True):
    """An old line field's value from `data`, if it's of `kinds` and `ok`, else `default`."""
    value = data.get(name)
    if isinstance(value, bool) != (bool in kinds) or not isinstance(value, kinds):
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    return value if ok(value) else default


def _old_steps(data):
    """Steps from a line saved before format 10, in the order it did them: its
    cut, despiking, background fit, smoothing, and the fit a Fit as new line
    line made of that. Before 6 the cut was one range, `cut_from` to `cut_to`;
    before 7 a fit had an x range of its own, which only it saw, made a Keep
    cut here as it was then."""
    positive = lambda v: v > 0
    mode = _old(data, "cut", (str,), "")
    cuts = splicing.tidy(data["cuts"] if "cuts" in data
                         else [[data.get("cut_from"), data.get("cut_to")]])
    background_mode = _old(data, "background", (str,), "")
    if background_mode in ("subtract", "fit"):
        fit_range = splicing.tidy([[data.get("fit_from"), data.get("fit_to")]])
        if fit_range:
            start, end = fit_range[0]
            if mode not in ("keep", "remove") or not cuts:
                mode, cuts = "keep", ((start, end),)
            else:
                ranges = cuts if mode == "keep" else splicing.gaps(cuts, start, end)
                mode, cuts = "keep", splicing.within(ranges, start, end)
    steps = []
    if mode in ("keep", "remove") and cuts:
        steps.append(Step("cut", cut=mode, cuts=cuts))
    if data.get("despike") is True:
        steps.append(Step("despike", spike_window=_old(data, "despike_window", (int,), 21, positive),
                          threshold=float(_old(data, "despike_threshold", (int, float), 5.0,
                                               positive))))

    def fit_step(kind, fitting):
        """A Subtract or Fit curve step from an old (mode, degree, function, start)."""
        _, degree, function, start = fitting
        if degree is None:
            return Step(kind, advanced=True, function=function, start=start)
        return Step(kind, degree=degree)
    own = (background_mode, _old(data, "degree", (int,), 5, lambda v: v >= 0),
           _old(data, "fit_function", (str,), ""), _old(data, "fit_start", (str,), ""))
    if data.get("advanced") is True:
        own = (own[0], None) + own[2:]
    fits_drawn = data.get("fits_drawn") is True and background_mode == "fit"
    if fits_drawn:  # its copied line's fit, before the smoothing, then its own after
        first = background.tidy_fitting(data.get("first_fit"))
        if first:
            steps.append(fit_step(first[0], first))
    elif background_mode in ("subtract", "fit"):
        steps.append(fit_step(background_mode, own))
    method = _old(data, "smooth", (str,), "")
    if method in smoothing.METHODS and method:
        steps.append(Step("smooth", method=method, window=_old(data, "window", (int,), 21, positive),
                          in_x=data.get("in_x") is True,
                          span=_old(data, "span", (int, float), None, positive),
                          order=_old(data, "order", (int,), 2)))
    if fits_drawn:
        steps.append(fit_step("fit", own))
    return tuple(steps)


def load(state):
    """(rows, cols, panels, links) from `dump`'s output, links being
    {frozenset of two panel ids: Link}. Raises ValueError if it isn't one."""
    try:
        rows, cols = int(state["rows"]), int(state["cols"])
        saved = {key_cell(k): v for k, v in state["panels"].items()}
    except (KeyError, TypeError, ValueError, AttributeError) as err:
        raise ValueError(f"not a session ({err})") from None
    if not (1 <= rows <= MAX_GRID and 1 <= cols <= MAX_GRID):
        raise ValueError(f"a {rows} x {cols} layout is bigger than the plotter allows")
    grid = [(r, c) for r in range(rows) for c in range(cols)]
    fmt = state.get("format")
    if fmt not in (*OLD_FORMATS, FORMAT):
        panels, groups = _load_old(saved, grid)
        return rows, cols, _blank_is_auto(panels), _group_links(panels, groups)
    panels = {}
    for cell in grid:
        data = saved.get(cell)
        if data is None:
            panels[cell] = Panel()
            continue
        lines = [_line(l, fmt) for l in data.get("lines") or []] or [Line()]
        panels[cell] = _build(Panel, data, lines=lines, source=_source(data, grid))
    seen = set()
    for p in panels.values():  # ids must tell panels apart; a file edited by hand may not
        if p.id in seen:
            p.id = Panel().id
        seen.add(p.id)
    if fmt in (8, 9, FORMAT):
        return rows, cols, panels, _links(state.get("links"), panels)
    # Before 8 some of what is now "steps" couldn't be shared (the cut before 5,
    # despiking before 8), so a link that shared all the rest shares it.
    new = {"steps"}
    if fmt in (4, 5, 6, 7):
        return rows, cols, panels, _share_new(_links(state.get("links"), panels), panels, new)
    groups = {cell: _group_of(saved.get(cell), old_ticks=False) for cell in panels}
    if fmt == 2:
        panels = _blank_is_auto(panels)
    return rows, cols, panels, _share_new(_group_links(panels, groups), panels, new)


def _share_new(links, panels, new):
    """Links saved before all that the SYNC keys in `new` share could be
    shared (the cut before format 5, despiking before 8, both now "steps").
    Those that shared everything else (a derived panel's with its data panel,
    at least) share them too, so an FFT or derivative still follows its data panel."""
    derived = {frozenset((p.id, panels[p.source].id)) for p in panels.values()
               if p.derived and p.source in panels}
    for pair, link in links.items():
        if pair in derived or set(SYNC) - new <= link.synced:
            link.sync = " ".join(k for k in SYNC if k in link.synced | new)
    return links


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
            link.sync = _new_words(link.sync.split())
            links[pair] = link
    return links


# SYNC keys before format 10, each now part of "steps".
STEP_WORDS = {"cut", "despike", "smoothing", "background"}


def _new_words(words):
    """A link's `sync` from its words, any of the old keys for what's now a
    line's steps sharing "steps"."""
    words = set(words)
    if words & STEP_WORDS:
        words.add("steps")
    return " ".join(k for k in SYNC if k in words)


def _group_of(data, old_ticks):
    """(group, ticks, frozen) of a panel as saved before format 4, when links
    were groups and each panel ticked what it shared; None if in no group.
    `old_ticks`: X and Y axis then covered their Function boxes too."""
    if not isinstance(data, dict) or not isinstance(data.get("link_group"), int) \
            or isinstance(data.get("link_group"), bool):
        return None
    words = data.get("sync") if isinstance(data.get("sync"), str) else Link().sync
    words = words.split()
    if old_ticks or not {"x_fn", "y_fn"} & set(words):
        words += [k for k, axis in (("x_fn", "x"), ("y_fn", "y")) if axis in words]
    return data["link_group"], set(_new_words(words).split()), data.get("frozen") is True


def _group_links(panels, groups):
    """Links for panels saved in groups ({cell: _group_of}): every pair in a
    group, sharing what both ticked, frozen if either was, as they behaved."""
    links = {}
    members = [(cell, g) for cell, g in groups.items() if g is not None and cell in panels]
    for i, (a, (group_a, ticks_a, frozen_a)) in enumerate(members):
        for b, (group_b, ticks_b, frozen_b) in members[i + 1:]:
            if group_a == group_b:
                shared = ticks_a & ticks_b
                links[frozenset((panels[a].id, panels[b].id))] = Link(
                    " ".join(k for k in SYNC if k in shared), frozen_a or frozen_b)
    return links


def _blank_is_auto(panels):
    """Before format 3 a blank title, label or legend name was the automatic one."""
    for p in panels.values():
        for obj in (p, *p.lines):
            for name in TEXTS[type(obj)]:
                if getattr(obj, name) == "":
                    setattr(obj, name, None)
    return panels


def _source(data, grid):
    """The cell in `data`'s "source", if it's one of the grid's."""
    try:
        source = key_cell(data["source"])
    except (TypeError, KeyError, ValueError, AttributeError):
        return None
    return source if source in grid else None


def _load_old(saved, grid):
    """(panels, groups) from a version 1 layout. There a derived panel had a
    source and no lines of its own, and all panels an operation ("fft" by
    default). Here it gets copies of its data panel's lines, in its group,
    sharing everything, as that list did."""
    panels, groups = {}, {}
    for cell in grid:  # data panels first: derived panels copy their lines
        data = saved.get(cell)
        if data is None or (isinstance(data, dict) and "source" in data):
            continue
        lines = [_line(l, 1) for l in data.get("lines") or []] or [Line()]
        panels[cell] = _build(Panel, data, lines=lines)
        panels[cell].operation = ""
        groups[cell] = _group_of(data, old_ticks=True)
    numbers = [g[0] for g in groups.values() if g is not None]
    for cell in grid:
        data = saved.get(cell)
        if cell in panels:
            continue
        source = _source(data, grid) if isinstance(data, dict) else None
        if source not in panels:  # missing, or its data panel isn't there: an empty panel
            panels[cell] = Panel()
            continue
        data_panel = panels[source]
        p = _build(Panel, data, lines=[l.copy() for l in data_panel.lines], source=source)
        p.operation = p.operation or "fft"
        p.selected = min(data_panel.selected, len(p.lines) - 1)
        if groups[source] is None:
            numbers.append(max(numbers, default=0) + 1)
            groups[source] = numbers[-1], set(SYNC), False
        groups[source] = groups[source][0], set(SYNC), groups[source][2]  # shares everything
        groups[cell] = groups[source][0], set(SYNC), False
        panels[cell] = p
    return panels, groups
