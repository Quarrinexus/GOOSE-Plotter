"""What's plotted: panels holding lines, and how those lines are coloured and labelled."""

from dataclasses import dataclass, field, replace
import math
from uuid import uuid4

from matplotlib.colors import to_rgb
import numpy as np

from goose_plotter import colourmaps
from goose_plotter.axis_functions import file_part
from goose_plotter.columns import sample_of
from goose_plotter.datasets import describe
from goose_plotter import background, despike, smoothing, splicing

# What a link between two panels can share, line by line: a Link's `sync`
# names the keys it shares.
SYNC = {"run": ("run",), "x": ("x",), "x_fn": ("x_fn",), "y": ("y",), "y_fn": ("y_fn",),
        "colour": ("colour",), "steps": ("steps",),
        "style": ("style", "width", "marker", "marker_size")}
SYNC_DEFAULT = "run x x_fn y y_fn colour style"
X_UNITS = ("span", "cuts")  # Step settings in the plotted x

AUTO_MARKER_SIZE = 3.0  # matplotlib's markersize, in points

NEUTRAL = "#2464d6"  # line colour when no sample column is plotted

# Colours for extra lines once a panel's sample colours are taken.
PALETTE = ("#2464d6", "#eb6834", "#2e9e5b", "#8e5bd1", "#c23b6e", "#1a9aa0",
           "#b58900", "#52514e")


# Step kind -> its name in the list and the Add step menu.
KINDS = {"cut": "Cut", "despike": "Despike", "subtract": "Subtract", "smooth": "Smooth",
         "fit": "Fit curve"}


@dataclass(frozen=True)
class Step:
    """One step of a line's processing, done in the line's order. Frozen, so a
    line's steps are a tuple that can key the data cache; change one with
    `dataclasses.replace`. It holds every kind's settings, only its own used.

    "subtract" takes its fit off the data for the steps after it; "fit" draws
    its fit of the data as it is there, over the line, leaving the data as it was."""
    kind: str = "smooth"  # a key of KINDS
    on: bool = True
    cut: str = "keep"  # cut: a key of splicing.MODES (not ""), for all the ranges
    cuts: tuple = ()  # cut: x ranges, (start, end) pairs as splicing.tidy makes them
    spike_window: int = 21  # despike: points around each one, for its median
    threshold: float = 5.0  # despike: how many local spreads off it makes a spike
    degree: int = 5  # subtract, fit: of the polynomial
    advanced: bool = False  # subtract, fit: fit `function` instead of the polynomial
    function: str = ""  # with capital-letter unknowns, e.g. "A * sin(B * x) + C"
    start: str = ""  # start values for its unknowns, e.g. "B=314"; 1 for any not given
    method: str = "savgol"  # smooth: a key of smoothing.METHODS other than ""
    window: int = 21  # smooth: in points
    in_x: bool = False  # smooth: window counted in the plotted x (span) instead
    span: float | None = None  # smooth: window in x; None: estimated from `window`
    order: int = 2  # smooth: Savitzky–Golay polynomial order
    colour: str | None = None  # fit: its curve's; None: picked automatically

    @property
    def cutting(self):
        """(mode, ranges), or None while it has no ranges."""
        return (self.cut, self.cuts) if self.kind == "cut" and self.cut and self.cuts else None

    @property
    def despiking(self):
        return (self.spike_window, self.threshold) if self.kind == "despike" else None

    @property
    def fitting(self):
        """(mode, degree, function, start), as background.apply takes it, the
        mode "subtract" or "fit" by kind; the function and start "" for the
        polynomial, the degree None for a function."""
        if self.kind not in ("subtract", "fit"):
            return None
        if self.advanced:
            return (self.kind, None, self.function, self.start)
        return (self.kind, self.degree, "", "")

    @property
    def smoothing(self):
        """(method, window or span, order, in_x), as smoothing.smooth takes it."""
        if self.kind != "smooth" or not self.method:
            return None
        return (self.method, self.span if self.in_x else self.window, self.order, self.in_x)

    @property
    def active(self):
        """Whether it does anything: on, and a cut with ranges to cut by."""
        return self.on and (self.kind != "cut" or self.cutting is not None)

    def describe(self):
        """Its settings in short, for the step list, the legend and the line list."""
        if self.cutting:
            return splicing.describe(*self.cutting)
        if self.kind == "cut":
            return "no ranges"
        if self.despiking:
            return despike.describe(*self.despiking)
        if self.fitting:
            fitted = background.describe(*self.fitting)
            return fitted if self.kind == "subtract" else fitted.removeprefix("fit ")
        return smoothing.describe(*self.smoothing) if self.smoothing else ""

    def file_part(self):
        """Its piece of a figure's filename."""
        if self.cutting:
            return splicing.file_part(*self.cutting)
        if self.despiking:
            return despike.file_part(*self.despiking)
        if self.fitting:
            return background.file_part(*self.fitting)
        return smoothing.file_part(*self.smoothing) if self.smoothing else ""


@dataclass
class Line:
    """One plotted line. x and y are column names."""
    run: str = ""
    x: str = ""  # "": the profile's default, see Plotter._load
    x_fn: str = "x"
    y: str = ""
    y_fn: str = "y"
    steps: tuple = ()  # its processing, Steps in the order they're done
    step: int = 0  # the step the Process tab shows; not saved, like Panel.selected
    # What the last draw found, per step: spikes removed, or a fit's
    # ((name, value), ...); None for the rest. Not saved.
    found: tuple = ()
    colour: str | None = None  # None: picked automatically, see line_colours
    # How it's drawn; not in `shown`, so changing them keeps the zoom.
    style: str = "-"  # a key of STYLES
    width: float | None = None  # None: auto_width
    marker: str = ""  # a key of MARKERS
    marker_size: float | None = None  # None: AUTO_MARKER_SIZE
    label: str | None = None  # its name in the legend; None: the automatic one, "": none
    # (run, x, x_fn, y, y_fn, steps) as last drawn
    shown: tuple | None = None
    error: str = ""  # why the last draw failed, if it did

    def copy(self):
        return replace(self, shown=None, error="", found=())

    auto_width = 0.7  # the width when none is set

    def plot_style(self):
        """matplotlib keywords for its line and markers (colour aside)."""
        kwargs = {"ls": "None" if self.style == "none" else self.style,
                  "lw": self.auto_width if self.width is None else self.width}
        if self.marker:
            kwargs |= {"marker": self.marker,
                       "ms": AUTO_MARKER_SIZE if self.marker_size is None else self.marker_size}
        return kwargs

    @property
    def chosen(self):
        """The step the Process tab shows, or None if it has none."""
        return self.steps[min(self.step, len(self.steps) - 1)] if self.steps else None

    def clear_x_units(self):
        """Forget the step settings in the plotted x (X_UNITS), when x or its function changes."""
        self.steps = without_x_units(self.steps)

    def parts(self):
        """(run, y part, x part, the steps that do anything) as they'd appear
        in a filename and the legend."""
        run, x, x_fn, y, y_fn, steps = self.shown
        return (run, file_part(y_fn, y), file_part(x_fn, x),
                tuple(s for s in steps if s.active))


def without_x_units(steps):
    """`steps` with their settings in the plotted x (X_UNITS) forgotten."""
    return tuple(replace(s, cuts=(), span=None) for s in steps)


@dataclass
class Link:
    """A link between two panels (the Plotter keeps them by the pair of their
    ids): which settings it shares, line by line, both ways, and whether it's
    frozen, kept but sharing nothing for now."""
    sync: str = SYNC_DEFAULT  # SYNC keys, space-separated
    frozen: bool = False

    @property
    def synced(self):
        """The SYNC keys in `sync` (unknown words, e.g. from a session, are ignored)."""
        return set(self.sync.split()) & set(SYNC)


@dataclass
class Panel:
    """One subplot: its lines and which of them the controls edit.

    A derived panel (an FFT or a derivative, by `operation`) draws that of its
    lines. It's made in its data panel's link group, with copies of its lines,
    so the two stay in step through the link (and can be frozen or partly
    synced); `source` is that data panel's cell while it still follows it.
    A data panel can also be turned into its own FFT or derivative, keeping
    what Undo needs in `data_view`.
    `id` names the panel for its links, as its cell can change."""
    lines: list = field(default_factory=lambda: [Line()])
    selected: int = 0
    source: tuple | None = None
    operation: str = ""  # a key of OPERATIONS; "" for a data panel
    window: str = "hann"  # FFT window, a key of spectrum.WINDOWS
    pad: int = 1  # FFT zero-padding factor
    f_max: float | None = None  # highest frequency drawn; None: all
    # FFT panels: the Splicing tab's cut of the spectrum itself, in its F, as a
    # line's cut is of the data (a key of splicing.MODES, and splicing.tidy ranges).
    cut: str = ""
    cuts: tuple = ()
    derivative_window: int = 51  # derivative panels: grid points per fit, odd
    # A data panel turned into its own FFT or derivative ("This panel"): its
    # DATA_VIEW values from before, for Undo. None for any other panel.
    data_view: tuple | None = None
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    # Typed axis ranges, in the plotted units; None: that end is automatic.
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    # Typed text; None for the automatic one, "" for none. Mathtext like $B$ works.
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    legend: str = "auto"  # a key of LEGENDS
    grid: str = "major"  # a key of GRIDS
    grid_axis: str = "both"  # a key of GRID_AXES
    grid_style: str = "-"  # a key of GRID_STYLES
    # Measure: an x region, in the plotted x as measure.tidy_region keeps it
    # (() for the whole line), and up to two points read off the selected line.
    region: tuple = ()
    points: tuple = ()
    marks: bool = False  # mark the max and min (or the peaks) on the plot
    marks_saved: bool = False  # and keep them in saved figures
    peak_count: int = 5  # FFT panels: how many peaks to list
    peak_floor: float = 10.0  # ignoring those under this % of the highest

    @property
    def line(self):
        return self.lines[self.selected]

    @property
    def derived(self):
        return bool(self.operation)

    def copy(self):
        """An independent data panel with copies of the lines (and nothing else of
        the panel's: ranges and such start fresh)."""
        return Panel([l.copy() for l in self.lines], self.selected)


# Line style and marker -> the text in their menus.
STYLES = {"-": "Solid", "--": "Dashed", ":": "Dotted", "-.": "Dash-dot",
          "none": "None"}
MARKERS = {"": "None", ".": "Dots", "o": "Circles", "s": "Squares", "^": "Triangles",
           "x": "Crosses"}

RANGES = ("x_min", "x_max", "y_min", "y_max")
# What a data panel turned into its own FFT or derivative gets back on Undo.
DATA_VIEW = RANGES + ("title", "x_label", "y_label")


def tidy_data_view(saved):
    """`Panel.data_view` from a session: numbers or None for the ranges, text or
    None for the rest, else None (and the panel has no Undo)."""
    if not isinstance(saved, (list, tuple)) or len(saved) != len(DATA_VIEW):
        return None
    ranges, texts = saved[:len(RANGES)], saved[len(RANGES):]
    if not all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)
                             and math.isfinite(v)) for v in ranges):
        return None
    if not all(v is None or isinstance(v, str) for v in texts):
        return None
    return tuple(None if v is None else float(v) for v in ranges) + tuple(texts)

# Legend placement -> the text in its menu. "auto": only with two or more
# lines, wherever there's room; a placement shows it even for one line.
LEGENDS = {"auto": "Auto", "off": "Off", "upper right": "Top right",
           "upper left": "Top left", "lower left": "Bottom left",
           "lower right": "Bottom right", "outside": "Outside right"}


# What a panel draws of its lines: the lines themselves, their FFT, or a derivative.
OPERATIONS = ("", "fft", "d1", "d2")

# Grid lines -> the text on their buttons. "minor" draws the major lines too.
GRIDS = {"off": "Off", "major": "Major", "minor": "Major + minor"}
GRID_AXES = {"both": "Both", "x": "x only", "y": "y only"}
GRID_STYLES = {"-": "Solid", "--": "Dashed", ":": "Dotted"}


def clear_ranges(panel, axes="xy"):
    """Forget typed ranges on those axes, e.g. when what's plotted on them changes,
    and what Measure holds in them: its region in x, its points in either."""
    for name in RANGES:
        if name[0] in axes:
            setattr(panel, name, None)
    if "x" in axes:
        panel.region = ()
        panel.cuts = ()  # an FFT's cut, in its F
    panel.points = ()


def clear_data_ranges(panel, axes="xy"):
    """Forget the ranges Undo would give a panel This panel turned, on those axes,
    when what they'd be plotted against changes (as `clear_ranges` does for its own)."""
    if panel.data_view is not None:
        panel.data_view = tuple(None if name in RANGES and name[0] in axes else value
                                for name, value in zip(DATA_VIEW, panel.data_view))


def swap_data_view(panel):
    """A turned panel's ranges and axis labels for Undo, swapped with its lines' x and y."""
    if panel.data_view is not None:
        v = dict(zip(DATA_VIEW, panel.data_view))
        for x, y in (("x_min", "y_min"), ("x_max", "y_max"), ("x_label", "y_label")):
            v[x], v[y] = v[y], v[x]
        panel.data_view = tuple(v[name] for name in DATA_VIEW)


def near(colour, others, distance=0.25):
    """True if `colour` is hard to tell from any of `others` (RGB distance)."""
    rgb = np.array(to_rgb(colour))
    return any(np.linalg.norm(rgb - to_rgb(o)) < distance for o in others)


def fit_curves(panel):
    """(line index, step index) of each fit curve the panel draws, in order:
    the fit steps that are on, on a data panel only."""
    if panel.derived:
        return []
    return [(i, j) for i, l in enumerate(panel.lines) for j, s in enumerate(l.steps)
            if s.kind == "fit" and s.on]


def curve_colours(panel, samples, colour_map="classic"):
    """([each line's colour], {(line, step): each fit curve's colour}): a pick,
    else from `colour_map` (a key of colourmaps.MAPS) spread over every curve
    drawn, each line followed by its fit curves; or with "classic" a line's
    sample's colour, else an unused PALETTE one, for lines and curves alike."""
    lines = panel.lines
    curves = fit_curves(panel)
    order = sorted([(i, -1) for i in range(len(lines))] + curves)  # each line, then its fits
    picked = {key: (lines[key[0]].colour if key[1] < 0 else lines[key[0]].steps[key[1]].colour)
              for key in order}
    colours = dict(picked)
    if colour_map == "classic":
        used = [c for c in picked.values() if c]
        for key in order:
            if colours[key]:
                continue
            wanted = NEUTRAL
            if key[1] < 0:
                l = lines[key[0]]
                wanted = samples.get(sample_of(samples, l.y, l.x), NEUTRAL)
            if near(wanted, used):
                wanted = next((c for c in PALETTE if not near(c, used)),
                              PALETTE[len(used) % len(PALETTE)])
            colours[key] = wanted
            used.append(wanted)
    else:
        # A picked colour keeps its curve's place, so picking one moves no other.
        for key, colour in zip(order, colourmaps.colours(colour_map, len(order))):
            colours[key] = colours[key] or colour
    return ([colours[(i, -1)] for i in range(len(lines))],
            {key: colours[key] for key in curves})


def line_colours(panel, samples, colour_map="classic"):
    """Each line's colour, as `curve_colours` gives it."""
    return curve_colours(panel, samples, colour_map)[0]


def legend_labels(lines):
    """Legend text naming only what differs between the lines. Fit curves
    don't change a line's data, so they're left out of what's compared."""
    parts = [l.parts()[:3] + (tuple(s for s in l.parts()[3] if s.kind != "fit"),)
             for l in lines]
    differs = [len({p[i] for p in parts}) > 1 for i in range(4)]
    if not any(differs[:3]):
        differs[1] = True  # identical, or only processed differently: say what's on y
    labels = []
    for run, y, x, steps in parts:
        bits = [describe(run)] if differs[0] else []
        if differs[1]:
            bits.append(y)
        if differs[2]:
            bits.append(f"vs {x}")
        if differs[3]:
            bits += [s.describe() for s in steps]
        labels.append(" · ".join(bits))
    return labels


def shared(labels):
    """One axis label for several lines, from (full, without-sample) pairs."""
    for texts in zip(*labels):
        unique = list(dict.fromkeys(texts))
        if len(unique) == 1:
            return unique[0]
    return " / ".join(dict.fromkeys(plain for _, plain in labels))
