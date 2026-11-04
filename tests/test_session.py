"""Sessions and undo snapshots: the panels to plain data and back."""

import json

import pytest

from goose_plotter import session
from goose_plotter.model import (
    SYNC,
    Layout,
    Line,
    Link,
    Panel,
    Step,
    curve_colours,
    legend_labels,
    line_colours,
)

STEPS = (
    Step("cut", cuts=((0.3, None),)),
    Step("subtract", degree=4),
    Step("smooth", window=31),
    Step("fit", advanced=True, function="A * sin(B * x)", start="B=314", on=False),
)


def state(**panel_kw):
    """A 2 x 1 layout: a data panel with a line with steps, and its FFT linked to it."""
    line = Line(
        run="r.005",
        x="B",
        x_fn="1/x",
        y="M006_AH",
        steps=STEPS,
        colour="#123456",
        label="mine",
    )
    data = Panel([line, Line(run="r.003")], **panel_kw)
    fft = Panel([l.copy() for l in data.lines], source=(0, 0))
    for l in fft.lines:
        l.steps += (Step("fft", pad=4),)
    links = {frozenset((data.id, fft.id)): Link(" ".join(SYNC))}
    return {(0, 0): data, (1, 0): fft}, links


def test_round_trip_through_json():
    panels, links = state(title="$B$ sweep", x_min=0.1)
    dumped = session.dump(panels, Layout.grid(2, 1), links)
    layout, loaded, loaded_links = session.load(json.loads(json.dumps(dumped)))
    assert layout == Layout.grid(2, 1)
    assert loaded[(0, 0)].lines[0].steps == STEPS
    assert loaded[(0, 0)].title == "$B$ sweep" and loaded[(0, 0)].x_min == 0.1
    assert loaded[(1, 0)].lines[0].steps[-1] == Step("fft", pad=4)
    assert loaded[(1, 0)].source == (0, 0)
    assert loaded_links == links
    # Undo compares dumps, so a loaded state must dump the same as the original.
    assert session.dump(loaded, layout, loaded_links) == dumped


def test_bad_values_are_dropped_not_fatal():
    panels, links = state()
    dumped = json.loads(json.dumps(session.dump(panels, Layout.grid(2, 1), links)))
    line = dumped["panels"]["0,0"]["lines"][0]
    line.update(width="thick", colour=None)
    line["steps"][2].update(method="wobble", window=-3)
    line["steps"][0].update(cut="sometimes")
    line["steps"] += [{"kind": "teleport"}, "not a step"]
    dumped["panels"]["0,0"]["legend"] = "centre stage"
    loaded = session.load(dumped)[1][(0, 0)]
    cut, _, smooth, _ = loaded.lines[0].steps
    assert (smooth.method, smooth.window, cut.cut) == ("savgol", 21, "remove")
    assert len(loaded.lines[0].steps) == 4 and loaded.legend == "auto"


def test_not_a_session():
    for bad in ({}, {"rows": 1}, [1, 2], {"format": session.FORMAT, "panels": {}}):
        with pytest.raises(ValueError):
            session.load(bad)
    with pytest.raises(ValueError, match="bigger"):
        session.load(
            {"format": session.FORMAT, "layout": {"counts": [7]}, "panels": {}}
        )


def test_older_formats_are_refused():
    panels, links = state()
    dumped = json.loads(json.dumps(session.dump(panels, Layout.grid(2, 1), links)))
    for fmt in (None, 9, 11):
        with pytest.raises(ValueError, match="this plotter's format"):
            session.load({**dumped, "format": fmt})


def test_duplicate_ids_are_made_distinct():
    panels, links = state()
    dumped = session.dump(panels, Layout.grid(2, 1), links)
    dumped["panels"]["1,0"]["id"] = dumped["panels"]["0,0"]["id"]
    loaded = session.load(dumped)[1]
    assert loaded[(0, 0)].id != loaded[(1, 0)].id


def test_legend_names_only_what_differs():
    base = ("r.005", "B", "x", "M006_AH", "y")
    cut = Step("cut", cuts=((1.0, 2.0),))
    lines = [Line(shown=base + ((),)), Line(shown=base + ((cut,),))]
    assert legend_labels(lines) == ["M006_AH", "M006_AH · without x 1–2"]
    lines[1].shown = ("r.003",) + base[1:] + ((),)
    assert legend_labels(lines) == ["run 005", "run 003"]
    lines[1].shown = base + ((Step("despike"),),)
    assert legend_labels(lines) == ["M006_AH", "M006_AH · despiked 5σ/21"]
    lines[1].shown = base + (
        (Step("fit"), Step("cut")),
    )  # a fit leaves the data; a cut
    assert legend_labels(lines) == ["M006_AH", "M006_AH"]  # without ranges does nothing


def test_line_colours_avoid_clashes():
    panel = Panel(
        [Line(y="M006_AH"), Line(y="M006_AH"), Line(y="M011_AH", colour="#000000")]
    )
    colours = line_colours(panel, {"M006": "#2a78d6", "M011": "#eb6834"})
    assert colours[0] == "#2a78d6" and colours[2] == "#000000"
    assert len(set(colours)) == 3


def test_line_colours_from_a_colour_map():
    lines = [Line(), Line(steps=(Step("fit"),)), Line()]
    colours, curves = curve_colours(Panel(lines), {}, "nanoanalysis_muted")
    # Every curve, a fit curve too, a colour of its own, from one end of the
    # map to the other: each line, then its fits.
    assert (
        colours[0] == "#2f95ac" and colours[2] == "#2fac55"
    )  # its ends: from its blue
    assert len(set(colours) | set(curves.values())) == 4 and list(curves) == [(1, 0)]
    lines[1].colour = "#000000"  # picked: keeps its place, so no other moves
    again, _ = curve_colours(Panel(lines), {}, "nanoanalysis_muted")
    assert again == [colours[0], "#000000", colours[2]]
    lines[1].steps = (Step("fit", colour="#ff0000"),)
    assert curve_colours(Panel(lines), {}, "viridis")[1] == {(1, 0): "#ff0000"}
    lines[1].steps += (Step("fft"),)  # no fit curves once the line is a spectrum
    assert curve_colours(Panel(lines), {}, "viridis")[1] == {}
    assert line_colours(Panel([Line()]), {}, "viridis") == [
        "#440154"
    ]  # one line: the start
    stepped = line_colours(Panel([Line() for _ in range(11)]), {}, "tab10")
    assert stepped[0] == stepped[10] == "#1f77b4" and len(set(stepped)) == 10
    # Classic: the sample's colour, as before; a fit curve an unused palette one.
    colours, curves = curve_colours(
        Panel([Line(y="M006_AH", steps=(Step("fit"),))]), {"M006": "#2a78d6"}
    )
    assert colours == ["#2a78d6"] and curves[(0, 0)] not in colours


def test_what_a_draw_found_and_the_chosen_step_are_not_saved():
    panels, links = state()
    line = panels[(0, 0)].lines[0]
    line.found, line.step = (None, (("A", 1.0),), None, None), 2
    dumped = json.loads(json.dumps(session.dump(panels, Layout.grid(2, 1), links)))
    saved = dumped["panels"]["0,0"]["lines"][0]
    assert "found" not in saved and "step" not in saved
    loaded = session.load(dumped)[1][(0, 0)].lines[0]
    assert loaded.found == () and loaded.step == 0 and line.copy().found == ()


def test_changing_x_clears_every_x_unit_setting():
    line = Line(
        steps=(Step("smooth", in_x=True, span=0.1), Step("cut", cuts=((1.0, 2.0),)))
    )
    line.clear_x_units()
    smooth, cut = line.steps
    assert (smooth.span, cut.cuts) == (None, ())
    assert cut.cutting is None and cut.cut == "remove"
