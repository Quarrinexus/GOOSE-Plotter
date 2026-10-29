"""Sessions and undo snapshots: the panels to plain data and back, old formats included."""

import json

import pytest

from goose_plotter import session
from goose_plotter.model import (SYNC, Line, Link, Panel, Step, curve_colours, legend_labels,
                                 line_colours)

STEPS = (Step("cut", cut="remove", cuts=((0.1, 0.2), (0.3, None))),
         Step("subtract", degree=4), Step("smooth", window=31),
         Step("fit", advanced=True, function="A * sin(B * x)", start="B=314", on=False))


def state(**panel_kw):
    """A 2 x 1 layout: a data panel with a line with steps, and its FFT linked to it."""
    line = Line(run="r.005", x="B", x_fn="1/x", y="M006_AH", steps=STEPS,
                colour="#123456", label="mine")
    data = Panel([line, Line(run="r.003")], **panel_kw)
    fft = Panel([l.copy() for l in data.lines], source=(0, 0), operation="fft", pad=4)
    links = {frozenset((data.id, fft.id)): Link(" ".join(SYNC))}
    return {(0, 0): data, (1, 0): fft}, links


def test_round_trip_through_json():
    panels, links = state(title="$B$ sweep", x_min=0.1)
    dumped = session.dump(panels, 2, 1, links)
    rows, cols, loaded, loaded_links = session.load(json.loads(json.dumps(dumped)))
    assert (rows, cols) == (2, 1)
    assert loaded[(0, 0)].lines[0].steps == STEPS
    assert loaded[(0, 0)].title == "$B$ sweep" and loaded[(0, 0)].x_min == 0.1
    assert loaded[(1, 0)].operation == "fft" and loaded[(1, 0)].source == (0, 0)
    assert loaded_links == links
    # Undo compares dumps, so a loaded state must dump the same as the original.
    assert session.dump(loaded, rows, cols, loaded_links) == dumped


def test_bad_values_are_dropped_not_fatal():
    panels, links = state()
    dumped = json.loads(json.dumps(session.dump(panels, 2, 1, links)))
    line = dumped["panels"]["0,0"]["lines"][0]
    line.update(width="thick", colour=None)
    line["steps"][2].update(method="wobble", window=-3)
    line["steps"][0].update(cut="sometimes")
    line["steps"] += [{"kind": "teleport"}, "not a step"]
    dumped["panels"]["0,0"]["legend"] = "centre stage"
    loaded = session.load(dumped)[2][(0, 0)]
    cut, _, smooth, _ = loaded.lines[0].steps
    assert (smooth.method, smooth.window, cut.cut) == ("savgol", 21, "keep")
    assert len(loaded.lines[0].steps) == 4 and loaded.legend == "auto"


def test_not_a_session():
    for bad in ({}, {"rows": 1}, {"rows": "x", "cols": 1, "panels": {}}, [1, 2]):
        with pytest.raises(ValueError):
            session.load(bad)
    with pytest.raises(ValueError, match="bigger"):
        session.load({"rows": 99, "cols": 1, "panels": {}})


def test_duplicate_ids_are_made_distinct():
    panels, links = state()
    dumped = session.dump(panels, 2, 1, links)
    dumped["panels"]["1,0"]["id"] = dumped["panels"]["0,0"]["id"]
    loaded = session.load(dumped)[2]
    assert loaded[(0, 0)].id != loaded[(1, 0)].id


def old(fmt, **line):
    """A session from before format 10: a data panel whose line has `line`'s
    old fields, and its FFT, linked sharing everything there was then."""
    words = "run x x_fn y y_fn colour smoothing background style cut despike"
    return {"format": fmt, "rows": 2, "cols": 1, "links": [{"a": "d", "b": "f", "sync": words}],
            "panels": {"0,0": {"id": "d", "lines": [{"run": "r.005", **line}]},
                       "1,0": {"id": "f", "operation": "fft", "source": "0,0",
                               "lines": [{"run": "r.005", **line}]}}}


def test_format_5_single_cut_becomes_one_range():
    loaded = session.load(old(5, cut="keep", cut_from=0.5, cut_to=0.2))[2][(0, 0)].lines[0]
    assert loaded.steps == (Step("cut", cut="keep", cuts=((0.2, 0.5),)),)


@pytest.mark.parametrize("cut, cuts, wanted", [
    ("", [], ((0.2, 0.5),)),
    ("keep", [[0.1, 0.3], [0.4, None]], ((0.2, 0.3), (0.4, 0.5))),
    ("remove", [[0.3, 0.4]], ((0.2, 0.3), (0.4, 0.5))),
])
def test_format_6_fit_range_becomes_a_cut(cut, cuts, wanted):
    loaded = session.load(old(6, background="subtract", fit_from=0.5, fit_to=0.2, cut=cut,
                              cuts=cuts))[2][(0, 0)].lines[0]
    assert loaded.steps == (Step("cut", cut="keep", cuts=wanted), Step("subtract"))


def test_format_6_fit_range_is_ignored_with_the_background_off():
    loaded = session.load(old(6, background="", fit_from=0.2, fit_to=0.5, cut="remove",
                              cuts=[[1, 2]]))[2][(0, 0)].lines[0]
    assert loaded.steps == (Step("cut", cut="remove", cuts=((1.0, 2.0),)),)


def test_before_format_10_settings_become_steps_in_the_old_order():
    line = dict(cut="keep", cuts=[[0.1, 0.2]], despike=True, despike_window=31,
                despike_threshold=4.5, despiked=7, background="subtract", advanced=True,
                fit_function="A + B * x", fit_start="B=2", fit_values=[["A", 1]],
                smooth="median", window=41, in_x=True, span=0.01)
    loaded = session.load(old(9, **line))[2][(0, 0)].lines[0]
    assert loaded.steps == (Step("cut", cuts=((0.1, 0.2),)),
                            Step("despike", spike_window=31, threshold=4.5),
                            Step("subtract", advanced=True, function="A + B * x", start="B=2"),
                            Step("smooth", method="median", window=41, in_x=True, span=0.01))
    assert loaded.found == ()
    # Show fit: a fit curve where the background was; nothing on: no steps.
    shown = session.load(old(8, background="fit", degree=3))[2][(0, 0)].lines[0]
    assert shown.steps == (Step("fit", degree=3),)
    assert session.load(old(8, cut="", smooth=""))[2][(0, 0)].lines[0].steps == ()


def test_before_format_5_derived_links_share_the_cut():
    dumped = old(4)
    dumped["links"][0]["sync"] = "run x x_fn y y_fn colour style"  # nothing of the steps
    loaded_links = session.load(dumped)[3]
    assert all("steps" in link.synced for link in loaded_links.values())


def test_before_format_5_partial_links_between_data_panels_stay_partial():
    a, b = Panel([Line(run="r")]), Panel([Line(run="r")])
    dumped = session.dump({(0, 0): a, (0, 1): b}, 1, 2,
                          {frozenset((a.id, b.id)): Link("run x y")})
    dumped["format"] = 4
    (link,) = session.load(dumped)[3].values()
    assert link.synced == {"run", "x", "y"}


def test_version_1_layout():
    """Derived panels had a source and no lines; they get copies, linked."""
    old = {"rows": 2, "cols": 1, "panels": {
        "0,0": {"lines": [{"run": "r.005", "y": "M006_AH"}], "link_group": None},
        "1,0": {"source": "0,0"}}}
    rows, cols, panels, links = session.load(old)
    fft = panels[(1, 0)]
    assert fft.operation == "fft" and fft.lines[0].y == "M006_AH"
    (link,) = links.values()
    assert link.synced == set(SYNC)


def test_format_2_blank_text_was_automatic():
    old = {"format": 2, "rows": 1, "cols": 1,
           "panels": {"0,0": {"title": "", "lines": [{"label": ""}]}}}
    panel = session.load(old)[2][(0, 0)]
    assert panel.title is None and panel.lines[0].label is None


def test_format_8_auto_style_is_solid_or_dashed_for_a_shown_fit():
    old = {"format": 8, "rows": 1, "cols": 1,
           "panels": {"0,0": {"lines": [{"style": "auto"},
                                        {"style": "auto", "background": "fit"},
                                        {"style": ":", "background": "fit"}]}}}
    lines = session.load(old)[2][(0, 0)].lines
    assert [l.style for l in lines] == ["-", "--", ":"]


# --- model ------------------------------------------------------------------

def test_legend_names_only_what_differs():
    base = ("r.005", "B", "x", "M006_AH", "y")
    cut = Step("cut", cuts=((1.0, 2.0),))
    lines = [Line(shown=base + ((),)), Line(shown=base + ((cut,),))]
    assert legend_labels(lines) == ["M006_AH", "M006_AH · x 1–2"]
    lines[1].shown = ("r.003",) + base[1:] + ((),)
    assert legend_labels(lines) == ["run 005", "run 003"]
    lines[1].shown = base + ((Step("despike"),),)
    assert legend_labels(lines) == ["M006_AH", "M006_AH · despiked 5σ/21"]
    lines[1].shown = base + ((Step("fit"), Step("cut")),)  # a fit leaves the data; a cut
    assert legend_labels(lines) == ["M006_AH", "M006_AH"]  # without ranges does nothing


def test_line_colours_avoid_clashes():
    panel = Panel([Line(y="M006_AH"), Line(y="M006_AH"), Line(y="M011_AH", colour="#000000")])
    colours = line_colours(panel, {"M006": "#2a78d6", "M011": "#eb6834"})
    assert colours[0] == "#2a78d6" and colours[2] == "#000000"
    assert len(set(colours)) == 3


def test_line_colours_from_a_colour_map():
    lines = [Line(), Line(steps=(Step("fit"),)), Line()]
    colours, curves = curve_colours(Panel(lines), {}, "nanoanalysis_muted")
    # Every curve, a fit curve too, a colour of its own, from one end of the
    # map to the other: each line, then its fits.
    assert colours[0] == "#2f95ac" and colours[2] == "#2fac55"  # its ends: from its blue
    assert len(set(colours) | set(curves.values())) == 4 and list(curves) == [(1, 0)]
    lines[1].colour = "#000000"  # picked: keeps its place, so no other moves
    again, _ = curve_colours(Panel(lines), {}, "nanoanalysis_muted")
    assert again == [colours[0], "#000000", colours[2]]
    lines[1].steps = (Step("fit", colour="#ff0000"),)
    assert curve_colours(Panel(lines), {}, "viridis")[1] == {(1, 0): "#ff0000"}
    derived = Panel(lines, operation="fft")  # no fit curves on an FFT
    assert curve_colours(derived, {}, "viridis")[1] == {}
    assert line_colours(Panel([Line()]), {}, "viridis") == ["#440154"]  # one line: the start
    stepped = line_colours(Panel([Line() for _ in range(11)]), {}, "tab10")
    assert stepped[0] == stepped[10] == "#1f77b4" and len(set(stepped)) == 10
    # Classic: the sample's colour, as before; a fit curve an unused palette one.
    colours, curves = curve_colours(Panel([Line(y="M006_AH", steps=(Step("fit"),))]),
                                    {"M006": "#2a78d6"})
    assert colours == ["#2a78d6"] and curves[(0, 0)] not in colours


def test_what_a_draw_found_and_the_chosen_step_are_not_saved():
    panels, links = state()
    line = panels[(0, 0)].lines[0]
    line.found, line.step = (None, (("A", 1.0),), None, None), 2
    dumped = json.loads(json.dumps(session.dump(panels, 2, 1, links)))
    saved = dumped["panels"]["0,0"]["lines"][0]
    assert "found" not in saved and "step" not in saved
    loaded = session.load(dumped)[2][(0, 0)].lines[0]
    assert loaded.found == () and loaded.step == 0 and line.copy().found == ()




def test_old_link_keys_for_steps_share_steps():
    dumped = old(8)
    dumped["links"][0]["sync"] = "run x smoothing"
    assert session.load(dumped)[3][frozenset("df")].synced == {"run", "x", "steps"}
    dumped = old(7)
    del dumped["panels"]["1,0"]["source"]  # not known as derived: judged by what it shares
    dumped["links"][0]["sync"] = "run x x_fn y y_fn colour style"  # all but the steps
    assert "steps" in session.load(dumped)[3][frozenset("df")].synced
    dumped["links"][0]["sync"] = "run x y"  # partial links stay partial
    assert "steps" not in session.load(dumped)[3][frozenset("df")].synced


def test_changing_x_clears_every_x_unit_setting():
    line = Line(steps=(Step("smooth", in_x=True, span=0.1), Step("cut", cuts=((1.0, 2.0),))))
    line.clear_x_units()
    smooth, cut = line.steps
    assert (smooth.span, cut.cuts) == (None, ())
    assert cut.cutting is None and cut.cut == "keep"  # the mode stays for new ranges