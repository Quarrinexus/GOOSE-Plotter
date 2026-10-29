"""The window, driven as a user would, on the made-up data in conftest."""

import json

import numpy as np
import pytest

from goose_plotter import background, session, splicing
from conftest import B_HIGH, B_LOW, F, add_step, cutting, drawn, plot, steps_of

def test_plots_the_profile_default_axes(app):
    plot(app)
    line = app.panel.line
    assert (line.x, line.y) == ("Norminal_FIeld", "M006_AH")
    x, _ = drawn(app)
    assert x.max() == pytest.approx(B_HIGH) and x.min() == pytest.approx(B_LOW)
    assert app.axes[(0, 0)].get_title() == "run 005"
    assert app.filename.get() == "run_005_M006_AH_vs_Norminal_FIeld.png"
    assert app.error_label["text"] == ""


def test_a_bad_function_is_reported_under_the_axes(app):
    plot(app)
    app.y_fn.set("y +")
    app.apply_controls()
    assert app.error_label["text"].startswith("Function error")
    assert not app.axes[(0, 0)].lines


def test_fft_panel_finds_the_oscillation(app):
    plot(app, x_fn="1/x")
    add_step(app, "subtract", degree=10)  # 5, the default, leaves too much of the 1/x background
    app.new_derived_panel("fft")
    frequency, amplitude = drawn(app, (1, 0))
    assert frequency[np.argmax(amplitude)] == pytest.approx(F, rel=0.05)
    assert app.panels[(1, 0)].source == (0, 0)


def test_splicing_keeps_and_removes_several_ranges(app):
    plot(app)
    add_step(app, "cut")
    app.cut_mode.set(splicing.MODES["remove"])
    app.apply_controls()
    for start, end in (("10", "12"), ("20", "22")):
        app.cut_from.set(start)
        app.cut_to.set(end)
        app.add_cut()
    x, _ = drawn(app)
    finite = x[np.isfinite(x)]
    assert not np.any((finite >= 10) & (finite <= 12) | (finite >= 20) & (finite <= 22))
    assert np.isnan(x).any()  # gaps, not joins
    assert app.cut_list.get(0, "end") == ("x 10 to 12", "x 20 to 22")
    # One mode for every range.
    app.cut_mode.set(splicing.MODES["keep"])
    app.apply_controls()
    x, _ = drawn(app)
    assert np.all((x >= 10) & (x <= 12) | (x >= 20) & (x <= 22))
    assert app.filename.get().endswith("_x10-12_20-22.png")


def test_a_kept_range_sets_the_fft_resolution(app):
    plot(app, x_fn="1/x")
    add_step(app, "cut")  # keeps its ranges to start with
    app.cut_from.set("0.05")
    app.cut_to.set("0.15")
    app.add_cut()
    assert cutting(app.panel.line) == ("keep", ((0.05, 0.15),))
    app.new_derived_panel("fft")
    assert app.axes[(1, 0)].get_title().endswith("ΔF 10")


def test_splicing_an_fft_cuts_its_spectrum_in_f(app):
    plot(app, x_fn="1/x")
    add_step(app, "subtract")
    app.new_derived_panel("fft")
    fft = app.panels[(1, 0)]
    assert app.step_list.list.get("end").startswith("Cut spectrum (F)")  # after the steps
    app.choose_step(len(fft.line.steps))
    assert app.editors["cut"].winfo_manager() and app.cut_mode.get() == splicing.MODES[""]
    app.axes[(1, 0)].set_xlim(0, 5000)  # zoomed out: the cut fits the view to it
    app.cut_from.set("30")
    app.cut_to.set("70")
    app.add_cut()  # off, so it switches to keeping the range
    frequency, amplitude = drawn(app, (1, 0))
    assert frequency.min() >= 30 and frequency.max() <= 70
    assert frequency[np.argmax(amplitude)] == pytest.approx(F, rel=0.05)
    low, high = app.axes[(1, 0)].get_xlim()
    assert low > 25 and high < 75
    assert fft.cuts == ((30.0, 70.0),) and fft.cut == "keep"
    assert all(not steps_of(l, "cut") for p in app.panels.values() for l in p.lines)  # data uncut
    saved = json.loads(json.dumps(session.dump(app.panels, app.rows, app.cols, app.links)))
    _, _, panels, _ = session.load(saved)  # as a session file gives it back
    assert (panels[(1, 0)].cut, panels[(1, 0)].cuts) == ("keep", ((30.0, 70.0),))
    app.pick_range("cut")  # picking works on an FFT panel
    assert app.picker is not None
    app.stop_picking()
    app.selected = (0, 0)  # a data panel has no spectrum row
    app._load_controls()
    assert app.step_list.list.size() == 1 and app.editors["fit"].winfo_manager()
    app.x_fn.set("x")  # F in 1/T, not T: the spectrum's cut goes
    app.apply_controls()
    assert fft.cuts == ()


def test_editing_picking_and_deleting_ranges(app):
    plot(app)
    add_step(app, "cut")
    app.cut_from.set("5")
    app.cut_to.set("6")
    app.add_cut()
    app.cut_from.set("5.5")
    app.add_cut(replace=True)  # Enter changes the chosen range
    assert app.panel.line.chosen.cuts == ((5.5, 6.0),)
    app.pick_range("cut")
    app._use_range(20.0, 25.0)
    assert app.panel.line.chosen.cuts == ((5.5, 6.0), (20.0, 25.0))
    app.cut_index = 0
    app.delete_cut()
    assert app.panel.line.chosen.cuts == ((20.0, 25.0),)
    app.cut_from.set("")
    app.cut_to.set("")
    app.add_cut()
    assert "at least one end" in app.status["text"]


def test_a_cut_in_the_old_x_is_cleared(app):
    plot(app)
    add_step(app, "cut")
    app.cut_from.set("5")
    app.cut_to.set("9")
    app.add_cut()
    add_step(app, "smooth", in_x=True, span=0.5)
    app.x_fn.set("1/x")
    app.apply_controls()
    cut, smooth = app.panel.line.steps
    assert cut.cuts == () and smooth.span != 0.5  # estimated again, in 1/B
    app.choose_step(0)
    app.cut_from.set("0.1")
    app.cut_to.set("0.2")
    app.add_cut()
    app.swap()
    assert app.panel.line.steps[0].cuts == ()


def test_a_cut_range_with_no_points_explains_itself(app):
    plot(app, x_fn="1/x")
    add_step(app, "cut")
    app.cut_from.set("5")
    app.cut_to.set("9")  # in T, while x is 1/B
    app.add_cut()
    assert app.error_label["text"].startswith("Splicing error: there are no points")


def test_links_share_what_they_tick(app):
    plot(app)
    app.set_layout(1, 2)
    app.selected = (0, 0)
    app._link((0, 0), (0, 1))
    app.selected = (0, 0)
    app._load_controls()
    add_step(app, "smooth", method="mean")
    other = app.panels[(0, 1)].lines[0]
    assert other.steps == ()  # Steps isn't ticked to start with
    app.sync_vars["steps"].set(True)
    app.set_sync("steps")
    assert other.steps == app.panel.line.steps  # ticking sends them across
    add_step(app, "cut")
    app.cut_from.set("5")
    app.cut_to.set("9")
    app.add_cut()
    assert cutting(other) == ("keep", ((5.0, 9.0),))
    app.add_line()
    assert len(app.panels[(0, 1)].lines) == 2  # lines stay paired


def test_derived_panels_follow_every_setting(app):
    plot(app)
    app.new_derived_panel("d1")
    app.selected = (0, 0)
    app._load_controls()
    add_step(app, "cut")
    app.cut_from.set("5")
    app.cut_to.set("9")
    app.add_cut()
    add_step(app, "smooth", method="median")
    derived = app.panels[(1, 0)].lines[0]
    assert cutting(derived) == ("keep", ((5.0, 9.0),))
    assert [s.method for s in steps_of(derived, "smooth")] == ["median"]
    x, _ = drawn(app, (1, 0))
    assert x.min() >= 5 and x.max() <= 9


def buttons(widget):
    """The texts of the ttk buttons shown in `widget`, top to bottom."""
    found = []
    for child in widget.winfo_children():
        if child.winfo_manager() and child.winfo_class() == "TButton":
            found.append(child["text"])
        elif child.winfo_manager():
            found += buttons(child)
    return found


def test_this_panel_turns_into_its_fft_and_undo_turns_it_back(app):
    plot(app, x_fn="1/x")
    add_step(app, "subtract", degree=10)  # the 1/x background needs more than 5
    panel = app.panel
    panel.x_min, panel.title = 0.05, "My data"
    app.in_place("fft")
    assert len(app.panels) == 1 and app.panel is panel and panel.operation == "fft"
    assert panel.x_min is None and panel.title is None  # in T, not frequency
    frequency, amplitude = drawn(app)
    assert frequency[np.argmax(amplitude)] == pytest.approx(F, rel=0.05)
    derive = buttons(app.tabs["Derive"])
    assert derive.count("Undo") == 1 and "Back to data" not in derive
    # The Derivative section's This panel swaps it for a derivative; Undo still
    # goes back to the data.
    app.in_place("d1")
    assert panel.operation == "d1" and buttons(app.tabs["Derive"]).count("Undo") == 1
    app.undo_in_place()
    assert panel.operation == "" and panel.data_view is None
    assert (panel.x_min, panel.title) == (0.05, "My data")
    assert "Undo" not in buttons(app.tabs["Derive"])


def test_this_panel_is_only_for_data_panels(app):
    plot(app)
    app.new_derived_panel("fft")
    app.in_place("d1")
    assert app.panel.operation == "fft" and app.panel.data_view is None
    assert "Back to data" in buttons(app.tabs["Derive"])


def test_this_panel_survives_undo_and_sessions(app, tmp_path):
    plot(app)
    app.panel.y_max = 3.0
    app.in_place("d2")
    app.derivative_window.set("101")
    app.apply_derivative()
    app.undo()  # the window change, not the turning
    assert app.panel.operation == "d2" and app.panel.derivative_window == 51
    assert buttons(app.tabs["Derive"]).count("Undo") == 1
    path = tmp_path / "s.json"
    app.save_session(path)
    app.open_session(path)
    assert app.panel.data_view == (None, None, None, 3.0, None, None, None)
    app.undo_in_place()
    assert app.panel.y_max == 3.0 and not app.panel.derived


def test_undo_forgets_ranges_in_an_x_that_changed(app):
    plot(app)
    app.panel.x_min, app.panel.y_max, app.panel.x_label = 5.0, 3.0, "B"
    app.in_place("fft")
    app.x_fn.set("1/x")  # F needs 1/B; 5 T means nothing there
    app.apply_controls()
    app.undo_in_place()
    assert (app.panel.x_min, app.panel.y_max) == (None, 3.0)
    app.panel.x_min = 0.1
    app.in_place("d1")
    app.swap()
    app.undo_in_place()
    assert (app.panel.x_max, app.panel.y_min, app.panel.y_max) == (3.0, 0.1, None)
    assert app.panel.y_label == "B"


def click(app, x, y, cell=None):
    """A left click on the plot at data point (x, y) of `cell`'s axes."""
    from matplotlib.backend_bases import MouseEvent
    ax = app.axes[cell or app.selected]
    px, py = ax.transData.transform((x, y))
    app._on_click(MouseEvent("button_press_event", app.canvas, px, py, button=1))


def test_measure_reads_a_region_without_touching_the_plot(app):
    plot(app)
    ax = app.axes[(0, 0)]
    ax.set_xlim(8, 20)  # zoomed in: measuring mustn't undo it
    app.region_from.set("10")
    app.region_to.set("15")
    app.set_region()
    x, y = drawn(app)
    inside = (x >= 10) & (x <= 15)
    found = app.measured[(0, 0)]["extremes"]
    assert found["max"][1] == pytest.approx(np.nanmax(y[inside]))
    assert found["min"][1] == pytest.approx(np.nanmin(y[inside]))
    assert found["points"] == inside.sum()
    assert app.readout["Max"][1]["text"].startswith("y ")
    app.mark_var.set(True)
    app.apply_measure()
    ax = app.axes[(0, 0)]
    assert ax.get_xlim() == (8, 20)
    assert len(ax.lines) == 1 and len(ax.collections) == 2  # max and min, not lines
    # A new x function: 10 to 15 T means nothing in 1/B.
    app.x_fn.set("1/x")
    app.apply_controls()
    assert app.panel.region == () and app.region_from.get() == ""


def test_measure_reads_points_and_their_difference(app):
    plot(app, x_fn="1/x")
    x, y = drawn(app)
    app.read_points()
    for row in (1000, 3000):
        click(app, x[row], y[row])
    assert app.point_pick  # still reading, until Esc
    assert app.panel.points == ((x[1000], y[1000]), (x[3000], y[3000]))
    assert app.point_labels[2]["text"].startswith("dx ")
    assert f"1/dx {1 / (x[3000] - x[1000]):.6g}" in app.point_labels[2]["text"]
    click(app, x[5000], y[5000])  # a third replaces the first
    assert app.panel.points[0] == (x[3000], y[3000])
    app.toolbar.zoom()  # zooming in to pick closer: its clicks aren't points
    click(app, x[4000], y[4000])
    app.toolbar.zoom()
    assert app.panel.points[1] == (x[5000], y[5000])
    app.stop_picking()
    assert not app.point_pick
    app.undo()
    assert app.panel.points[1] == (x[3000], y[3000])


def test_measure_lists_an_ffts_peaks(app, tmp_path):
    plot(app, x_fn="1/x")
    add_step(app, "subtract", degree=10)  # the 1/x background needs more than 5
    app.in_place("fft")
    read = app.measured[(0, 0)]
    assert read["peaks"][0][0] == pytest.approx(F, rel=0.02)
    assert read["extremes"]["max"][0] == pytest.approx(F, rel=0.02)
    assert app.peak_list.size() == len(read["peaks"]) >= 1
    app.mark_var.set(True)
    app.apply_measure()
    assert len(app.axes[(0, 0)].lines) == 1
    # Saved figures leave the marks out unless the panel keeps them.
    seen = []
    app.fig.savefig = lambda *a, **k: seen.append([m.get_visible() for m in app.marks[(0, 0)]])
    app.save()
    app.marks_saved_var.set(True)
    app.apply_measure()
    app.save()
    assert not any(seen[0]) and all(seen[1])
    assert all(m.get_visible() for m in app.marks[(0, 0)])
    path = tmp_path / "m.csv"
    app.export_measurements(path)
    text = path.read_text()
    assert text.startswith("what,x,y\n") and "\npeak 1," in text


def test_measure_follows_the_selected_line(app):
    plot(app)
    app.add_line()
    app.y.set(next(v for v in app.y.box["values"] if "M011" in v))
    app.apply_controls()
    app.mark_var.set(True)
    app.apply_measure()
    ax = app.axes[(0, 0)]
    assert len(ax.lines) == 2 and len(ax.get_legend().get_texts()) == 2
    for index in (0, 1):
        app._set_selected_line((0, 0), index)
        app._select_panel((0, 0))
        _, y = drawn(app, index=index)
        assert app.measured[(0, 0)]["extremes"]["max"][1] == np.nanmax(y)


def test_measure_settings_are_saved_in_sessions(app, tmp_path):
    plot(app)
    app.set_region((5, 9))
    x, y = drawn(app)
    app.read_points()
    click(app, x[100], y[100])
    path = tmp_path / "s.json"
    app.save_session(path)
    app.set_region(())
    app.open_session(path)
    assert app.panel.region == (5.0, 9.0) and app.panel.points == ((x[100], y[100]),)


def test_plots_in_tabs_keep_their_own_panels(app):
    plot(app)
    app.set_region((5, 9))
    app.new_plot()
    assert len(app.plots) == 2 and app.panel.line.run == ""  # a fresh, empty plot
    assert app.filename.get() == ""
    plot(app, run="Cambridge_Sep_26.005", x_fn="1/x")
    app.set_layout(1, 2)
    app.switch_plot(1)
    assert (app.rows, app.cols) == (1, 1) and app.panel.region == (5.0, 9.0)
    assert app.panel.line.x_fn == "x" and app.filename.get().startswith("run_005")
    app.undo()  # plot 1's own last step: the region
    assert app.panel.region == ()
    app.switch_plot(2)
    assert (app.rows, app.cols) == (1, 2) and app.panel.line.x_fn == "1/x"
    names = [text for _, text in app.plot_strip.tabs]
    assert names == ["Plot 1", "Plot 2"] and app.plot_strip.chosen == 2
    app.close_plot(2)  # asks (answered yes); shows the one before
    assert len(app.plots) == 1 and app.plot_strip.chosen == 1
    assert app.panel.line.run and not app.plot_strip.close_spans  # the last has no ×


def test_clicking_the_plot_tabs(app):
    plot(app)
    app.deiconify()  # hidden, the plot's side is never laid out, so nothing gets clicks
    app.update()
    strip = app.plot_strip
    left, right = strip.add_span
    strip.event_generate("<Button-1>", x=(left + right) // 2, y=10)
    assert len(app.plots) == 2 and strip.chosen == 2
    app.update()
    left, right = strip.spans[1]
    strip.event_generate("<Button-1>", x=left + 10, y=10)
    assert strip.chosen == 1 and app.panel.line.run
    left, right = strip.close_spans[2]
    strip.event_generate("<Button-1>", x=(left + right) // 2, y=10)
    assert len(app.plots) == 1


def test_a_change_goes_down_a_chain_of_links(app):
    """Data on top, linked to its background subtracted below, and that's FFT
    under it: a new y on top reaches the FFT, through both links."""
    plot(app, x_fn="1/x")
    app.set_layout(2, 1)  # panel 2 starts as a copy of 1
    app._link((1, 0), (0, 0))  # linked 1-2, sharing the axes, not the steps
    app.selected = (1, 0)
    app._load_controls()
    add_step(app, "subtract")
    app.new_derived_panel("fft")  # panel 3, the FFT of 2
    fft = app.panels[(2, 0)]
    app.selected = (0, 0)
    app._load_controls()
    app.y.set(next(v for v in app.y.box["values"] if "M011" in v))
    app.apply_controls()
    assert app.panels[(1, 0)].line.y == "M011_AH"
    assert fft.line.y == "M011_AH" and [s.kind for s in fft.line.steps] == ["subtract"]
    assert app.panels[(0, 0)].line.steps == ()  # 2-3 shares them, 1-2 doesn't


def test_undo_one_step(app):
    plot(app)
    add_step(app, "cut")
    app.cut_from.set("5")
    app.cut_to.set("9")
    app.add_cut()
    app.cut_index = 0
    app.delete_cut()
    app.undo()
    assert cutting(app.panel.line) == ("keep", ((5.0, 9.0),))
    app.undo()
    assert app.status["text"] == "Nothing to undo."


def test_session_save_and_open(app, tmp_path):
    plot(app)
    add_step(app, "cut")
    app.cut_from.set("5")
    app.cut_to.set("9")
    app.add_cut()
    add_step(app, "fit", advanced=True, function="A + B * x")
    app.set_layout(1, 2)
    path = tmp_path / "s.json"
    app.save_session(path)
    saved = json.loads(path.read_text())
    assert saved["format"] == session.FORMAT and saved["cols"] == 2
    assert [s["kind"] for s in saved["panels"]["0,0"]["lines"][0]["steps"]] == ["cut", "fit"]
    app.set_layout(1, 1)
    app.remove_step()
    app.open_session(path)
    assert app.cols == 2
    line = app.panels[(0, 0)].lines[0]
    assert cutting(line) == ("keep", ((5.0, 9.0),))
    assert line.steps[1].fitting == ("fit", None, "A + B * x", "")


def test_save_figure(app, tmp_path):
    plot(app)
    app.save()
    assert (tmp_path / "output" / "run_005_M006_AH_vs_Norminal_FIeld.png").is_file()


def test_clicking_a_tab_shows_it(app):
    plot(app)
    app.update()
    strip = app.tab_strip
    assert max(right for _, right in strip.spans.values()) <= strip.winfo_width()  # all fit
    left, right = strip.spans["Derive"]
    strip.event_generate("<Button-1>", x=(left + right) // 2, y=10)
    assert app.tab.get() == "Derive"
    assert app.tabs["Derive"].winfo_manager() and not app.tabs["Process"].winfo_manager()


def test_tab_order(app):
    assert [name for name, _ in app.tab_strip.tabs] == ["Process", "Derive", "Linking", "Measure"]
    assert app.tab.get() == "Process"


def test_advanced_fitting_fits_the_typed_function(app):
    plot(app, x_fn="1/x")
    add_step(app, "subtract")
    assert entry_states(app) == ["disabled", "disabled"]  # greyed out until ticked
    step = lambda: app.panel.line.chosen
    shown_default = background.polynomial_text(step().degree)
    assert app.fit_function.get() == shown_default and app.fit_start.get() == ""
    app.advanced.set(True)
    app.apply_controls()
    app.fit_function.set("A + B / x")
    app.apply_controls()
    assert entry_states(app) == ["normal", "normal"]
    line = app.panel.line
    assert step().fitting == ("subtract", None, "A + B / x", "")
    assert [name for name, _ in line.found[0]] == ["A", "B"]
    assert dict(line.found[0])["B"] == pytest.approx(1e-3, rel=0.05)  # conftest's 0.001 B
    assert app.fit_result["text"].startswith("A = ")
    assert app.error_label["text"] == ""
    assert "_bg_A_+_B_-over-_x" in app.filename.get()
    app.fit_function.set("A * sin(y)")
    app.apply_controls()
    assert app.error_label["text"].startswith("Fit error: 'y'")
    app.advanced.set(False)  # the polynomial again, the function kept for later
    app.apply_controls()
    assert step().fitting == ("subtract", step().degree, "", "")
    assert step().function == "A * sin(y)"
    assert entry_states(app) == ["disabled", "disabled"] and not app.fit_result["text"]
    assert app.fit_function.get() == shown_default  # showing the polynomial, not the kept one
    app.degree.set("3")
    app.apply_controls()
    assert app.fit_function.get() == "A*x**3 + B*x**2 + C*x + D"
    assert step().function == "A * sin(y)"
    app.advanced.set(True)  # ticked again: the kept function is back
    app.apply_controls()
    assert app.fit_function.get() == "A * sin(y)"


def test_ticking_advanced_fitting_starts_from_the_polynomial(app):
    plot(app, x_fn="1/x")
    add_step(app, "subtract")
    x, polynomial = drawn(app)
    app.advanced.set(True)
    app.apply_controls()
    step = app.panel.line.chosen
    assert step.function == background.polynomial_text(step.degree)
    assert app.fit_function.get() == step.function
    _, typed = drawn(app)
    assert np.allclose(typed, polynomial, rtol=0, atol=1e-7)  # the oscillation is 1e-4
    app.fit_function.set("")  # cleared: it says what's missing
    app.apply_controls()
    assert app.error_label["text"].startswith("Fit error: type a function")
    assert "custom" in app.line_list.get(0) and "None" not in app.line_list.get(0)


def test_steps_are_done_in_their_order_and_can_be_moved(app):
    """Subtract then smooth is not smooth then subtract with a function that
    fits only part: moving a step changes what's drawn, and the order is kept."""
    plot(app, x_fn="1/x")
    add_step(app, "subtract", degree=1)
    add_step(app, "smooth", method="mean", window=301)
    _, first = drawn(app)
    app.move_step(1, 0)  # as a drag in the list does
    assert [s.kind for s in app.panel.line.steps] == ["smooth", "subtract"]
    assert app.panel.line.step == 0  # the moved step stays chosen
    assert app.step_list.list.get(0).startswith("1  Smooth")
    _, second = drawn(app)
    assert not np.allclose(first, second, equal_nan=True)
    app.choose_step(0)
    app.step_on.set(False)  # turned off: drawn as if it weren't there
    app.toggle_step()
    assert not app.panel.line.steps[0].on and "(off)" in app.step_list.list.get(0)
    _, off = drawn(app)
    app.remove_step()
    _, removed = drawn(app)
    assert np.allclose(off, removed, equal_nan=True)
    # Any kind, any number of times.
    add_step(app, "subtract", degree=3)
    assert [s.kind for s in app.panel.line.steps] == ["subtract", "subtract"]
    assert app.panel.line.step == 1 and app.step_title["text"] == "Step 2: Subtract"


def test_dragging_a_step_moves_it(app):
    plot(app)
    add_step(app, "cut")
    add_step(app, "smooth")
    app.update()
    rows = app.step_list.list
    first, second = rows.bbox(0), rows.bbox(1)
    rows.event_generate("<Button-1>", x=5, y=first[1] + 2)
    assert app.panel.line.step == 0
    rows.event_generate("<B1-Motion>", x=5, y=second[1] + 2)
    rows.event_generate("<ButtonRelease-1>", x=5, y=second[1] + 2)
    assert [s.kind for s in app.panel.line.steps] == ["smooth", "cut"]
    assert app.panel.line.step == 1


def test_a_fit_curve_is_drawn_over_the_line(app):
    """Subtract the background, then fit the oscillation left as a curve."""
    plot(app, x_fn="1/x")
    add_step(app, "subtract", advanced=True, function="A + B / x")  # conftest's background
    add_step(app, "fit")
    line = app.panel.line
    assert [s.kind for s in line.steps] == ["subtract", "fit"]
    # The polynomial to start with, drawn at once, as an extra artist.
    assert app.error_label["text"] == "" and len(app.curves[app.selected]) == 1
    assert app.fit_hint["text"].startswith("Draws its fit over the line")
    app.advanced.set(True)  # ticked: from that polynomial
    app.apply_controls()
    assert line.chosen.function == background.polynomial_text(line.chosen.degree)
    app.fit_function.set("A * sin(B * x + C)")
    app.fit_start.set("A=1e-4, B=314, C=0")
    app.apply_controls()
    assert app.error_label["text"] == ""
    values = dict(line.found[1])
    assert values["B"] == pytest.approx(2 * np.pi * F, rel=1e-3)
    assert abs(values["A"]) == pytest.approx(1e-4, rel=0.02)
    assert app.fit_result["text"].startswith("A = ")
    # The line is the subtracted data, unchanged by the fit; the curve follows it.
    x, y = drawn(app)
    (curve, key), = app.curves[app.selected].items()
    assert key == (0, 1) and np.allclose(curve.get_xdata(), x)
    assert np.nanmax(abs(curve.get_ydata() - y)) < 1e-5
    assert curve.get_linestyle() == "--" and curve.get_color() != app.line_list.itemcget(0, "foreground")
    assert len(app.axes[app.selected].lines) == 2 and len(app.artists[app.selected]) == 1
    legend = [t.get_text() for t in app.axes[app.selected].get_legend().get_texts()]
    assert legend[1] == "fit: A * sin(B * x + C)"
    # A bad function shows its error, but the data is still drawn.
    app.fit_function.set("A * sin(y)")
    app.apply_controls()
    assert app.error_label["text"].startswith("Fit error") and app.artists[app.selected]
    assert not app.curves[app.selected]
    # Not on an FFT of the line.
    app.fit_function.set("A * sin(B * x + C)")
    app.apply_controls()
    app.new_derived_panel("fft")
    assert not app.curves[(1, 0)] and len(app.axes[(1, 0)].lines) == 1


def test_fit_lines_and_old_settings_load_as_steps(app, tmp_path):
    """A format 9 session: a line's fixed settings, and a Fit as new line line."""
    old = {"goose_plotter_session": 8, "format": 9, "rows": 1, "cols": 1, "panels": {"0,0": {"lines": [
        {"run": "Cambridge_Sep_26.005", "x_fn": "1/x", "cut": "keep", "cuts": [[0.05, 0.2]],
         "despike": True, "despike_window": 31, "background": "subtract", "degree": 4,
         "smooth": "median", "window": 41},
        {"run": "Cambridge_Sep_26.005", "x_fn": "1/x", "background": "fit", "fits_drawn": True,
         "first_fit": ["subtract", 4, "", ""], "advanced": True,
         "fit_function": "A * sin(B * x + C)", "fit_start": "B=314", "smooth": "mean"}]}},
        "links": []}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    app.open_session(path)
    data, fit = app.panel.lines
    assert [s.kind for s in data.steps] == ["cut", "despike", "subtract", "smooth"]
    assert data.steps[0].cutting == ("keep", ((0.05, 0.2),)) and data.steps[1].spike_window == 31
    assert data.steps[2].fitting == ("subtract", 4, "", "") and data.steps[3].method == "median"
    assert [s.kind for s in fit.steps] == ["subtract", "smooth", "fit"]
    assert fit.steps[2].fitting == ("fit", None, "A * sin(B * x + C)", "B=314")
    assert app.curves[(0, 0)] and app.error_label["text"] == ""


def test_colour_maps_colour_every_panel_and_are_remembered(app, settings_file):
    from goose_plotter import colourmaps
    plot(app)
    app.add_line()
    app.set_layout(1, 2)
    app.selected = (0, 1)
    plot(app)
    assert app.colour_map == colourmaps.DEFAULT == "nanoanalysis_muted"
    assert app.line_list.itemcget(0, "foreground") == "#2f95ac"  # the map's start, its blue
    app.open_style()
    popup = app.style_popup
    assert popup.colour_map.get() == "nanoanalysis_muted"
    assert popup.map_menu.index("end") + 1 == len(colourmaps.MAPS)
    app.axes[(0, 0)].set_xlim(10, 20)  # a zoom, which recolouring keeps
    popup.map_menu.invoke(list(colourmaps.MAPS).index("viridis"))
    assert app.colour_map == "viridis" and popup.colour_map.get() == "viridis"
    assert json.loads(settings_file.read_text())["colour_map"] == "viridis"
    first = [a.get_color() for a, _ in sorted(app.artists[(0, 0)].items(), key=lambda kv: kv[1])]
    assert first == ["#440154", "#fde725"]  # both ends of viridis
    assert app.artists[(0, 1)] and next(iter(app.artists[(0, 1)])).get_color() == "#440154"
    assert app.axes[(0, 0)].get_xlim() == (10, 20)
    popup.destroy()


def test_despike_turns_spikes_into_gaps(app):
    plot(app)
    df = app.frames[app.panel.line.run]
    column = next(c for c in df.columns if c.startswith("M006_AH"))
    df.loc[df.index[[1000, 3000]], column] += 0.05
    app.cache.clear()
    add_step(app, "despike")
    line = app.panel.line
    _, y = drawn(app)
    assert np.isnan(y[[1000, 3000]]).all() and np.isfinite(y).sum() == len(y) - 2
    assert line.found == (2,) and app.despike_found["text"] == "2 points removed"
    assert app.filename.get().endswith("_ds5-21.png")
    assert "despiked" in app.line_list.get(0)
    app.despike_threshold.set("0")  # refused: the last good one stays
    app.apply_controls()
    assert line.chosen.threshold == 5
    app.new_derived_panel("fft")  # its line follows the data panel's steps
    assert app.panels[(1, 0)].lines[0].steps[0].despiking == (21, 5.0)
    app.selected = (0, 0)
    app._load_controls()
    app.step_on.set(False)
    app.toggle_step()
    assert np.isfinite(drawn(app)[1]).all()
    assert not app.panels[(1, 0)].lines[0].steps[0].on


def spinbox_states(app):
    """The states of the Despike window and threshold boxes."""
    found = []

    def walk(widget):
        for child in widget.winfo_children():
            if child.winfo_class() == "TSpinbox" and str(child["textvariable"]) in (
                    str(app.despike_window), str(app.despike_threshold)):
                found.append(str(child["state"]))
            walk(child)
    walk(app.tabs["Process"])
    return found


def entry_states(app):
    """The states of the Background's function and start-value boxes."""
    found = []

    def walk(widget):
        for child in widget.winfo_children():
            if child.winfo_class() == "TEntry" and str(child["textvariable"]) in (
                    str(app.fit_function), str(app.fit_start)):
                found.append(str(child["state"]))
            walk(child)
    walk(app.tabs["Process"])
    return found


def test_advanced_fitting_goes_along_the_chain_to_the_fft(app):
    """Data, a copy with the background subtracted, and that copy's FFT."""
    plot(app, x_fn="1/x")
    app.set_layout(2, 1)
    app.selected = (1, 0)
    app._load_controls()
    plot(app, x_fn="1/x")
    app._link((0, 0), (1, 0))
    app.selected = (1, 0)
    app._load_controls()
    add_step(app, "subtract")
    app.new_derived_panel("fft")
    fft = next(p for p in app.panels.values() if p.operation == "fft")
    app.selected = (1, 0)
    app._load_controls()
    app.advanced.set(True)
    app.apply_controls()
    app.fit_function.set("A + B / x")  # conftest's background, exactly
    app.apply_controls()
    assert fft.lines[0].steps == app.panels[(1, 0)].lines[0].steps
    assert fft.lines[0].found == app.panels[(1, 0)].lines[0].found
    assert app.panels[(0, 0)].lines[0].steps == ()  # Steps isn't ticked on 1-2
    cell = next(c for c, p in app.panels.items() if p is fft)
    frequency, amplitude = drawn(app, cell)
    assert frequency[np.argmax(amplitude)] == pytest.approx(F, rel=0.05)


def test_only_a_cut_step_shows_the_cut_boxes(app):
    plot(app)
    assert app.no_steps.winfo_manager() and "Pick" not in buttons(app.tabs["Process"])
    add_step(app, "smooth")
    assert "Pick" not in buttons(app.tabs["Process"]) and not app.no_steps.winfo_manager()
    add_step(app, "cut")
    assert "Pick" in buttons(app.tabs["Process"])


@pytest.mark.parametrize("language", ["en", "zh"])
def test_no_tab_widens_the_controls_column(app):
    """Every tab leaves the column as wide as the controls above the tabs make
    it, so switching tabs never pushes the plot over."""
    plot(app)
    widths = {}
    for tab in app.tabs:
        app.tab.set(tab)
        app._show_tab()
        app.update_idletasks()
        widths[tab] = app.side_inner.winfo_reqwidth()
    # The Process tab with each kind's settings shown, a long fit function
    # and the values it found included.
    app.tab.set("Process")
    app._show_tab()
    for kind in ("cut", "despike", "subtract", "smooth", "fit"):
        add_step(app, kind)
        app.update_idletasks()
        widths[f"Steps, {kind}"] = app.side_inner.winfo_reqwidth()
    app.choose_step(0)
    app.cut_from.set("5")
    app.cut_to.set("9")
    app.add_cut()  # a range in the cut list
    app.choose_step(4)
    app.advanced.set(True)
    app.apply_controls()
    app.update_idletasks()
    widths["Steps, a fit's values"] = app.side_inner.winfo_reqwidth()
    # The Measure tab filled in: a region, two points, an FFT's peaks.
    app.tab.set("Measure")
    app._show_tab()
    app.set_region((5, 9))
    x, y = drawn(app)
    app.read_points()
    click(app, x[100], y[100])
    click(app, x[200], y[200])
    app.stop_picking()
    app.update_idletasks()
    widths["Measure, filled in"] = app.side_inner.winfo_reqwidth()
    app.in_place("fft")
    app.update_idletasks()
    widths["Measure, peaks"] = app.side_inner.winfo_reqwidth()
    app.undo_in_place()
    # The Derive tab's This panel row, on panels it has turned.
    app.tab.set("Derive")
    app._show_tab()
    for operation in ("fft", "d2"):
        app.in_place(operation)
        app.update_idletasks()
        widths[f"Derive, {operation} here"] = app.side_inner.winfo_reqwidth()
    assert len(set(widths.values())) == 1, widths


@pytest.mark.parametrize("language", ["en", "zh"])
def test_save_area_does_not_widen_the_column(app):
    """The Save area below fits the controls and the scrollbar's slot, leaving
    no gap between the scrollbar and the plot."""
    app.update_idletasks()
    side = app.side.master
    slot = app.side_scroll.master
    assert side.winfo_reqwidth() == app.side.winfo_reqwidth() + slot.winfo_reqwidth()
