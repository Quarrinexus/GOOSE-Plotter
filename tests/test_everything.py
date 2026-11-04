"""Every control, tab, popup and pick, one after another, on one window: a
check that nothing in the window raises."""

from conftest import drawn, plot
from test_plotter import click

from goose_plotter import smoothing


def exercise(app, tmp_path):
    """Open everything there is to open, so every text is looked up."""
    plot(app, x_fn="1/x")
    for tab in app.tabs:
        app.tab.set(tab)
        app._show_tab()
    app.choose_step(0)  # no steps: says how to add one
    app.add_step("smooth")
    for key in smoothing.METHODS:
        if key:
            app.smooth.set(smoothing.METHODS[key])
            app.apply_controls()
    app.window_unit.set(smoothing.UNITS[True])
    app.apply_controls()
    app.window_unit.set(smoothing.UNITS[False])
    app.apply_controls()
    app.add_step("despike")
    app.step_on.set(False)
    app.toggle_step()
    app.step_on.set(True)
    app.toggle_step()
    app.add_step("subtract")
    app.advanced.set(True)
    app.apply_controls()
    app.fit_function.set("A * sin(B * x) + C")
    app.fit_start.set("B=314")
    app.apply_controls()
    app.advanced.set(False)
    app.apply_controls()
    app.add_step("fit")
    app.move_step(3, 0)
    app.remove_step()
    app.add_step("cut")
    app.cut_from.set("0.05")
    app.cut_to.set("0.1")
    app.add_cut()
    app.pick_range("cut")
    app.stop_picking()
    app.set_region((0.05, 0.1))
    app.mark_var.set(True)
    app.apply_measure()
    x, y = drawn(app)
    app.read_points()
    click(app, x[100], y[100])
    click(app, x[300], y[300])
    app.stop_picking()
    app.copy_measurements()
    app.export_measurements(tmp_path / "m.csv")
    app.add_step("fft")
    app.add_step("derivative")
    app.undo()
    app.new_derived_panel("fft")
    app.add_step("cut")  # of the spectrum
    app.existing_derived_panel("derivative")
    app.stop_picking()
    app.new_derived_panel("derivative")
    app.link_panels()
    app.stop_picking()
    app.freeze()
    app.freeze()
    app.selected = (0, 0)
    app._load_controls()
    app.undo()
    app.undo()
    app.open_axes()
    app.open_style()
    app.open_save_options()
    app.save_session(tmp_path / "s.json")
    app.open_session(tmp_path / "s.json")
    app.save()
    app.update_idletasks()



def test_every_control_runs(app, tmp_path):
    exercise(app, tmp_path)
    assert (tmp_path / "s.json").is_file() and (tmp_path / "m.csv").is_file()
