"""Reusing a line's steps: Copy steps / Paste steps, and recipes kept in the
data folder's profile."""

import json

from conftest import add_step, plot

from goose_plotter import profile


def two_panels(app):
    """Run 005 with a cut, a subtract and a smooth in panel 1; panel 2 on run
    003 with no steps, selected."""
    plot(app)
    add_step(app, "cut", cuts=((10.0, 12.0),))
    add_step(app, "subtract", degree=3)
    add_step(app, "smooth", window=31)
    steps = app.panel.line.steps
    app.set_layout(1, 2)
    app.selected = (0, 1)
    app._load_controls()
    app.run.set("Cambridge_Sep_26.003")
    app.apply_controls()
    app.panel.line.steps = ()
    app._steps_changed(steps)
    return steps


def test_paste_puts_the_copied_steps_on_another_panels_line(app):
    steps = two_panels(app)
    app.selected = (0, 0)
    app._load_controls()
    app.choose_step(None, "all")
    app.copy_steps()
    app.selected = (0, 1)
    app._load_controls()
    app.paste_steps()
    line = app.panel.line
    assert line.steps == steps and line.run == "Cambridge_Sep_26.003"
    assert line.shown and not line.error
    app.undo()
    assert app.panel.line.steps == ()


def test_paste_onto_another_x_leaves_out_its_cut_ranges(app):
    two_panels(app)
    app.selected = (0, 0)
    app._load_controls()
    app.choose_step(None, "all")
    app.copy_steps()
    app.selected = (0, 1)
    app._load_controls()
    app.x_fn.set("1/x")
    app.apply_controls()
    app.paste_steps()
    line = app.panel.line
    assert [s.kind for s in line.steps] == ["cut", "subtract", "smooth"]
    assert line.steps[0].cuts == () and line.steps[1].degree == 3
    assert "left out" in app.status["text"]


def test_paste_to_every_line_in_the_panel(app):
    steps = two_panels(app)
    app.add_line()
    app.selected = (0, 0)
    app._load_controls()
    app.choose_step(None, "all")
    app.copy_steps()
    app.selected = (0, 1)
    app._load_controls()
    assert app._paste_menu() is not None  # two lines: it asks which
    app.paste_steps(every=True)
    assert [l.steps for l in app.panel.lines] == [steps, steps]


def test_paste_with_nothing_copied_says_so(app):
    plot(app)
    app.paste_steps()
    assert app.panel.line.steps == () and "Copy" in app.status["text"]


def test_a_recipe_is_kept_in_the_profile_and_applied_anywhere(app, data_dir):
    steps = two_panels(app)
    app.selected = (0, 0)
    app._load_controls()
    app.choose_step(None, "all")
    app.save_recipe("standard")
    saved = json.loads((data_dir / profile.PROFILE_NAME).read_text())
    assert "standard" in saved["recipes"] and saved["units"]  # the rest is kept
    assert profile.load_profile(data_dir)[0].recipes == saved["recipes"]
    app.selected = (0, 1)
    app._load_controls()
    app.apply_recipe("standard")
    assert app.panel.line.steps == steps
    menu = app._recipes_menu()
    labels = [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)
              if menu.type(i) != "separator"]
    assert "standard" in labels and "Delete recipe" in labels
    app.delete_recipe("standard")
    assert "recipes" not in json.loads((data_dir / profile.PROFILE_NAME).read_text())
    assert app.profile.recipes == {}


def test_saving_a_recipe_under_a_taken_name_asks_first(app, monkeypatch):
    from goose_plotter import plotter
    plot(app)
    add_step(app, "smooth", window=11)
    app.save_recipe("mine")
    add_step(app, "despike")
    monkeypatch.setattr(plotter.messagebox, "askyesno", lambda *a, **k: False)
    app.save_recipe("mine")
    assert len(app.profile.recipes["mine"]["steps"]) == 1  # not replaced


def test_profile_drops_recipes_it_cant_use(tmp_path):
    (tmp_path / profile.PROFILE_NAME).write_text(json.dumps(
        {"recipes": {"good": {"x": "B", "x_fn": "x", "steps": []}, "bad": "no"}}))
    loaded, error = profile.load_profile(tmp_path)
    assert list(loaded.recipes) == ["good"] and not error


def test_ctrl_and_shift_click_pick_several_steps(app):
    plot(app)
    for kind in ("despike", "subtract", "smooth", "fit"):
        add_step(app, kind)
    line = app.panel.line
    app.choose_step(0)
    app.choose_step(2, "toggle")  # Ctrl-click
    assert line.picks == (0, 2) and line.step == 2
    app.choose_step(2, "toggle")  # and again: unpicked
    assert line.picks == (0,) and line.step == 0
    app.choose_step(2, "range")  # Shift-click: 0 to 2
    assert line.picks == (0, 1, 2)
    assert set(app.step_list.list.curselection()) == {0, 1, 2}
    app.choose_step(3)  # a plain click picks just that one
    assert line.picks == (3,)
    app.choose_step(None, "all")
    assert line.picks == (0, 1, 2, 3)


def test_copy_paste_remove_and_on_act_on_the_picked_steps(app):
    plot(app)
    for kind in ("despike", "subtract", "smooth"):
        add_step(app, kind)
    line = app.panel.line
    app.choose_step(0)
    app.choose_step(2, "toggle")
    app.copy_steps()
    assert [s.kind for s in app.copied[0]] == ["despike", "smooth"]
    app.choose_step(1)  # paste goes in after the chosen step, and is picked
    app.paste_steps()
    assert [s.kind for s in line.steps] == ["despike", "subtract", "despike", "smooth", "smooth"]
    assert line.picks == (2, 3)
    app.step_on.set(False)
    app.toggle_step()
    assert [s.on for s in line.steps] == [True, True, False, False, True]
    app.remove_step()
    assert [s.kind for s in line.steps] == ["despike", "subtract", "smooth"]
    assert line.picks == (2,)


def test_a_recipe_keeps_only_the_picked_steps(app):
    plot(app)
    add_step(app, "despike")
    add_step(app, "smooth", window=41)
    app.choose_step(1)
    app.save_recipe("smooth only")
    saved = app.profile.recipes["smooth only"]["steps"]
    assert [s["kind"] for s in saved] == ["smooth"] and saved[0]["window"] == 41


def test_the_step_list_reads_ctrl_and_shift_on_a_click(app):
    plot(app)
    for kind in ("despike", "subtract", "smooth"):
        add_step(app, kind)
    app.deiconify()  # hidden, the list is never laid out
    app.update()
    rows = app.step_list.list

    def click(row, state=0):
        x, y, _, h = rows.bbox(row)
        rows.event_generate("<Button-1>", x=x + 5, y=y + h // 2, state=state)
        rows.event_generate("<ButtonRelease-1>", x=x + 5, y=y + h // 2, state=state)
        app.update()

    click(0)
    click(2, state=0x4)  # Ctrl
    assert app.panel.line.picks == (0, 2)
    click(1, state=0x1)  # Shift: from the chosen step, 2
    assert app.panel.line.picks == (1, 2)
    rows.focus_force()
    rows.event_generate("<Control-a>")
    app.update()
    assert app.panel.line.picks == (0, 1, 2)
