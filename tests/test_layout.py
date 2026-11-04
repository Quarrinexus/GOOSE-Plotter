"""Layouts that aren't grids: rows with their own number of panels, and
sizes set by dragging the gaps between panels."""

import json

import pytest
from conftest import plot
from matplotlib.backend_bases import MouseButton, MouseEvent

from goose_plotter.model import Layout


def test_layout_rows_numbers_and_sizes():
    layout = Layout((1, 3))
    assert layout.cells() == [(0, 0), (1, 0), (1, 1), (1, 2)]
    assert layout.number((1, 2)) == 4 and layout.rows == 2
    assert Layout((2,), widths=((1, 1),)) == Layout((2,))  # equal shares are no shares
    sized = Layout((1, 2), heights=(2, 1), widths=((), (1, 3)))
    assert sized.with_counts((1, 2, 2)).heights == (2, 1, 1)
    assert sized.with_counts((1, 3)).widths == (
        (),
        (),
    )  # a row's widths go with its count
    assert sized.without((1, 0)) == Layout((1, 1), heights=(2, 1))
    assert sized.without((0, 0)) == Layout((2,), widths=((1, 3),))
    assert Layout.from_plain(json.loads(json.dumps(sized.plain()))) == sized
    with pytest.raises(ValueError):
        Layout.from_plain({"counts": [7]})
    assert Layout.from_plain({"counts": [2], "widths": [["x", 1]]}) == Layout((2,))


def test_rows_can_hold_different_numbers_of_panels(app):
    plot(app)
    app.set_counts((1, 2))
    assert sorted(app.axes) == [(0, 0), (1, 0), (1, 1)]
    top, left = app.axes[0, 0].get_position(), app.axes[1, 0].get_position()
    assert top.width > 1.8 * left.width  # the top one spans the row
    app.selected = (1, 1)
    app._step_panel(-1, 0)  # up: the nearest in the shorter row
    assert app.selected == (0, 0)
    assert app._number((1, 1)) == 3


def test_deleting_a_panel_closes_up_its_row(app):
    plot(app)
    app.set_counts((2, 1))
    app._put_derived((0, 1), (1, 0), "fft")
    app.selected = (0, 0)
    app.delete_panel()
    assert app.layout.counts == (1, 1)
    assert app.panels[1, 0].source == (0, 0)  # its data panel moved along
    app.selected = (1, 0)
    app.delete_panel()  # the last in its row: the row goes
    assert app.layout.counts == (1,)


def test_one_go_in_the_layout_picker_is_one_undo_step(app):
    plot(app)
    app.choose_layout()
    picker = app.layout_popup
    picker._step(0, 1)
    picker._change(picker.counts + (3,))  # Add row, then three in it
    picker._step(0, -1)
    assert app.layout.counts == (1, 3)
    app.undo()
    assert app.layout.counts == (1,) and picker.counts == (1,)


def test_sizes_are_saved_with_the_session(app, tmp_path):
    plot(app)
    app.set_counts((2, 1))
    app.layout = app.layout.resized(heights=(3, 1))
    app._build_axes()
    path = tmp_path / "s.json"
    app.save_session(path)
    app.set_counts((1,))
    app.open_session(path)
    assert app.layout == Layout((2, 1), heights=(3, 1))


def drag(app, start, end):
    """Press at `start`, move to `end` and let go there, in figure pixels."""
    for name, (x, y), button in (
        ("button_press_event", start, MouseButton.LEFT),
        ("motion_notify_event", end, None),
        ("button_release_event", end, MouseButton.LEFT),
    ):
        MouseEvent(name, app.canvas, x, y, button=button)._process()


def test_dragging_a_gap_resizes_its_two_sides_in_one_undo_step(app):
    plot(app)
    app.set_counts((2, 2))
    app.deiconify()  # hidden, the plot's side is never laid out
    app.update()
    app.canvas.draw()
    ((kind, _, (x0, y0, x1, y1)),) = [z for z in app.dividers if z[1] == (0, 0)]
    assert kind == "col"
    middle = ((x0 + x1) / 2, (y0 + y1) / 2)
    drag(app, middle, (middle[0] + 150, middle[1]))
    left, right = app.layout.col_widths(0)
    assert left > right and app.layout.counts == (2, 2) and not app.layout.widths[1]
    app.canvas.draw()
    assert app.axes[0, 0].get_position().width > app.axes[0, 1].get_position().width
    app.undo()
    assert app.layout == Layout((2, 2))
    ((_, _, (x0, y0, x1, y1)),) = [z for z in app.dividers if z[0] == "row"]
    middle = ((x0 + x1) / 2, (y0 + y1) / 2)
    drag(app, middle, (middle[0], middle[1] - 2000))  # far past the end: held at LEAST
    top, bottom = app.layout.row_heights()
    assert top / (top + bottom) == pytest.approx(1 - app.LEAST, abs=0.01)
    app.even_sizes()
    assert app.layout == Layout((2, 2))
