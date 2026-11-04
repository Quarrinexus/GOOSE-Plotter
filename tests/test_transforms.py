"""FFT and Derivative steps, and links between a data panel and its FFT:
they share the steps before the FFT, and each keeps its own from there."""

from conftest import add_step, plot

from goose_plotter.model import Step, split_steps, view


def test_split_and_view():
    steps = (Step("smooth"), Step("fft", on=False), Step("cut"), Step("derivative", d_order="d2"))
    assert split_steps(steps) == (steps[:1], steps[1:])  # at the FFT, on or off
    assert view(steps) == "d2" and view(steps[:3]) == "" and view((Step("fft"),)) == "fft"


def fft_of_data(app):
    plot(app, x_fn="1/x")
    app.new_derived_panel("fft")
    return app.panels[0, 0], app.panels[1, 0]


def select(app, cell):
    app.selected = cell
    app._load_controls()


def test_the_data_panels_steps_reach_its_fft_and_back(app):
    data, fft = fft_of_data(app)
    select(app, (0, 0))
    add_step(app, "smooth", window=31)
    assert fft.line.steps == (Step("smooth", window=31), Step("fft"))
    select(app, (1, 0))
    app.choose_step(0)
    add_step(app, "despike")  # before the FFT: shared
    assert data.line.steps == (Step("smooth", window=31), Step("despike"))
    app.choose_step(2)  # the FFT
    add_step(app, "cut", cuts=((None, 20.0),))  # after it: the FFT panel's own
    assert data.line.steps == (Step("smooth", window=31), Step("despike"))
    assert [s.kind for s in fft.line.steps] == ["smooth", "despike", "fft", "cut"]


def test_an_fft_turned_off_is_kept_through_a_sync(app):
    data, fft = fft_of_data(app)
    select(app, (1, 0))
    app.step_on.set(False)
    app.toggle_step()
    select(app, (0, 0))
    add_step(app, "smooth")
    assert fft.line.steps == (Step("smooth"), Step("fft", on=False))
    assert data.line.steps == (Step("smooth"),)


def test_two_fft_panels_keep_their_own_settings(app):
    _, fft = fft_of_data(app)
    select(app, (0, 0))
    app.new_derived_panel("fft")
    other = app.panels[2, 0]
    app._link((1, 0), (2, 0))
    select(app, (2, 0))
    app.fft_pad.set("4")
    app.apply_controls()
    select(app, (0, 0))
    add_step(app, "smooth")
    assert fft.line.steps == (Step("smooth"), Step("fft"))
    assert other.line.steps == (Step("smooth"), Step("fft", pad=4))


def test_a_cut_before_the_fft_cant_be_picked_on_the_spectrum(app):
    _, fft = fft_of_data(app)
    select(app, (0, 0))
    add_step(app, "cut")  # reaches the FFT panel, before its FFT
    select(app, (1, 0))
    app.choose_step(0)
    assert fft.line.chosen.kind == "cut"
    app.pick_range("cut")
    assert app.picker is None and "before the FFT" in app.status["text"]
