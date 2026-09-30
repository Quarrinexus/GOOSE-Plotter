import numpy as np
import pytest

from goose_plotter import despike


def noisy(rows=5000, seed=1):
    """A gentle curve with noise of at most 0.01: never a spike at 5 spreads."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, rows)
    return x, np.sin(6 * x) + rng.uniform(-0.01, 0.01, rows)


def test_spikes_become_gaps_and_nothing_else_changes():
    x, y = noisy()
    spiked = y.copy()
    spiked[[1000, 1001, 3000]] += [0.5, -0.4, 0.3]  # two side by side, one alone
    new_x, new_y, count = despike.despike(x, spiked, 21, 5)
    assert count == 3
    assert np.isnan(new_x[[1000, 1001, 3000]]).all() and np.isnan(new_y[[1000, 1001, 3000]]).all()
    kept = np.isfinite(new_y)
    assert np.array_equal(new_y[kept], spiked[kept]) and np.array_equal(new_x[kept], x[kept])


def test_clean_data_is_left_alone():
    x, y = noisy()
    assert despike.despike(x, y, 21, 5)[2] == 0


def test_a_parked_stretch_isnt_all_spikes():
    rng = np.random.default_rng(2)
    y = 1 + rng.uniform(-0.01, 0.01, 2000)
    y[800:900] = 1.0  # parked, read to the same digits
    y[850] = 1.2
    assert np.flatnonzero(despike.spikes(y, 21, 5)).tolist() == [850]


def test_nan_is_never_a_spike_and_doesnt_hide_one():
    rng = np.random.default_rng(3)
    y = 1 + rng.uniform(-0.01, 0.01, 2000)
    y[500:530] = np.nan  # a removed range
    y[535] = 1.2
    assert np.flatnonzero(despike.spikes(y, 21, 5)).tolist() == [535]


def test_a_wider_spike_needs_a_wider_window():
    x, y = noisy()
    y[2000:2012] += 0.5  # 12 points: most of a 21-point window's median
    assert despike.spikes(y, 21, 5)[2000:2012].sum() < 12
    assert despike.spikes(y, 51, 5)[2000:2012].all()


def test_bad_settings_say_why():
    y = np.ones(10)
    with pytest.raises(ValueError, match="at least 3"):
        despike.spikes(y, 1, 5)
    with pytest.raises(ValueError, match="above 0"):
        despike.spikes(y, 5, 0)
    with pytest.raises(ValueError, match="longer than the line"):
        despike.spikes(y, 21, 5)


def test_describe_and_file_part():
    assert despike.describe(21, 5.0) == "despiked 5σ/21"
    assert despike.file_part(21, 4.5) == "ds4.5-21"
