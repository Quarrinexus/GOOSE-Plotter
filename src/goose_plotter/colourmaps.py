"""The colour maps automatic line colours come from: a panel's lines spread
evenly along a continuous map, or stepping through a set of distinct colours."""

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, to_hex

# The nanoanalysis maps, from github.com/igtz-hub/nanoanalysis_cm (copied, not
# installed: that package registers them with plt.register_cmap, which
# matplotlib 3.9 removed).
NANOANALYSIS = {
    "nanoanalysis_muted": ("#BD2020", "#EDC430", "#2FAC55", "#2F95AC", "#352FAC", "#B63896"),
    "nanoanalysis_bright": ("#B50E0E", "#FF8E00", "#F6F93A", "#0080FF", "#452ABE", "#DE46AE"),
    "nanoanalysis_hazy": ("#EA4A4A", "#EFD267", "#5BDD82", "#67D5EE", "#5F66E9", "#E484CC"),
}

# Map key -> name shown in the menu. "classic" is no map: the profile's sample
# colours, else model.PALETTE.
MAPS = {"nanoanalysis_muted": "nanoanalysis muted", "nanoanalysis_bright": "nanoanalysis bright",
        "nanoanalysis_hazy": "nanoanalysis hazy", "viridis": "viridis", "plasma": "plasma",
        "coolwarm": "coolwarm", "tab10": "tab10", "classic": "Classic"}
DEFAULT = "nanoanalysis_muted"
STEPPED = {"tab10"}  # distinct colours, taken in turn rather than spread


def colourmap(name):
    """The matplotlib colormap for a MAPS key other than "classic"."""
    if name in NANOANALYSIS:
        return LinearSegmentedColormap.from_list(name, NANOANALYSIS[name])
    return matplotlib.colormaps[name]


def colours(name, n):
    """`n` colours from map `name`: from one end of it to the other, or for a
    stepped map its colours in turn."""
    cmap = colourmap(name)
    if name in STEPPED:
        return [to_hex(cmap(i % cmap.N)) for i in range(n)]
    return [to_hex(cmap(i / (n - 1) if n > 1 else 0.0)) for i in range(n)]


def strip(name, steps=64):
    """Colours along a map (not "classic") from end to end, for its picture
    in the menu."""
    cmap = colourmap(name)
    count = cmap.N if name in STEPPED else steps
    return [to_hex(cmap(i if name in STEPPED else i / (count - 1))) for i in range(count)]
