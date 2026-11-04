"""The plot's toolbar: matplotlib's, drawn in the window's own style.

matplotlib's Tk toolbar does the work (home, back and forward, pan, zoom,
the subplot tool, saving); this only changes its look. Its icons become
drawn line icons in the colours of the rest of the window (theme.MUTED,
like the swap and axes buttons, theme.ACCENT for pan or zoom while on), its
buttons go flat with the theme's hover, and its tooltips are the window's.
Tk draws a disabled button's picture stippled, so buttons that can't be used
(Back with nothing to go back to) stay enabled, show a faint picture, and
do nothing when clicked (`enable`)."""

import tkinter as tk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from PIL import Image, ImageDraw, ImageTk

from goose_plotter import theme
from goose_plotter.widgets import Tooltip

GRID = 24  # icons are drawn on a 24 x 24 grid, then scaled to the button
K = 4  # and drawn K times larger, then shrunk, for smooth edges


def _stroke(draw, points, width, colour):
    """A polyline with round joints and ends."""
    draw.line(points, fill=colour, width=round(width), joint="curve")
    r = width / 2
    for x, y in points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)


def _icon(paint, size, colour):
    """`paint(draw, u, w, colour)` on a fresh image, `u` the grid unit in
    pixels and `w` the stroke width, as a PhotoImage `size` pixels square."""
    u = size * K / GRID
    image = Image.new("RGBA", (size * K, size * K))
    paint(image, ImageDraw.Draw(image), u, 2 * u, colour)
    return ImageTk.PhotoImage(image.resize((size, size), Image.Resampling.LANCZOS))


def _points(u, *xy):
    return [(x * u, y * u) for x, y in zip(xy[::2], xy[1::2])]


def _home(image, draw, u, w, c):
    _stroke(draw, _points(u, 3, 11, 12, 3.5, 21, 11), w, c)
    _stroke(draw, _points(u, 5.5, 9, 5.5, 20.5, 18.5, 20.5, 18.5, 9), w, c)
    _stroke(draw, _points(u, 10, 20.5, 10, 15, 14, 15, 14, 20.5), w, c)


def _back(image, draw, u, w, c):
    _stroke(draw, _points(u, 20, 12, 4.5, 12), w, c)
    _stroke(draw, _points(u, 11, 5.5, 4.5, 12, 11, 18.5), w, c)


def _forward(image, draw, u, w, c):
    _stroke(draw, _points(u, 4, 12, 19.5, 12), w, c)
    _stroke(draw, _points(u, 13, 5.5, 19.5, 12, 13, 18.5), w, c)


def _pan(image, draw, u, w, c):
    _stroke(draw, _points(u, 12, 2.5, 12, 21.5), w, c)
    _stroke(draw, _points(u, 2.5, 12, 21.5, 12), w, c)
    for head in ((9, 5.5, 12, 2.5, 15, 5.5), (9, 18.5, 12, 21.5, 15, 18.5),
                 (5.5, 9, 2.5, 12, 5.5, 15), (18.5, 9, 21.5, 12, 18.5, 15)):
        _stroke(draw, _points(u, *head), w, c)


def _zoom(image, draw, u, w, c):
    x, y, r = 10.5 * u, 10.5 * u, 7 * u
    draw.ellipse((x - r, y - r, x + r, y + r), outline=c, width=round(w))
    _stroke(draw, _points(u, 15.8, 15.8, 21, 21), w, c)


def _sliders(image, draw, u, w, c):
    for y, knob in ((5.5, 15), (12, 8), (18.5, 13)):
        _stroke(draw, _points(u, 3, y, 21, y), w, c)
        r = 2.6 * u
        draw.ellipse((knob * u - r, y * u - r, knob * u + r, y * u + r), fill=c)


def _save(image, draw, u, w, c):
    _stroke(draw, _points(u, 4, 3.5, 16.5, 3.5, 20.5, 7.5, 20.5, 20.5, 4, 20.5, 4, 3.5), w, c)
    _stroke(draw, _points(u, 8, 3.5, 8, 8.5, 15, 8.5, 15, 3.5), w, c)
    _stroke(draw, _points(u, 7.5, 20.5, 7.5, 14, 17, 14, 17, 20.5), w, c)


def _layout(image, draw, u, w, c):
    for x0, y0 in ((3, 3), (13, 3), (3, 13), (13, 13)):
        draw.rounded_rectangle((x0 * u, y0 * u, (x0 + 8) * u, (y0 + 8) * u), radius=1.5 * u,
                               outline=c, width=round(w))


def _delete_panel(image, draw, u, w, c):
    draw.rounded_rectangle((3 * u, 7 * u, 17 * u, 21 * u), radius=1.5 * u, outline=c,
                           width=round(w))
    # The cross, on a cleared disc so it stands apart from the panel's edge.
    x, y, r = 18 * u, 6 * u, 5.5 * u
    clear = Image.new("L", image.size, 0)
    ImageDraw.Draw(clear).ellipse((x - r, y - r, x + r, y + r), fill=255)
    image.paste((0, 0, 0, 0), (0, 0), clear)
    _stroke(draw, _points(u, 15.5, 3.5, 20.5, 8.5), w, c)
    _stroke(draw, _points(u, 15.5, 8.5, 20.5, 3.5), w, c)


# By matplotlib's names for its buttons, and the two GOOSE adds.
ICONS = {"Home": _home, "Back": _back, "Forward": _forward, "Pan": _pan, "Zoom": _zoom,
         "Subplots": _sliders, "Save": _save, "Layout": _layout,
         "Delete panel": _delete_panel}


class Toolbar(NavigationToolbar2Tk):
    """matplotlib's toolbar with GOOSE's look; see the module's docstring."""

    def __init__(self, canvas, window):
        self.icons = {}  # button -> (normal, faint, on) PhotoImages, kept alive
        self.spacers = []
        super().__init__(canvas, window, pack_toolbar=False)
        tips = {text: tip for text, tip, _, _ in self.toolitems if text}
        for text, button in self._buttons.items():
            for event in ("<Enter>", "<Leave>"):  # matplotlib's own tooltip
                button.unbind(event)
            self._hover(button)
            if tips.get(text):
                Tooltip(button, tips[text])
        self._message_label.configure(foreground=theme.MUTED)
        self.set_history_buttons()

    def _size(self, button):
        return button.winfo_pixels("18p")

    def _draw(self, button, name):
        """Give `button` its icon, at the size matplotlib gives its own."""
        button.name = name
        size, paint = self._size(button), ICONS[name]
        normal, faint, on = (_icon(paint, size, c) for c in (theme.MUTED, theme.HINT,
                                                              theme.ACCENT))
        self.icons[button] = (normal, faint, on)
        options = {"image": faint if getattr(button, "enabled", True) is False else normal}
        if isinstance(button, tk.Checkbutton):
            options["selectimage"] = on
        button.configure(**options, width=size, height=size)

    def _style(self, button):
        button.configure(relief=tk.FLAT, overrelief=tk.FLAT, borderwidth=0, padx=3, pady=3,
                         background=theme.BACKGROUND, activebackground=theme.HOVER,
                         highlightthickness=1, highlightbackground=theme.BACKGROUND,
                         highlightcolor=theme.BACKGROUND, cursor="hand2")
        if isinstance(button, tk.Checkbutton):
            button.configure(offrelief=tk.FLAT, selectcolor=theme.ACCENT_SOFT)

    def _hover(self, button):
        """A border on hover, as the window's buttons have."""
        button.bind("<Enter>", lambda _: button.configure(
            highlightbackground=theme.BORDER_HOVER), add="+")
        button.bind("<Leave>", lambda _: button.configure(
            highlightbackground=theme.BACKGROUND), add="+")

    def _Button(self, text, image_file, toggle, command):
        button = super()._Button(text, image_file, toggle, command)
        self._style(button)
        if text in ICONS:
            self._draw(button, text)
        return button

    def _Spacer(self):
        spacer = self.spacer()
        spacer.pack(side=tk.LEFT, padx="3p")
        return spacer

    def _rescale(self):
        """matplotlib redraws its own icons on a DPI change; draw ours again after."""
        super()._rescale()
        for button in self.icons:
            self._draw(button, button.name)
        for spacer in self.spacers:
            spacer.configure(width=1, height=self.winfo_pixels("14p"))

    def button(self, name, command, tip):
        """A button of GOOSE's own (Layout, Delete panel), like the others; not packed."""
        button = tk.Button(self, command=lambda: getattr(button, "enabled", True) and command())
        self._style(button)
        self._draw(button, name)
        self._hover(button)
        Tooltip(button, tip)
        return button

    def spacer(self):
        """A divider like the ones between matplotlib's groups; not packed."""
        spacer = tk.Frame(self, width=1, height=self.winfo_pixels("14p"),
                          background=theme.BORDER)
        self.spacers.append(spacer)
        return spacer

    def enable(self, button, on):
        """Show `button` as usable or not; it does nothing while it isn't."""
        button.enabled = on
        normal, faint, _ = self.icons[button]
        button.configure(image=normal if on else faint,
                         cursor="hand2" if on else "")

    def set_history_buttons(self):
        """Back and Forward faint when there's nowhere to go, rather than stippled."""
        if not getattr(self, "icons", None):
            return  # called while matplotlib is still building the toolbar
        stack = self._nav_stack
        for text, can in (("Back", stack._pos > 0), ("Forward", stack._pos < len(stack) - 1)):
            if text in self._buttons:
                self.enable(self._buttons[text], can)

    def back(self, *args):
        if getattr(self._buttons.get("Back"), "enabled", True):
            super().back(*args)

    def forward(self, *args):
        if getattr(self._buttons.get("Forward"), "enabled", True):
            super().forward(*args)
