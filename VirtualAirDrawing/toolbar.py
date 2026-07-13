"""
toolbar.py
----------
Renders the fixed top toolbar (dark glassmorphism panel) and resolves
pinch-based selection against it. The toolbar never touches drawing
logic directly - it only reports back *what* was selected (a color
index or an action name) and main.py decides what to do with that.
"""

import time

import cv2
import numpy as np

import config
from utils import draw_glass_panel, draw_ripple, put_text, distance


ACTIONS = [
    ("brush", "Brush"),
    ("eraser", "Eraser"),
]

# simple glyphs drawn with primitive shapes instead of external icon
# files, so the project has zero binary asset dependencies
_ICON_GLYPHS = {
    "brush": "B",
    "eraser": "E",
}


class Toolbar:
    """Fixed top toolbar: color palette + action buttons + live status."""

    def __init__(self, width):
        self.width = width
        self.height = config.TOOLBAR_HEIGHT
        self.selected_color_index = config.DEFAULT_COLOR_INDEX
        self.active_action = "brush"

        self._swatch_centers = []   # [(cx, cy, radius, color_index), ...]
        self._button_centers = []   # [(cx, cy, radius, action_key), ...]
        self._layout()

        self._ripples = []  # list of {"center", "color", "start", "duration"}
        self._hover_key = None

        # dwell (hover-and-hold) selection state - replaces pinch-select
        self._dwell_key = None
        self._dwell_start = None
        self._dwell_confirmed = False
        self._dwell_progress = 0.0

    # -- layout -------------------------------------------------
    def _layout(self):
        margin = config.TOOLBAR_MARGIN
        cy = self.height // 2

        # color swatches on the left
        self._swatch_centers = []
        x = margin + config.SWATCH_RADIUS + 90  # leave room for a label
        for i, (_, bgr) in enumerate(config.COLOR_PALETTE):
            self._swatch_centers.append((x, cy, config.SWATCH_RADIUS, i))
            x += config.SWATCH_SPACING

        # action buttons on the right
        self._button_centers = []
        bx = self.width - margin - config.ICON_BUTTON_RADIUS
        for key, _ in reversed(ACTIONS):
            self._button_centers.append((bx, cy, config.ICON_BUTTON_RADIUS, key))
            bx -= config.ICON_BUTTON_RADIUS * 2 + 18

    def resize(self, width):
        self.width = width
        self._layout()

    # -- hit testing (pinch selection) -------------------------------------------------
    def hit_test(self, point):
        """Return ("color", index) or ("action", key) if `point` lands on
        a swatch/button, else None. Used only when a pinch is engaged."""
        for cx, cy, r, idx in self._swatch_centers:
            if distance(point, (cx, cy)) <= r + 6:
                return ("color", idx)
        for cx, cy, r, key in self._button_centers:
            if distance(point, (cx, cy)) <= r + 6:
                return ("action", key)
        return None

    def update_hover(self, point):
        """Track which element the cursor currently hovers, purely for
        the subtle hover-highlight animation."""
        hit = self.hit_test(point) if point else None
        self._hover_key = hit[1] if hit else None
        return hit

    def dwell_check(self, hit):
        """Call every frame while in a selection-eligible gesture (Cursor
        mode) with the current hit_test result. Returns (progress in
        0..1, triggered). Resets automatically if the fingertip moves
        off the element or leaves the toolbar entirely, so a fresh
        DWELL_SECONDS hold is required for every new selection."""
        key = hit[1] if hit else None
        now = time.time()

        if key is None or key != self._dwell_key:
            self._dwell_key = key
            self._dwell_start = now if key is not None else None
            self._dwell_confirmed = False
            self._dwell_progress = 0.0
            return 0.0, False

        elapsed = now - self._dwell_start
        self._dwell_progress = min(elapsed / config.DWELL_SECONDS, 1.0)

        if self._dwell_progress >= 1.0 and not self._dwell_confirmed:
            self._dwell_confirmed = True
            return 1.0, True
        return self._dwell_progress, False

    def reset_dwell(self):
        """Call whenever the hand leaves Cursor mode, so a half-finished
        hold never carries over into the next selection attempt."""
        self._dwell_key = None
        self._dwell_start = None
        self._dwell_confirmed = False
        self._dwell_progress = 0.0

    def confirm_selection(self, hit):
        """Apply a confirmed pinch-release selection and spawn a ripple
        animation at that element's location."""
        if hit is None:
            return
        kind, value = hit
        if kind == "color":
            self.selected_color_index = value
            center = next((c[:2] for c in self._swatch_centers if c[3] == value), None)
            color = config.COLOR_PALETTE[value][1]
        else:
            self.active_action = value
            center = next((b[:2] for b in self._button_centers if b[3] == value), None)
            color = config.UI_ACCENT_COLOR
        if center:
            self._ripples.append({
                "center": center, "color": color,
                "start": time.time(), "duration": 0.5,
            })

    @property
    def selected_color(self):
        return config.COLOR_PALETTE[self.selected_color_index][1]

    # -- rendering -------------------------------------------------
    def render(self, frame):
        """Draw the toolbar panel, swatches, buttons and any active
        ripple animations directly onto `frame` (in place)."""
        draw_glass_panel(frame, (config.TOOLBAR_MARGIN, config.TOOLBAR_MARGIN),
                          (self.width - config.TOOLBAR_MARGIN, self.height))

        put_text(frame, "AIR DRAW", (config.TOOLBAR_MARGIN + 14, self.height // 2 + 6),
                 scale=0.65, color=config.UI_ACCENT_COLOR, thickness=2)

        self._render_swatches(frame)
        self._render_buttons(frame)
        self._render_ripples(frame)

    def _render_swatches(self, frame):
        for cx, cy, r, idx in self._swatch_centers:
            _, bgr = config.COLOR_PALETTE[idx]
            selected = idx == self.selected_color_index
            hovered = idx == self._hover_key
            dwelling = idx == self._dwell_key

            if selected:
                cv2.circle(frame, (cx, cy), r + 6, config.UI_ACCENT_COLOR, 2, cv2.LINE_AA)
            elif hovered:
                cv2.circle(frame, (cx, cy), r + 4, config.UI_BORDER_COLOR, 1, cv2.LINE_AA)

            cv2.circle(frame, (cx, cy), r, bgr, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), r, (20, 20, 20), 1, cv2.LINE_AA)

            if dwelling and self._dwell_progress > 0:
                # filling arc = hold-to-select progress, starts at top
                # and sweeps clockwise as the dwell timer fills up
                end_angle = -90 + 360 * self._dwell_progress
                cv2.ellipse(frame, (cx, cy), (r + 9, r + 9), 0, -90, end_angle,
                            config.UI_ACCENT_COLOR, 3, cv2.LINE_AA)

    def _render_buttons(self, frame):
        for cx, cy, r, key in self._button_centers:
            active = key == self.active_action
            hovered = key == self._hover_key
            dwelling = key == self._dwell_key

            fill = config.UI_ACCENT_COLOR if active else (55, 52, 50)
            cv2.circle(frame, (cx, cy), r, fill, -1, cv2.LINE_AA)
            if hovered and not active:
                cv2.circle(frame, (cx, cy), r + 3, config.UI_BORDER_COLOR, 1, cv2.LINE_AA)

            if dwelling and self._dwell_progress > 0:
                end_angle = -90 + 360 * self._dwell_progress
                cv2.ellipse(frame, (cx, cy), (r + 6, r + 6), 0, -90, end_angle,
                            config.UI_ACCENT_COLOR, 3, cv2.LINE_AA)

            glyph = _ICON_GLYPHS[key]
            text_color = (25, 25, 25) if active else config.UI_TEXT_COLOR
            (tw, th), _ = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, glyph, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            put_text(frame, key.capitalize(), (cx - 24, cy + r + 16), scale=0.38,
                     color=config.UI_SUBTEXT_COLOR, thickness=1, shadow=False)

    def _render_ripples(self, frame):
        now = time.time()
        alive = []
        for ripple in self._ripples:
            t = (now - ripple["start"]) / ripple["duration"]
            if t >= 1.0:
                continue
            radius = int(20 + 40 * t)
            draw_ripple(frame, ripple["center"], radius, ripple["color"], thickness=2)
            alive.append(ripple)
        self._ripples = alive
