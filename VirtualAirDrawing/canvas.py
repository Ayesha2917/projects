"""
canvas.py
---------
The drawing engine. Owns the persistent drawing layer (separate from
the live camera frame), the stroke history (for undo/redo), and the
compositing logic that blends strokes onto whichever background mode
is active (camera feed / solid black / solid white).

Design notes
~~~~~~~~~~~~
* The drawing layer is stored as a BGRA numpy array so strokes have a
  real alpha channel - this is what makes "transparent drawing layer"
  and brush opacity possible, and what lets the eraser only affect
  drawn ink instead of ever touching the toolbar or camera feed.
* Each completed stroke is stored as its own object (points, color,
  thickness, brush style, opacity) so undo/redo simply replays the
  stroke list onto a blank layer rather than trying to "subtract"
  pixels, which is both simpler and artifact-free.
"""

import os

import cv2
import numpy as np

import config
from utils import quadratic_bezier, timestamp_filename


class Stroke:
    """One continuous pen-down-to-pen-up drawing action."""

    __slots__ = ("points", "color", "thickness", "brush_type", "opacity", "is_eraser")

    def __init__(self, color, thickness, brush_type="normal", opacity=1.0, is_eraser=False):
        self.points = []          # list of (x, y)
        self.color = color
        self.thickness = thickness
        self.brush_type = brush_type   # "normal" | "glow" | "neon" | "rainbow"
        self.opacity = opacity
        self.is_eraser = is_eraser

    def add_point(self, point):
        self.points.append(point)


class Canvas:
    """Owns the transparent drawing layer and stroke history."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # BGRA layer - all completed + in-progress strokes live here
        self.layer = np.zeros((height, width, 4), dtype=np.uint8)

        self.strokes = []          # completed strokes (undo stack source)
        self.redo_stack = []       # strokes popped by undo, for redo
        self.active_stroke = None  # stroke currently being drawn

        self.background_mode = config.DEFAULT_BACKGROUND_MODE
        self.show_camera = True

        os.makedirs(config.SAVE_DIRECTORY, exist_ok=True)

    # -- stroke lifecycle -------------------------------------------------
    def begin_stroke(self, color, thickness, brush_type="normal",
                      opacity=config.BRUSH_OPACITY, is_eraser=False):
        self.active_stroke = Stroke(color, thickness, brush_type, opacity, is_eraser)

    def extend_stroke(self, point):
        """Add a point to the in-progress stroke and rasterize the new
        segment immediately (bezier-smoothed) so drawing feels live."""
        if self.active_stroke is None:
            return
        pts = self.active_stroke.points
        pts.append(point)
        if len(pts) >= 2:
            self._rasterize_segment(self.active_stroke, pts[-2], pts[-1])

    def end_stroke(self):
        """Commit the in-progress stroke to history (enables undo) and
        clear the redo stack, since a new action invalidates old redos."""
        if self.active_stroke is not None and len(self.active_stroke.points) >= 1:
            self.strokes.append(self.active_stroke)
            self.redo_stack.clear()
        self.active_stroke = None

    def cancel_stroke(self):
        self.active_stroke = None

    # -- rasterization -------------------------------------------------
    def _rasterize_segment(self, stroke, p0, p1):
        """Draw the segment between two consecutive raw points, using a
        quadratic bezier through their midpoint so the line stays smooth
        even when the underlying points arrive at an irregular pace."""
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        curve_points = quadratic_bezier(p0, mid, p1)

        color_bgra = (*stroke.color, int(255 * stroke.opacity))

        for i in range(len(curve_points) - 1):
            a, b = curve_points[i], curve_points[i + 1]
            if stroke.is_eraser:
                # erasing = writing fully transparent (0,0,0,0) pixels;
                # cv2's drawing functions overwrite pixels directly (no
                # alpha blending), so this genuinely clears ink without
                # ever touching the toolbar/HUD, which is drawn later
                # on a separate layer each frame.
                cv2.line(self.layer, a, b, (0, 0, 0, 0), stroke.thickness,
                          lineType=cv2.LINE_AA)
                cv2.circle(self.layer, b, max(1, stroke.thickness // 2),
                           (0, 0, 0, 0), -1)
                continue

            if stroke.brush_type == "glow":
                self._draw_glow_segment(a, b, color_bgra, stroke.thickness)
            else:
                cv2.line(self.layer, a, b, color_bgra, stroke.thickness,
                          lineType=cv2.LINE_AA)

    def _draw_glow_segment(self, a, b, color_bgra, thickness):
        """Soft outer glow + solid core, for the 'glow brush' bonus."""
        core = color_bgra
        outer = (*color_bgra[:3], max(0, int(color_bgra[3] * 0.25)))
        cv2.line(self.layer, a, b, outer, thickness * 3, lineType=cv2.LINE_AA)
        cv2.line(self.layer, a, b, core, thickness, lineType=cv2.LINE_AA)

    # -- undo / redo -------------------------------------------------
    def undo(self):
        if not self.strokes:
            return False
        stroke = self.strokes.pop()
        self.redo_stack.append(stroke)
        self._rebuild_layer()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        stroke = self.redo_stack.pop()
        self.strokes.append(stroke)
        self._rebuild_layer()
        return True

    def _rebuild_layer(self):
        """Replay every remaining stroke onto a blank layer. Simpler and
        more robust than trying to erase pixels of just the undone
        stroke, especially where strokes overlap."""
        self.layer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for stroke in self.strokes:
            pts = stroke.points
            for i in range(len(pts) - 1):
                self._rasterize_segment(stroke, pts[i], pts[i + 1])

    def clear(self):
        self.layer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.strokes.clear()
        self.redo_stack.clear()
        self.active_stroke = None

    # -- compositing -------------------------------------------------
    def composite(self, camera_frame_bgr):
        """Blend the drawing layer onto the chosen background and
        return a BGR frame ready for the toolbar/HUD to draw on top of."""
        if self.background_mode == "black":
            base = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elif self.background_mode == "white":
            base = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        else:  # "camera"
            base = camera_frame_bgr.copy()

        alpha = self.layer[:, :, 3:4].astype(np.float32) / 255.0
        ink = self.layer[:, :, :3].astype(np.float32)
        base_f = base.astype(np.float32)
        blended = ink * alpha + base_f * (1 - alpha)
        return blended.astype(np.uint8)

    # -- save -------------------------------------------------
    def save_png(self):
        """Save the current drawing layer (transparent background) as a
        timestamped PNG inside config.SAVE_DIRECTORY. Returns the path."""
        filename = timestamp_filename(prefix="drawing", ext="png")
        path = os.path.join(config.SAVE_DIRECTORY, filename)
        cv2.imwrite(path, self.layer)
        return path

    def resize(self, width, height):
        """Handle a change in frame size (e.g. camera reconnect at a
        different resolution) by re-rasterizing strokes at the new
        dimensions instead of stretching pixels."""
        if width == self.width and height == self.height:
            return
        self.width, self.height = width, height
        self._rebuild_layer()
