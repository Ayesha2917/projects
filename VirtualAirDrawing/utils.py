"""
utils.py
--------
Small, reusable helpers shared across the project: geometry, smoothing,
drawing primitives (rounded rectangles, glass panels, glow effects) and
text rendering. Keeping these here avoids duplicating drawing code in
canvas.py / toolbar.py / main.py.
"""

import math
import time
from collections import deque

import cv2
import numpy as np

import config


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clamp(value, low, high):
    """Clamp a numeric value into [low, high]."""
    return max(low, min(high, value))


def lerp(a, b, t):
    """Linear interpolation between a and b at t in [0, 1]."""
    return a + (b - a) * t


# ------------------------------------------------------------------
# Moving-average smoothing for jitter-free fingertip tracking
# ------------------------------------------------------------------
class MovingAveragePoint:
    """Smooths a stream of (x, y) points with a sliding-window average.

    Using a small deque keeps this O(1) per update and removes the
    high-frequency jitter that raw MediaPipe landmarks exhibit.
    """

    def __init__(self, window=config.SMOOTHING_WINDOW):
        self.window = window
        self._xs = deque(maxlen=window)
        self._ys = deque(maxlen=window)

    def update(self, point):
        self._xs.append(point[0])
        self._ys.append(point[1])
        return self.value

    @property
    def value(self):
        if not self._xs:
            return None
        return (int(sum(self._xs) / len(self._xs)),
                int(sum(self._ys) / len(self._ys)))

    def reset(self):
        self._xs.clear()
        self._ys.clear()


# ------------------------------------------------------------------
# Bezier interpolation for smooth, non-jagged strokes
# ------------------------------------------------------------------
def quadratic_bezier(p0, p1, p2, segments=config.BEZIER_SEGMENTS):
    """Return a list of points along a quadratic bezier curve.

    p1 acts as the control point (typically the midpoint of the raw
    segment), which rounds off sharp corners between fast mouse/finger
    movements and produces smooth, continuous strokes instead of a
    polyline of straight segments.
    """
    points = []
    for i in range(segments + 1):
        t = i / segments
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        points.append((int(x), int(y)))
    return points


# ------------------------------------------------------------------
# Rounded-rectangle / glass panel drawing (used heavily by the toolbar)
# ------------------------------------------------------------------
def draw_rounded_rect(img, top_left, bottom_right, color, radius, thickness=-1):
    """Draw a filled or outlined rectangle with rounded corners."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius < 0:
        radius = 0

    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    return img


def draw_glass_panel(img, top_left, bottom_right, color=config.UI_BG_PANEL,
                      alpha=config.UI_BG_PANEL_ALPHA, radius=config.UI_CORNER_RADIUS,
                      border_color=config.UI_BORDER_COLOR):
    """Draw a translucent 'glassmorphism' panel with a subtle border and
    soft drop shadow, alpha-blended onto img in place.

    Perf note: only the ROI covering the panel (+ shadow offset) is
    copied/blended instead of the whole frame - at 1280x720 with many
    chips drawn per frame, full-frame copies here were the single
    biggest source of lag.
    """
    h, w = img.shape[:2]
    shadow_offset = 6
    pad = radius + shadow_offset + 2

    x1 = clamp(top_left[0] - pad, 0, w)
    y1 = clamp(top_left[1] - pad, 0, h)
    x2 = clamp(bottom_right[0] + pad, 0, w)
    y2 = clamp(bottom_right[1] + pad, 0, h)
    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]
    rel_tl = (top_left[0] - x1, top_left[1] - y1)
    rel_br = (bottom_right[0] - x1, bottom_right[1] - y1)

    overlay = roi.copy()
    draw_rounded_rect(
        overlay,
        (rel_tl[0] + shadow_offset, rel_tl[1] + shadow_offset),
        (rel_br[0] + shadow_offset, rel_br[1] + shadow_offset),
        config.UI_SHADOW_COLOR, radius,
    )
    roi[:] = cv2.addWeighted(overlay, 0.18, roi, 0.82, 0)

    overlay = roi.copy()
    draw_rounded_rect(overlay, rel_tl, rel_br, color, radius)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)

    draw_rounded_rect(roi, rel_tl, rel_br, border_color, radius, thickness=1)
    img[y1:y2, x1:x2] = roi
    return img


def draw_glow_circle(img, center, radius, color, intensity=3):
    """Draw a circle surrounded by soft concentric glow rings, used for
    the animated brush cursor / glow brush effect.

    Perf note: blends only the ROI around the cursor rather than the
    whole frame - this runs every single frame, so it was a steady
    source of lag at full resolution.
    """
    h, w = img.shape[:2]
    pad = radius + intensity * 4 + 4
    cx, cy = center

    x1 = clamp(cx - pad, 0, w)
    y1 = clamp(cy - pad, 0, h)
    x2 = clamp(cx + pad, 0, w)
    y2 = clamp(cy + pad, 0, h)
    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]
    rel_center = (cx - x1, cy - y1)

    overlay = roi.copy()
    for i in range(intensity, 0, -1):
        alpha = 0.10 * (intensity - i + 1)
        cv2.circle(overlay, rel_center, radius + i * 4, color, -1)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    cv2.circle(roi, rel_center, radius, color, -1)
    cv2.circle(roi, rel_center, radius, (255, 255, 255), 1)
    img[y1:y2, x1:x2] = roi
    return img


def draw_ripple(img, center, radius, color, thickness=2):
    """Draw a single ripple ring, used for the color-selection animation."""
    cv2.circle(img, center, radius, color, thickness, lineType=cv2.LINE_AA)
    return img


# ------------------------------------------------------------------
# Text rendering
# ------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(img, text, org, scale=0.6, color=config.UI_TEXT_COLOR, thickness=1,
             shadow=True):
    """Draw anti-aliased text, optionally with a soft drop shadow for
    legibility against the camera feed."""
    if shadow:
        cv2.putText(img, text, (org[0] + 1, org[1] + 1), _FONT, scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)
    return img


def put_text_with_chip(img, text, org, scale=0.55, text_color=config.UI_TEXT_COLOR,
                        chip_color=config.UI_BG_PANEL, alpha=0.55, padding=8):
    """Draw text on top of a small rounded background chip - used for HUD
    labels (FPS, gesture name, mode, etc.)."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    top_left = (x - padding, y - th - padding)
    bottom_right = (x + tw + padding, y + baseline + padding)
    draw_glass_panel(img, top_left, bottom_right, color=chip_color, alpha=alpha, radius=10)
    put_text(img, text, (x, y), scale=scale, color=text_color, shadow=False)
    return img


def timestamp_filename(prefix="drawing", ext="png"):
    """Return a unique, timestamped filename e.g. drawing_20260711_142233.png"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{ext}"


class FPSCounter:
    """Simple exponential-moving-average FPS counter."""

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last_time = time.time()

    def tick(self):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        return self.fps
