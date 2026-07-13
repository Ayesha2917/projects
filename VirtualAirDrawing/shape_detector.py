"""
shape_detector.py
------------------
Standalone shape / object recognition + auto-correction engine.

This is the entire "Shape Detection" feature. It is intentionally
self-contained: it does not import canvas.py, main.py, toolbar.py or
gesture_detector.py, and none of those files' existing logic is
changed to make this work - main.py only adds a couple of new lines
that *call into* this module after a stroke finishes (see the
"NEW:" comments in main.py / canvas.py). If this file were deleted,
the rest of the app would run exactly as it did before.

What it does
~~~~~~~~~~~~
Given the list of raw (x, y) points that make up one completed
freehand stroke, `ShapeDetector.detect()` returns either `None`
(nothing recognized confidently enough - the original stroke is left
completely untouched) or a `DetectionResult` describing:

    - shape_name   e.g. "Circle", "Rectangle", "Star", "Arrow"
    - confidence    0..1
    - points        a clean, "perfect" set of points describing that
                     shape, positioned/sized/rotated to match what was
                     drawn
    - closed        whether `points` describes a closed polygon/loop
                     (circle, square, triangle, star, heart, ...) or
                     an open path (line, arrow)

How it classifies shapes
~~~~~~~~~~~~~~~~~~~~~~~~
No neural network / model weights are bundled - training and shipping
a CNN (e.g. on the Google "Quick, Draw!" dataset) is possible but is
a separate undertaking with its own dataset + training pipeline.
Instead this uses classic OpenCV contour geometry, which is fast
enough to run once per completed stroke without affecting the live
30 FPS drawing loop:

  * The stroke's points are filled into a small binary mask and the
    resulting contour's vertex count (via `cv2.approxPolyDP`),
    circularity, aspect ratio and solidity classify the "basic"
    shapes directly: rectangle/square, triangle, circle, ellipse.
  * Star and heart don't reduce to a simple vertex count, so those
    are matched against pre-generated reference contours using
    `cv2.matchShapes` (Hu-moment based -> translation/rotation/scale
    invariant).
  * Open (non-looped) strokes are tested for straightness (line) or
    an elongated shape with a "fanned out" end (arrow).

`ShapeDetector._classify_closed` / `_classify_open` are the only
methods that would need to change to swap in a trained classifier
later - everything else (mask building, corrected-shape generation,
confidence gating) stays the same.
"""

import math

import cv2
import numpy as np

import shape_config
from utils import distance


class DetectionResult:
    """Result of a successful shape recognition."""

    __slots__ = ("shape_name", "confidence", "points", "closed")

    def __init__(self, shape_name, confidence, points, closed):
        self.shape_name = shape_name
        self.confidence = confidence
        self.points = points
        self.closed = closed


class ShapeDetector:
    """Stateless-per-call shape recognizer (holds only cached reference
    contours built once at construction time)."""

    def __init__(self):
        self._templates = {
            "Star": self._build_reference_contour(
                self._generate_star((0.0, 0.0), 200.0, 200.0, 0.0)),
            "Heart": self._build_reference_contour(
                self._generate_heart((0.0, 0.0), 200.0, 200.0, 0.0,
                                      shape_config.TEMPLATE_SAMPLE_POINTS)),
        }

    # -- public API -------------------------------------------------
    def detect(self, raw_points):
        """Classify one completed stroke's points. Returns a
        DetectionResult or None if nothing was recognized confidently
        enough (caller should leave the original stroke as-is)."""
        if raw_points is None or len(raw_points) < shape_config.MIN_STROKE_POINTS:
            return None

        points = [(float(x), float(y)) for x, y in raw_points]
        diagonal = self._bounding_diagonal(points)
        if diagonal < shape_config.MIN_BOUNDING_DIAGONAL:
            return None

        path_length = sum(distance(points[i], points[i + 1]) for i in range(len(points) - 1))
        if path_length / diagonal > shape_config.MAX_PATH_TO_DIAGONAL_RATIO:
            return None  # too incoherent to be a real fingertip trail

        gap = distance(points[0], points[-1])
        is_closed = (gap / diagonal) < shape_config.CLOSED_GAP_RATIO

        result = (self._classify_closed(points, diagonal) if is_closed
                  else self._classify_open(points, diagonal))

        if result is None or result.confidence < shape_config.CONFIDENCE_THRESHOLD:
            return None
        return result

    # -- closed-loop shapes: circle / ellipse / square / rectangle / --
    # -- triangle / star / heart -------------------------------------------------
    def _classify_closed(self, points, diagonal):
        contour, offset = self._build_filled_contour(points)
        if contour is None or cv2.contourArea(contour) < 30:
            return None

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
        circularity = 4 * math.pi * area / (perimeter ** 2)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 0 else 0.0

        epsilon = shape_config.APPROX_EPSILON_FACTOR * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        aspect = max(rw, rh) / max(1e-3, min(rw, rh))

        # Rectangle / Square - four corners, high solidity (i.e. the
        # filled stroke area nearly matches its own convex hull)
        if vertices == 4 and solidity > 0.85:
            box = cv2.boxPoints(rect)
            box_pts = [(float(x + offset[0]), float(y + offset[1])) for x, y in box]
            is_square = abs(aspect - 1.0) < shape_config.SQUARE_ASPECT_TOLERANCE
            name = "Square" if is_square else "Rectangle"
            return DetectionResult(name, min(1.0, solidity), box_pts, True)

        # Triangle
        if vertices == 3 and solidity > 0.80:
            _, tri = cv2.minEnclosingTriangle(contour)
            tri_pts = [(float(p[0][0] + offset[0]), float(p[0][1] + offset[1]))
                       for p in tri]
            return DetectionResult("Triangle", min(1.0, solidity), tri_pts, True)

        # Circle
        if circularity >= shape_config.CIRCLE_CIRCULARITY_MIN and aspect < 1.3:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            center = (cx + offset[0], cy + offset[1])
            pts = self._generate_circle(center, radius, shape_config.CIRCLE_RENDER_POINTS)
            return DetectionResult("Circle", min(1.0, circularity), pts, True)

        # Ellipse (looser circularity but fits an ellipse well)
        if circularity >= shape_config.ELLIPSE_CIRCULARITY_MIN and len(contour) >= 5:
            (ecx, ecy), (ew, eh), eangle = cv2.fitEllipse(contour)
            center = (ecx + offset[0], ecy + offset[1])
            pts = self._generate_ellipse(center, (ew / 2, eh / 2), eangle,
                                          shape_config.ELLIPSE_RENDER_POINTS)
            span = max(1e-3, shape_config.CIRCLE_CIRCULARITY_MIN -
                       shape_config.ELLIPSE_CIRCULARITY_MIN)
            confidence = 0.55 + 0.3 * (circularity - shape_config.ELLIPSE_CIRCULARITY_MIN) / span
            return DetectionResult("Ellipse", float(np.clip(confidence, 0.0, 1.0)), pts, True)

        # Star / Heart - Hu-moment template match against reference
        # shapes, gated by a solidity sanity check (star and heart each
        # have a fairly characteristic filled-area/hull-area ratio; this
        # catches random closed scribbles that happen to score a low
        # Hu-moment distance by coincidence).
        best_name, best_dist = self._match_template(contour)
        if best_name == "Star":
            lo, hi = shape_config.STAR_SOLIDITY_RANGE
            if not (lo <= solidity <= hi):
                best_name = None
        elif best_name == "Heart":
            if solidity < shape_config.HEART_SOLIDITY_MIN:
                best_name = None

        if best_name is not None:
            (cx, cy), (rw2, rh2), angle = rect
            center = (cx + offset[0], cy + offset[1])
            confidence = float(np.clip(
                1.0 - best_dist / shape_config.TEMPLATE_MATCH_MAX_DISTANCE, 0.0, 1.0))
            if best_name == "Star":
                pts = self._generate_star(center, rw2, rh2, angle)
            else:
                pts = self._generate_heart(center, rw2, rh2, angle,
                                            shape_config.HEART_RENDER_POINTS)
            return DetectionResult(best_name, confidence, pts, True)

        return None

    # -- open (non-looped) shapes: line / arrow -------------------------------------------------
    def _classify_open(self, points, diagonal):
        pts_arr = np.array(points, dtype=np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        def project(pt):
            t = (pt[0] - x0) * vx + (pt[1] - y0) * vy
            return (float(x0 + t * vx), float(y0 + t * vy))

        residuals = []
        for x, y in points:
            t = (x - x0) * vx + (y - y0) * vy
            proj = (x0 + t * vx, y0 + t * vy)
            residuals.append(distance((x, y), proj))
        max_residual = max(residuals) if residuals else 0.0
        straightness = 1.0 - min(1.0, max_residual / max(1.0, diagonal * 0.15))

        p_tail = project(points[0])
        p_tip = project(points[-1])

        # Line - the whole stroke hugs a single straight line closely
        if straightness >= shape_config.LINE_STRAIGHTNESS_MIN:
            return DetectionResult("Line", float(np.clip(straightness, 0.0, 1.0)),
                                    [p_tail, p_tip], False)

        # Arrow - elongated stroke where one end (the sketched arrowhead
        # barbs) deviates from the fitted line far more than the other
        # end. Perpendicular deviation (not raw x/y spread, which is
        # dominated by travel distance along a long shaft) is what
        # actually isolates "the barbs fan out here".
        n = len(points)
        head_res = max(residuals[2 * n // 3:]) if n >= 3 else 0.0
        tail_res = max(residuals[: max(1, n // 3)]) if n >= 3 else 0.0
        # the head is whichever end deviates more - the user may have
        # drawn shaft-then-head or head-then-shaft
        near_end_res, far_end_res = max(head_res, tail_res), min(head_res, tail_res)
        elongated = diagonal > shape_config.MIN_BOUNDING_DIAGONAL * shape_config.ARROW_ASPECT_MIN

        if elongated and near_end_res > max(6.0, far_end_res * 2.5):
            if head_res >= tail_res:
                arrow_tail, arrow_tip = p_tail, p_tip
            else:
                arrow_tail, arrow_tip = p_tip, p_tail
            ratio = near_end_res / max(1.0, far_end_res) - 1.0
            confidence = float(np.clip(0.55 + 0.15 * min(1.0, ratio / 3.0), 0.0, 1.0))
            head_size = max(14.0, diagonal * 0.12)
            arrow_pts = self._generate_arrow(arrow_tail, arrow_tip, head_size)
            return DetectionResult("Arrow", confidence, arrow_pts, False)

        return None

    # -- contour construction -------------------------------------------------
    @staticmethod
    def _bounding_diagonal(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0

    @staticmethod
    def _build_filled_contour(points):
        """Fill the hand-drawn loop's interior and return its largest
        external contour, plus the (x, y) offset needed to map
        mask-local coordinates back to original frame coordinates."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 10
        w = int(max_x - min_x) + pad * 2 + 1
        h = int(max_y - min_y) + pad * 2 + 1
        if w <= 0 or h <= 0:
            return None, None

        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = np.array(
            [[int(x - min_x + pad), int(y - min_y + pad)] for x, y in points],
            dtype=np.int32)
        cv2.fillPoly(mask, [shifted], 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        contour = max(contours, key=cv2.contourArea)
        offset = (min_x - pad, min_y - pad)
        return contour, offset

    def _match_template(self, contour):
        best_name, best_dist = None, float("inf")
        for name, ref_contour in self._templates.items():
            dist = cv2.matchShapes(contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        if best_dist <= shape_config.TEMPLATE_MATCH_MAX_DISTANCE:
            return best_name, best_dist
        return None, best_dist

    @staticmethod
    def _build_reference_contour(points):
        arr = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
        return arr.reshape(-1, 1, 2)

    # -- "perfect shape" generators -------------------------------------------------
    @staticmethod
    def _generate_circle(center, radius, n):
        pts = []
        for i in range(n + 1):
            theta = 2 * math.pi * i / n
            pts.append((center[0] + radius * math.cos(theta),
                        center[1] + radius * math.sin(theta)))
        return pts

    @staticmethod
    def _generate_ellipse(center, axes, angle_deg, n):
        a, b = axes
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        pts = []
        for i in range(n + 1):
            theta = 2 * math.pi * i / n
            x = a * math.cos(theta)
            y = b * math.sin(theta)
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            pts.append((center[0] + rx, center[1] + ry))
        return pts

    @staticmethod
    def _generate_star(center, w, h, angle_deg, spikes=5):
        """5-point star with straight edges - a fixed 2*spikes+1
        vertex polygon is enough (no curve sampling needed)."""
        outer_r = max(w, h) / 2.0
        inner_r = outer_r * 0.42
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        total = spikes * 2
        pts = []
        for i in range(total + 1):
            r = outer_r if i % 2 == 0 else inner_r
            theta = math.pi / 2 + 2 * math.pi * i / total
            x = r * math.cos(theta)
            y = -r * math.sin(theta)
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            pts.append((center[0] + rx, center[1] + ry))
        return pts

    @staticmethod
    def _generate_heart(center, w, h, angle_deg, n):
        """Classic parametric heart curve, normalized to the given
        bounding box and rotated to match the sketch's orientation."""
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        xs, ys = [], []
        for i in range(n + 1):
            t = 2 * math.pi * i / n
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            xs.append(x)
            ys.append(-y)  # flip so the cusp points down on screen
        max_x = max(abs(v) for v in xs) or 1.0
        max_y = max(abs(v) for v in ys) or 1.0
        pts = []
        for x, y in zip(xs, ys):
            nx = (x / max_x) * (w / 2.0)
            ny = (y / max_y) * (h / 2.0)
            rx = nx * cos_a - ny * sin_a
            ry = nx * sin_a + ny * cos_a
            pts.append((center[0] + rx, center[1] + ry))
        return pts

    @staticmethod
    def _generate_arrow(tail, tip, head_size):
        """A single connected path (shaft -> tip -> barb -> tip -> barb)
        so it can be rasterized by the existing point-to-point stroke
        renderer with no 'pen up' needed."""
        angle = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
        back1 = angle + math.radians(150)
        back2 = angle - math.radians(150)
        barb1 = (tip[0] + head_size * math.cos(back1), tip[1] + head_size * math.sin(back1))
        barb2 = (tip[0] + head_size * math.cos(back2), tip[1] + head_size * math.sin(back2))
        return [tail, tip, barb1, tip, barb2]
