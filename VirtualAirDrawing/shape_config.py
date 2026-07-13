"""
shape_config.py
----------------
Tunable constants for the Shape / Object Detection feature only.

Kept as its own file (mirroring how config.py centralizes tunables
for the rest of the app) so every threshold used by shape_detector.py
lives in one place and nothing in the original config.py has to be
touched or reorganized.
"""

# ------------------------------------------------------------------
# Feature toggle
# ------------------------------------------------------------------
ENABLE_AUTO_DETECT_DEFAULT = True   # detection runs automatically on
                                     # every completed stroke; 'd' key
                                     # toggles this at runtime

# ------------------------------------------------------------------
# Pre-filtering (skip strokes that are too small/short to be a shape,
# e.g. accidental taps or the tiny wobble of resting a fingertip)
# ------------------------------------------------------------------
MIN_STROKE_POINTS = 12
MIN_BOUNDING_DIAGONAL = 40   # pixels

# ------------------------------------------------------------------
# Closed-vs-open decision
# A stroke is treated as "closed" (candidate for circle / square /
# rectangle / triangle / star / heart) if its start and end points
# are close relative to the shape's own size. Otherwise it is treated
# as "open" (candidate for line / arrow).
# ------------------------------------------------------------------
CLOSED_GAP_RATIO = 0.28   # gap / bounding_diagonal below this = closed

# A real fingertip moves continuously between frames, so a genuine
# stroke's total path length stays within a small multiple of its own
# bounding-box diagonal. Point sequences that jump around far more than
# that (e.g. noise) are rejected before any shape-specific logic runs.
MAX_PATH_TO_DIAGONAL_RATIO = 6.0

# ------------------------------------------------------------------
# Polygon approximation (cv2.approxPolyDP)
# ------------------------------------------------------------------
APPROX_EPSILON_FACTOR = 0.02   # epsilon = factor * contour perimeter

# ------------------------------------------------------------------
# Basic shape classification thresholds
# ------------------------------------------------------------------
CIRCLE_CIRCULARITY_MIN = 0.80      # 4*pi*area/perimeter^2, 1.0 = perfect circle
ELLIPSE_CIRCULARITY_MIN = 0.65     # looser circularity + elongated aspect
SQUARE_ASPECT_TOLERANCE = 0.18     # |1 - w/h| below this counts as "square"
LINE_STRAIGHTNESS_MIN = 0.90       # fitted-line coverage ratio
ARROW_ASPECT_MIN = 1.8             # elongated open stroke with a head

# ------------------------------------------------------------------
# Hu-moment template matching (used for star / heart, which don't
# reduce to a simple vertex count or circularity check)
# ------------------------------------------------------------------
TEMPLATE_MATCH_MAX_DISTANCE = 0.12   # cv2.matchShapes: lower = closer match.
                                      # Real hand-drawn stars/hearts score
                                      # well under 0.05; random closed
                                      # scribbles typically score 0.10+, so
                                      # this is kept tight to avoid false
                                      # positives rather than maximizing
                                      # recall on very rough sketches.
TEMPLATE_SAMPLE_POINTS = 200         # resolution of generated reference shapes

# Extra sanity gates for star/heart (on top of the Hu-moment match) since
# a low match distance alone can occasionally fool a random closed loop.
STAR_SOLIDITY_RANGE = (0.35, 0.75)   # filled_area / hull_area
HEART_SOLIDITY_MIN = 0.75

# ------------------------------------------------------------------
# Confidence gating - the corrected shape only replaces the freehand
# stroke if the classifier is at least this confident, otherwise the
# original hand-drawn stroke is left completely untouched.
# ------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.55

# ------------------------------------------------------------------
# Corrected-shape rendering resolution (points generated per shape,
# converted back into a Stroke's point list so it reuses the existing
# rasterizer / undo-redo machinery unchanged)
# ------------------------------------------------------------------
CIRCLE_RENDER_POINTS = 72
ELLIPSE_RENDER_POINTS = 72
STAR_RENDER_POINTS = 200
HEART_RENDER_POINTS = 200
