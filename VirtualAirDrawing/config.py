"""
config.py
---------
Central configuration for the Virtual Air Drawing application.
Every tunable constant lives here so behaviour can be adjusted
without touching the logic in the rest of the codebase.
"""

# ------------------------------------------------------------------
# Window / Camera
# ------------------------------------------------------------------
WINDOW_NAME = "Virtual Air Drawing  |  AI Hand-Gesture Studio"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
MIRROR_CAMERA = True  # flip horizontally so movement feels natural

# ------------------------------------------------------------------
# MediaPipe Hands
# ------------------------------------------------------------------
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6

# ------------------------------------------------------------------
# Smoothing / stability
# ------------------------------------------------------------------
SMOOTHING_WINDOW = 5          # moving-average window for fingertip position
GESTURE_STABILITY_FRAMES = 10  # consecutive frames required before a
                                # gesture is considered "confirmed"
PALM_HOLD_SECONDS = 2.0        # open palm hold time required to clear canvas
PINCH_DISTANCE_THRESHOLD = 40  # pixels; below this = pinch engaged
PINCH_RELEASE_THRESHOLD = 55   # pixels; above this = pinch released (hysteresis)
DWELL_SECONDS = 0.5            # hold fingertip on a swatch/button this long to select it

# ------------------------------------------------------------------
# Brush
# ------------------------------------------------------------------
DEFAULT_BRUSH_SIZE = 8
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 60
BRUSH_STEP = 3
DEFAULT_ERASER_SIZE = 40
MIN_ERASER_SIZE = 15
MAX_ERASER_SIZE = 120
BRUSH_OPACITY = 1.0             # 0..1, blended when compositing
BEZIER_SEGMENTS = 12            # interpolation resolution between points

# ------------------------------------------------------------------
# Colors (BGR, OpenCV convention)
# ------------------------------------------------------------------
COLOR_PALETTE = [
    ("Black",  (30, 30, 30)),
    ("White",  (245, 245, 245)),
    ("Blue",   (255, 120, 30)),
    ("Green",  (80, 220, 90)),
    ("Red",    (60, 60, 235)),
    ("Yellow", (30, 220, 245)),
    ("Purple", (200, 80, 160)),
    ("Orange", (30, 140, 255)),
    ("Pink",   (180, 105, 255)),
]
DEFAULT_COLOR_INDEX = 4  # Red

# ------------------------------------------------------------------
# UI / Theme (dark glassmorphism)
# ------------------------------------------------------------------
UI_BG_PANEL = (35, 32, 30)          # base panel color (BGR)
UI_BG_PANEL_ALPHA = 0.55            # glass translucency
UI_BORDER_COLOR = (90, 85, 80)
UI_ACCENT_COLOR = (255, 190, 60)    # highlight / selection accent
UI_TEXT_COLOR = (235, 235, 235)
UI_SUBTEXT_COLOR = (160, 160, 160)
UI_SHADOW_COLOR = (0, 0, 0)
UI_CORNER_RADIUS = 18

TOOLBAR_HEIGHT = 110
TOOLBAR_MARGIN = 16
SWATCH_RADIUS = 22
SWATCH_SPACING = 58
ICON_BUTTON_RADIUS = 26

STATUS_BAR_HEIGHT = 46
NOTIFICATION_DURATION = 2.2  # seconds a toast notification stays visible

FONT = "FONT_HERSHEY_SIMPLEX"  # resolved via cv2 in utils

# ------------------------------------------------------------------
# Canvas
# ------------------------------------------------------------------
BACKGROUND_MODES = ["camera", "black", "white"]
DEFAULT_BACKGROUND_MODE = "camera"
SAVE_DIRECTORY = "SavedDrawings"

# ------------------------------------------------------------------
# Gesture identifiers
# ------------------------------------------------------------------
GESTURE_DRAW = "DRAW"
GESTURE_CURSOR = "CURSOR"
GESTURE_PINCH_SELECT = "PINCH_SELECT"
GESTURE_STOP = "STOP"
GESTURE_CLEAR_HOLD = "CLEAR_HOLD"
GESTURE_INCREASE_SIZE = "INCREASE_SIZE"
GESTURE_DECREASE_SIZE = "DECREASE_SIZE"
GESTURE_SAVE = "SAVE"
GESTURE_UNDO = "UNDO"
GESTURE_REDO = "REDO"
GESTURE_NONE = "NONE"

# Cooldown (seconds) applied to one-shot action gestures so a single
# hold doesn't fire the action repeatedly every frame.
ACTION_COOLDOWN = 1.0
