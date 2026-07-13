# Virtual Air Drawing

Draw in mid-air using nothing but your webcam and your hand. Built with
OpenCV, MediaPipe Hands, and NumPy — a real-time hand-gesture drawing
studio with a dark, glassmorphism-inspired interface, smooth
bezier-interpolated ink, and a full undo/redo history.

## Overview

Virtual Air Drawing tracks a single hand's 21 landmarks per frame,
classifies a debounced gesture from the finger pose, and turns that
gesture into a drawing action — no mouse, no touch, no tiny buttons to
hunt for with a fingertip. Selecting a color or tool is done by holding the fingertip over it in
Cursor mode for half a second (dwell-to-select), and every gesture
requires ~10 stable frames before it fires, so accidental triggers
from a hand simply passing through a pose are filtered out.

## Features

- **Real-time hand tracking** — MediaPipe Hands, single-hand, all 21
  landmarks, moving-average smoothing to remove fingertip jitter.
- **10 gestures**: draw, cursor + dwell-to-select, stop, hold-to-clear,
  increase/decrease brush size, save, undo, redo.
- **Smooth ink** — quadratic bezier interpolation between points, so
  strokes are continuous and anti-aliased instead of dotted or jagged.
- **True eraser** — clears only drawn ink from the transparent layer;
  the toolbar and camera feed are never affected.
- **9-color palette**, selected by dwelling on a swatch in Cursor
  mode, with an animated ripple on selection.
- **Toolbar**: color palette + Brush/Eraser mode toggle, selected by
  dwelling on them in Cursor mode. Save, Clear, Undo, and Redo are
  intentionally **not** toolbar buttons — they only fire from their
  dedicated hand gesture or a keyboard shortcut, so a stray hover near
  the toolbar can never accidentally save or clear your drawing.
- **Undo / Redo** — every completed stroke is one history entry.
- **Save to PNG** — timestamped, transparent-background export to
  `SavedDrawings/`, with an on-screen "Drawing Saved Successfully"
  toast.
- **Background modes** — live camera, solid black, or solid white,
  with the camera overlay togglable independently.
- **Dark glassmorphism UI** — translucent rounded toolbar, soft
  shadows, hover/selection animations, glowing animated brush cursor.
- **Live HUD** — FPS, current gesture, current mode, brush/eraser
  size, current color, hand-tracking confidence, status messages.
- **Robust error handling** — no webcam, no hand in frame, multiple
  hands, and camera disconnects are all handled gracefully instead of
  crashing.
- **Keyboard fallback** for every action, so the app is fully usable
  even without a hand in frame.

## Installation

```bash
git clone <this-repo>
cd VirtualAirDrawing
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.12+
- A webcam
- opencv-python
- mediapipe
- numpy

## Folder Structure

```
VirtualAirDrawing/
├── main.py              # entry point: video loop + app state machine
├── hand_tracker.py       # MediaPipe Hands wrapper
├── gesture_detector.py   # finger-state classification + gesture debouncing
├── canvas.py             # drawing engine: strokes, undo/redo, save, compositing
├── toolbar.py            # glassmorphism UI panel, palette, buttons, animations
├── config.py              # every tunable constant
├── utils.py               # geometry, smoothing, bezier, drawing primitives
├── assets/                # icons / fonts / sounds (optional, currently empty)
├── SavedDrawings/         # PNG exports land here
├── requirements.txt
└── README.md
```

## How Gestures Work

| Gesture | Hand pose | Action |
|---|---|---|
| 1 | Index finger only | Draw |
| 2 | Index + middle, fingers together | Cursor mode (no ink) |
| 3 | Index + middle, held over a color swatch or Brush/Eraser button for 0.5s | Select that color or mode |
| 4 | Closed fist | Stop drawing |
| 5 | Open palm, held 2 seconds | Clear canvas |
| 6 | Index + middle + ring | Increase brush size |
| 7 | Index + middle + ring + pinky | Decrease brush size |
| 8 | Thumbs up | Save drawing |
| 9 | Index + middle, spread apart ("V") | Undo |
| 10 | Little finger only | Redo |

Save, Clear, Undo, and Redo are **gesture/keyboard-only** — they are
not toolbar buttons, so a stray hover near the toolbar can never
trigger them by accident.

Every gesture (except the palm hold, which uses its own 2-second timer)
must be held for ~10 consecutive frames before it is confirmed — this
is what keeps transitions between poses from spamming actions. Once in
Cursor mode, dwelling on a swatch/button for `DWELL_SECONDS` (0.5s by
default, in `config.py`) confirms the selection.

Keyboard fallback: `q` quit · `c` clear · `s` save · `z` undo · `y`
redo · `b` cycle background · `m` toggle camera · `e`/`p` eraser/brush
· `+`/`-` brush size · `1`-`9` pick a color.

## Screenshots

_Add screenshots or a short GIF of the app in action here._

## Future Improvements

- Shape tools (rectangle, circle, triangle, arrow) and a text tool
- Rainbow / neon / particle brush styles
- Adaptive low-light correction and a hand-confidence indicator overlay
- Canvas zoom/pan and a gesture-calibration onboarding flow
- Voice feedback and a guided gesture-training mode
- Mouse-only fallback mode for machines without a webcam
