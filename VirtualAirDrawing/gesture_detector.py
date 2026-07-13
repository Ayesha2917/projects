"""
gesture_detector.py
--------------------
Turns raw hand landmarks into a stable, debounced gesture label.

Two problems are solved here, deliberately kept separate from
hand_tracker.py (which only knows about MediaPipe) and canvas.py
(which only knows about pixels/strokes):

1. Finger-state classification (which fingers are extended).
2. Temporal stability - a gesture must persist for
   config.GESTURE_STABILITY_FRAMES consecutive frames before it is
   reported as "confirmed", which is what prevents a hand passing
   through a pose for one noisy frame from triggering an action.
"""

import time
from collections import deque

import config
from hand_tracker import (
    WRIST, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_TIP, RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP,
)
from utils import distance


class GestureDetector:
    """Stateful gesture classifier - one instance per tracked hand."""

    def __init__(self, stability_frames=config.GESTURE_STABILITY_FRAMES):
        self.stability_frames = stability_frames
        self._history = deque(maxlen=stability_frames)
        self.confirmed_gesture = config.GESTURE_NONE

        self._palm_hold_start = None
        self._clear_triggered = False

        # cooldowns so one-shot gestures (save/undo/redo/resize) fire once
        # per hold instead of every single frame they remain true
        self._last_action_time = {
            config.GESTURE_SAVE: 0.0,
            config.GESTURE_UNDO: 0.0,
            config.GESTURE_REDO: 0.0,
            config.GESTURE_INCREASE_SIZE: 0.0,
            config.GESTURE_DECREASE_SIZE: 0.0,
        }

        self.is_pinching = False  # hysteresis state for pinch gesture

    # -- finger state -------------------------------------------------
    @staticmethod
    def _finger_states(landmarks, handedness_label):
        """Return a dict of booleans: which of the 5 fingers are extended.

        Index/middle/ring/pinky: extended if the tip is above (smaller y)
        the corresponding PIP joint (image y grows downward).
        Thumb: extended if the tip is further from the wrist, on the
        opposite side of the hand, than the thumb's MCP joint - using x
        alone (and flipping for left vs right hand) is the standard,
        lightweight heuristic that avoids needing 3D depth reasoning.
        """
        def tip_above_pip(tip_idx, pip_idx):
            return landmarks[tip_idx][1] < landmarks[pip_idx][1]

        index = tip_above_pip(INDEX_TIP, INDEX_PIP)
        middle = tip_above_pip(MIDDLE_TIP, MIDDLE_MCP + 1)
        ring = tip_above_pip(RING_TIP, RING_MCP + 1)
        pinky = tip_above_pip(PINKY_TIP, PINKY_MCP + 1)

        thumb_tip_x = landmarks[THUMB_TIP][0]
        thumb_mcp_x = landmarks[2][0]
        if handedness_label == "Right":
            thumb = thumb_tip_x < thumb_mcp_x
        else:
            thumb = thumb_tip_x > thumb_mcp_x

        return {
            "thumb": thumb,
            "index": index,
            "middle": middle,
            "ring": ring,
            "pinky": pinky,
        }

    # -- raw classification --------------------------------------------
    def _classify_raw(self, landmarks, handedness_label):
        f = self._finger_states(landmarks, handedness_label)

        # Thumb + Index pinch is checked independently (distance based),
        # it can overlay any finger pattern, so it is resolved by the
        # caller via `pinch_distance()` rather than here.

        if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
            return config.GESTURE_DRAW, f

        if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
            # Two valid poses share this finger pattern: a relaxed
            # "index + middle" (Cursor Mode) and a spread-apart "Victory
            # sign" (Undo). Spread distance between the two tips is what
            # a human eye actually uses to tell them apart, so it is
            # used here too, normalized by hand size (wrist-to-MCP) so
            # it holds regardless of distance from the camera.
            spread = distance(landmarks[INDEX_TIP][:2], landmarks[MIDDLE_TIP][:2])
            hand_scale = distance(landmarks[WRIST][:2], landmarks[MIDDLE_MCP][:2]) or 1
            normalized_spread = spread / hand_scale
            if normalized_spread > 0.55:
                return config.GESTURE_UNDO, f
            return config.GESTURE_CURSOR, f

        if not any([f["thumb"], f["index"], f["middle"], f["ring"], f["pinky"]]):
            return config.GESTURE_STOP, f
        if all([f["thumb"], f["index"], f["middle"], f["ring"], f["pinky"]]):
            return config.GESTURE_CLEAR_HOLD, f
        if f["index"] and f["middle"] and f["ring"] and not f["pinky"] and not f["thumb"]:
            return config.GESTURE_INCREASE_SIZE, f
        if f["index"] and f["middle"] and f["ring"] and f["pinky"] and not f["thumb"]:
            return config.GESTURE_DECREASE_SIZE, f
        if f["thumb"] and not any([f["index"], f["middle"], f["ring"], f["pinky"]]):
            return config.GESTURE_SAVE, f
        if f["pinky"] and not any([f["thumb"], f["index"], f["middle"], f["ring"]]):
            return config.GESTURE_REDO, f

        return config.GESTURE_NONE, f

    # -- public API -------------------------------------------------
    def update(self, landmarks, handedness_label):
        """Feed one frame of landmarks in. Returns the confirmed
        (debounced) gesture string for this frame."""
        raw_gesture, fingers = self._classify_raw(landmarks, handedness_label)
        self._history.append(raw_gesture)

        # confirm only once the same gesture fills the whole window
        if len(self._history) == self.stability_frames and \
                all(g == raw_gesture for g in self._history):
            self.confirmed_gesture = raw_gesture
        # otherwise keep reporting the previous confirmed gesture; this
        # avoids flicker between poses during a hand transition

        self._update_palm_hold(raw_gesture)
        return self.confirmed_gesture, fingers

    def _update_palm_hold(self, raw_gesture):
        """Open-palm-held-for-2-seconds -> clear canvas. Tracked on the
        raw (not debounced) gesture so the hold timer is accurate."""
        if raw_gesture == config.GESTURE_CLEAR_HOLD:
            if self._palm_hold_start is None:
                self._palm_hold_start = time.time()
        else:
            self._palm_hold_start = None
            self._clear_triggered = False

    def palm_hold_progress(self):
        """Returns 0..1 progress toward the clear-canvas hold trigger."""
        if self._palm_hold_start is None:
            return 0.0
        elapsed = time.time() - self._palm_hold_start
        return min(elapsed / config.PALM_HOLD_SECONDS, 1.0)

    def should_trigger_clear(self):
        """One-shot: returns True exactly once per completed hold."""
        if self._palm_hold_start is None or self._clear_triggered:
            return False
        elapsed = time.time() - self._palm_hold_start
        if elapsed >= config.PALM_HOLD_SECONDS:
            self._clear_triggered = True
            return True
        return False

    def pinch_distance(self, landmarks):
        """Pixel distance between thumb tip and index tip."""
        return distance(landmarks[THUMB_TIP][:2], landmarks[INDEX_TIP][:2])

    def update_pinch_state(self, landmarks):
        """Hysteresis-based pinch state: engages below the tight
        threshold, releases only above the looser threshold, which
        prevents rapid on/off flicker right at the boundary."""
        d = self.pinch_distance(landmarks)
        was_pinching = self.is_pinching
        if not self.is_pinching and d < config.PINCH_DISTANCE_THRESHOLD:
            self.is_pinching = True
        elif self.is_pinching and d > config.PINCH_RELEASE_THRESHOLD:
            self.is_pinching = False
        just_engaged = self.is_pinching and not was_pinching
        just_released = (not self.is_pinching) and was_pinching
        return self.is_pinching, just_engaged, just_released

    def action_ready(self, gesture):
        """Cooldown gate for one-shot action gestures (save/undo/redo/
        resize) so a sustained pose fires the action once, then waits
        config.ACTION_COOLDOWN seconds before it can fire again."""
        now = time.time()
        last = self._last_action_time.get(gesture, 0.0)
        if now - last >= config.ACTION_COOLDOWN:
            self._last_action_time[gesture] = now
            return True
        return False

    def reset(self):
        self._history.clear()
        self.confirmed_gesture = config.GESTURE_NONE
        self._palm_hold_start = None
        self._clear_triggered = False
        self.is_pinching = False
