"""
main.py
-------
Application entry point. Owns the OpenCV video loop and wires together
HandTracker -> GestureDetector -> Canvas -> Toolbar each frame. Keeps
almost no drawing/gesture logic of its own - it is a thin state
machine that reacts to whatever gesture was confirmed this frame.

Run:
    python main.py

Keyboard fallback (also works without a hand in frame):
    q - quit            c - clear canvas       s - save drawing
    z - undo             y - redo                b - background mode
    m - toggle camera    +/- - brush size        1-9 - pick a color
"""

import sys
import time

import cv2
import numpy as np

import config
from canvas import Canvas
from gesture_detector import GestureDetector
from hand_tracker import HandTracker
from toolbar import Toolbar
from utils import (
    FPSCounter, MovingAveragePoint, draw_glow_circle, put_text,
    put_text_with_chip,
)


class Notification:
    """A short-lived toast message, e.g. 'Drawing Saved Successfully'."""

    def __init__(self):
        self.message = ""
        self._start = 0.0

    def show(self, message):
        self.message = message
        self._start = time.time()

    @property
    def visible(self):
        return self.message and (time.time() - self._start) < config.NOTIFICATION_DURATION

    def render(self, frame, width):
        if not self.visible:
            return
        (tw, th), _ = cv2.getTextSize(self.message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = (width - tw) // 2
        y = config.TOOLBAR_HEIGHT + 50
        put_text_with_chip(frame, self.message, (x, y), scale=0.7,
                            text_color=config.UI_ACCENT_COLOR, alpha=0.7)


class VirtualAirDrawingApp:
    """Top-level application: video capture, per-frame pipeline, HUD."""

    def __init__(self):
        self.cap = self._open_camera()
        ret, probe = self.cap.read()
        if not ret:
            raise RuntimeError("Webcam opened but returned no frames.")
        self.frame_h, self.frame_w = probe.shape[:2]

        self.tracker = HandTracker()
        self.gesture = GestureDetector()
        self.canvas = Canvas(self.frame_w, self.frame_h)
        self.toolbar = Toolbar(self.frame_w)
        self.fps_counter = FPSCounter()
        self.notification = Notification()

        self.smoother = MovingAveragePoint()
        self.brush_size = config.DEFAULT_BRUSH_SIZE
        self.eraser_size = config.DEFAULT_ERASER_SIZE
        self.mode = "brush"          # "brush" | "eraser"
        self.status_message = "Ready"
        self.is_drawing = False
        self.no_hand_frames = 0
        self.last_fingertip = None

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, self.frame_w, self.frame_h)

    # -- setup / teardown -------------------------------------------------
    def _open_camera(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check that a camera is connected "
                "and not in use by another application."
            )
        return cap

    def close(self):
        self.cap.release()
        self.tracker.close()
        cv2.destroyAllWindows()

    # -- main loop -------------------------------------------------
    def run(self):
        try:
            while True:
                ok = self._step()
                if not ok:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def _step(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.status_message = "Camera disconnected"
            # keep the window alive and show an error card instead of
            # crashing outright, in case the camera reconnects
            blank = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            put_text(blank, "No camera signal - reconnect webcam",
                     (40, self.frame_h // 2), scale=0.8, color=(0, 0, 255), thickness=2)
            cv2.imshow(config.WINDOW_NAME, blank)
            return cv2.waitKey(200) != ord('q')

        if config.MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.frame_w, self.frame_h))

        hands = self.tracker.process(frame)
        self._handle_hand(hands)

        composited = self.canvas.composite(frame if self.canvas.show_camera
                                            else np.zeros_like(frame))
        self._render_hud(composited, hands)

        cv2.imshow(config.WINDOW_NAME, composited)
        return self._handle_keys()

    # -- gesture -> action pipeline -------------------------------------------------
    def _handle_hand(self, hands):
        if not hands:
            self.no_hand_frames += 1
            if self.canvas.active_stroke is not None:
                self.canvas.end_stroke()
            self.is_drawing = False
            if self.no_hand_frames == 1:
                self.status_message = "No hand detected"
            self.gesture.reset()
            self.smoother.reset()
            return

        if len(hands) > 1:
            self.status_message = "Multiple hands detected - using the first"

        self.no_hand_frames = 0
        hand = hands[0]
        landmarks = hand["landmarks"]

        fingertip_raw = landmarks[8][:2]  # index fingertip
        fingertip = self.smoother.update(fingertip_raw)
        self.last_fingertip = fingertip

        confirmed_gesture, fingers = self.gesture.update(landmarks, hand["label"])

        self.toolbar.update_hover(fingertip)

        self._dispatch(confirmed_gesture, fingertip)

    def _dispatch(self, gesture, fingertip):
        g = config

        # Cursor mode (index + middle together) is the selection mode -
        # hold the fingertip over a swatch/button for DWELL_SECONDS to
        # select it. Any other gesture cancels a half-finished dwell.
        if gesture == g.GESTURE_CURSOR:
            self._stop_drawing()
            hit = self.toolbar.hit_test(fingertip) if fingertip else None
            progress, triggered = self.toolbar.dwell_check(hit)
            if triggered:
                self.toolbar.confirm_selection(hit)
                self._apply_selection(hit)
                self.status_message = "Selected!"
            elif hit is not None:
                self.status_message = f"Hold to select... {int(progress * 100)}%"
            else:
                self.status_message = "Cursor mode"
            return
        else:
            self.toolbar.reset_dwell()

        if gesture == g.GESTURE_DRAW:
            self._continue_drawing(fingertip)
        elif gesture == g.GESTURE_STOP:
            self._stop_drawing()
            self.status_message = "Stopped"
        elif gesture == g.GESTURE_CLEAR_HOLD:
            self._stop_drawing()
            progress = self.gesture.palm_hold_progress()
            self.status_message = f"Hold open palm to clear... {int(progress * 100)}%"
            if self.gesture.should_trigger_clear():
                self.canvas.clear()
                self.notification.show("Canvas Cleared")
        elif gesture == g.GESTURE_INCREASE_SIZE:
            self._stop_drawing()
            if self.gesture.action_ready(g.GESTURE_INCREASE_SIZE):
                self._change_brush_size(config.BRUSH_STEP)
        elif gesture == g.GESTURE_DECREASE_SIZE:
            self._stop_drawing()
            if self.gesture.action_ready(g.GESTURE_DECREASE_SIZE):
                self._change_brush_size(-config.BRUSH_STEP)
        elif gesture == g.GESTURE_SAVE:
            self._stop_drawing()
            if self.gesture.action_ready(g.GESTURE_SAVE):
                self._save_drawing()
        elif gesture == g.GESTURE_UNDO:
            self._stop_drawing()
            if self.gesture.action_ready(g.GESTURE_UNDO):
                if self.canvas.undo():
                    self.notification.show("Undo")
        elif gesture == g.GESTURE_REDO:
            self._stop_drawing()
            if self.gesture.action_ready(g.GESTURE_REDO):
                if self.canvas.redo():
                    self.notification.show("Redo")
        else:
            self._stop_drawing()

    def _continue_drawing(self, fingertip):
        if fingertip is None:
            return
        if not self.is_drawing:
            color = self.toolbar.selected_color
            size = self.eraser_size if self.mode == "eraser" else self.brush_size
            self.canvas.begin_stroke(color, size, is_eraser=(self.mode == "eraser"))
            self.is_drawing = True
        self.canvas.extend_stroke(fingertip)
        self.status_message = "Drawing" if self.mode == "brush" else "Erasing"

    def _stop_drawing(self):
        if self.is_drawing:
            self.canvas.end_stroke()
            self.is_drawing = False

    def _apply_selection(self, hit):
        if hit is None:
            return
        kind, value = hit
        if kind == "color":
            self.mode = "brush"
            self.status_message = f"Color set to {config.COLOR_PALETTE[value][0]}"
        elif kind == "action":
            if value == "brush":
                self.mode = "brush"
            elif value == "eraser":
                self.mode = "eraser"
            self.status_message = f"{value.capitalize()} selected"

    def _change_brush_size(self, delta):
        if self.mode == "eraser":
            self.eraser_size = int(np.clip(self.eraser_size + delta,
                                            config.MIN_ERASER_SIZE, config.MAX_ERASER_SIZE))
            self.status_message = f"Eraser size: {self.eraser_size}"
        else:
            self.brush_size = int(np.clip(self.brush_size + delta,
                                           config.MIN_BRUSH_SIZE, config.MAX_BRUSH_SIZE))
            self.status_message = f"Brush size: {self.brush_size}"

    def _save_drawing(self):
        path = self.canvas.save_png()
        self.notification.show("Drawing Saved Successfully")
        self.status_message = f"Saved to {path}"

    # -- HUD -------------------------------------------------
    def _render_hud(self, frame, hands):
        self.toolbar.render(frame)

        fps = self.fps_counter.tick()
        gesture_label = self.gesture.confirmed_gesture
        color_name = config.COLOR_PALETTE[self.toolbar.selected_color_index][0]

        hud_y = config.TOOLBAR_HEIGHT + 34
        put_text_with_chip(frame, f"FPS: {fps:0.0f}", (16, hud_y))
        put_text_with_chip(frame, f"Gesture: {gesture_label}", (140, hud_y))
        put_text_with_chip(frame, f"Mode: {self.mode.capitalize()}", (360, hud_y))
        size = self.eraser_size if self.mode == "eraser" else self.brush_size
        put_text_with_chip(frame, f"Size: {size}", (520, hud_y))
        put_text_with_chip(frame, f"Color: {color_name}", (630, hud_y))

        if hands:
            confidence = hands[0]["confidence"]
            put_text_with_chip(frame, f"Confidence: {confidence:0.2f}",
                                (frame.shape[1] - 220, hud_y))
        else:
            put_text_with_chip(frame, "No hand in frame", (frame.shape[1] - 220, hud_y),
                                text_color=(0, 165, 255))

        # animated brush cursor
        if self.last_fingertip and self.no_hand_frames == 0:
            color = self.toolbar.selected_color if self.mode == "brush" else (200, 200, 200)
            draw_glow_circle(frame, self.last_fingertip,
                              max(4, (self.brush_size if self.mode == "brush"
                                      else self.eraser_size) // 2), color)

        status_y = frame.shape[0] - 20
        put_text_with_chip(frame, self.status_message, (16, status_y), scale=0.5)

        self.notification.render(frame, frame.shape[1])

    # -- keyboard fallback -------------------------------------------------
    def _handle_keys(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('c'):
            self.canvas.clear()
            self.notification.show("Canvas Cleared")
        elif key == ord('s'):
            self._save_drawing()
        elif key == ord('z'):
            if self.canvas.undo():
                self.notification.show("Undo")
        elif key == ord('y'):
            if self.canvas.redo():
                self.notification.show("Redo")
        elif key == ord('b'):
            modes = config.BACKGROUND_MODES
            idx = (modes.index(self.canvas.background_mode) + 1) % len(modes)
            self.canvas.background_mode = modes[idx]
            self.status_message = f"Background: {self.canvas.background_mode}"
        elif key == ord('m'):
            self.canvas.show_camera = not self.canvas.show_camera
        elif key in (ord('+'), ord('=')):
            self._change_brush_size(config.BRUSH_STEP)
        elif key in (ord('-'), ord('_')):
            self._change_brush_size(-config.BRUSH_STEP)
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(config.COLOR_PALETTE):
                self.toolbar.selected_color_index = idx
        elif key == ord('e'):
            self.mode = "eraser"
        elif key == ord('p'):
            self.mode = "brush"
        return True


def main():
    try:
        app = VirtualAirDrawingApp()
    except RuntimeError as exc:
        print(f"[Startup Error] {exc}", file=sys.stderr)
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
