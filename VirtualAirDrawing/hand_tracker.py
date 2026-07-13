"""
hand_tracker.py
----------------
Thin, well-behaved wrapper around MediaPipe Hands. Isolating MediaPipe
here means the rest of the app (gesture_detector, main) only ever deals
with plain landmark dictionaries/lists - not the mediapipe library
directly - which keeps the codebase modular and easy to unit test.
"""

import cv2
import mediapipe as mp

import config


# Landmark index reference (MediaPipe Hands, 21 points):
#   0  Wrist
#   1-4   Thumb (CMC, MCP, IP, TIP)
#   5-8   Index (MCP, PIP, DIP, TIP)
#   9-12  Middle
#   13-16 Ring
#   17-20 Pinky
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20


class HandTracker:
    """Detects and tracks a single hand's 21 landmarks per frame."""

    def __init__(self,
                 max_hands=config.MAX_NUM_HANDS,
                 detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                 tracking_confidence=config.MIN_TRACKING_CONFIDENCE):
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.last_result = None
        self.num_hands_detected = 0

    def process(self, frame_bgr):
        """Run detection on a BGR frame. Returns a list of hands, each a
        dict with pixel-space landmarks and a confidence score. Empty
        list if no hand was found."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        self.last_result = result

        hands_out = []
        if result.multi_hand_landmarks:
            self.num_hands_detected = len(result.multi_hand_landmarks)
            handedness_list = result.multi_handedness
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                points = [(int(lm.x * w), int(lm.y * h), lm.z)
                          for lm in hand_landmarks.landmark]
                confidence = 0.0
                label = "Unknown"
                if handedness_list and idx < len(handedness_list):
                    classification = handedness_list[idx].classification[0]
                    confidence = classification.score
                    label = classification.label
                hands_out.append({
                    "landmarks": points,       # list of (x, y, z) in pixel space
                    "confidence": confidence,
                    "label": label,
                    "raw": hand_landmarks,      # kept for optional mp drawing
                })
        else:
            self.num_hands_detected = 0

        return hands_out

    def draw_landmarks(self, frame_bgr, hand_dict):
        """Overlay MediaPipe's skeleton drawing for debug / camera-overlay
        mode. Uses the stored raw landmark object."""
        self._mp_draw.draw_landmarks(
            frame_bgr,
            hand_dict["raw"],
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_styles.get_default_hand_landmarks_style(),
            self._mp_styles.get_default_hand_connections_style(),
        )
        return frame_bgr

    def close(self):
        self.hands.close()
