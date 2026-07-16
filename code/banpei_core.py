"""
banpei_core.py
==============
Shared module imported by gaze_media_controller.py.
Contains posture and fatigue detectors, the state machine,
and writes banpei_state.json (read by banpei.py).

Do not run directly — use gaze_media_controller.py.
"""

import json
import time
import math
import urllib.request
import os
from collections import deque
import config

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────────────────────────────────────
# BANPEI CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

STATE_FILE = "banpei_state.json"

# ── Pose Landmarker ──────────────────────────────────────
POSE_MODEL_PATH = "pose_landmarker_lite.task"
POSE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def write_state(state: str):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"state": state, "ts": time.time()}, f)
    except Exception as e:
        # Log error when writing shared state file
        print(f"[BANPEI] State writing : {e}")


def _download_if_needed(path: str, url: str):
    if not os.path.exists(path):
        # Download model artifact on demand
        print(f"[BANPEI] Downloading → {path}")
        urllib.request.urlretrieve(url, path)
        print("[BANPEI] Pose model ready.")


def _lm(landmarks, idx, w, h):
    p = landmarks[idx]
    return p.x * w, p.y * h


def _ear(landmarks, indices, w, h):
    pts = [np.array(_lm(landmarks, i, w, h)) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _smooth(deq: deque) -> float:
    return sum(deq) / len(deq) if deq else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# POSTURE DETECTOR (PoseLandmarker — Tasks API)
# ──────────────────────────────────────────────────────────────────────────────

class PostureDetector:
    # PoseLandmarker indices (identical to BlazePose COCO)
    L_SHOULDER = 11
    R_SHOULDER = 12
    NOSE       = 0

    def __init__(self):
        _download_if_needed(POSE_MODEL_PATH, POSE_MODEL_URL)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.55,
            min_pose_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(opts)

    def analyse(self, frame_rgb: np.ndarray):
        """
        Analyze one RGB frame for upper-body posture.

        Returns (is_bad_posture: bool, details: dict).
        `frame_rgb` is an RGB numpy array (already available in the main script).
        """
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_img)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return False, {}

        lms = result.pose_landmarks[0]   # liste de NormalizedLandmark

        ls = lms[self.L_SHOULDER]
        rs = lms[self.R_SHOULDER]
        ns = lms[self.NOSE]

          # Visibility check (attribute provided by Tasks API).
          # If shoulders are not reliably visible, we skip posture decision.
        if getattr(ls, "visibility", 1.0) < 0.4 or \
              getattr(rs, "visibility", 1.0) < 0.4:
                return False, {"visibility": "low"}

        # Compute simple geometric features:
        # - tilt: vertical difference between shoulders
        # - head_ratio: relative head height over shoulder width (low head => slouch)
        tilt           = abs(ls.y - rs.y)
        mid_shoulder_y = (ls.y + rs.y) / 2.0
        shoulder_width = abs(ls.x - rs.x)
        head_height    = mid_shoulder_y - ns.y

        ratio    = (head_height / shoulder_width) if shoulder_width > 1e-4 else 99.0
        bad_tilt = tilt  > config.SHOULDER_TILT_THRESHOLD
        bad_head = ratio < config.HEAD_LOW_RATIO

        details = {
            "tilt":       round(tilt, 4),
            "head_ratio": round(ratio, 3),
            "bad_tilt":   bad_tilt,
            "bad_head":   bad_head,
        }
        return (bad_tilt or bad_head), details

    def close(self):
        self._detector.close()


# ──────────────────────────────────────────────────────────────────────────────
# FATIGUE DETECTOR (reuses Face Landmarker mp_result from main script)
# ──────────────────────────────────────────────────────────────────────────────

class FatigueDetector:
    """
    This detector is initialized without creating a face detector itself.
    It consumes the `mp_result` already produced by the main script to avoid duplicate detections.
    Call `.analyse_result(mp_result, w, h)` on every frame.
    """

    def __init__(self):
        self.ear_buffer       = deque(maxlen=config.EAR_SMOOTH_N)
        self.slow_buffer      = deque(maxlen=config.SLOW_BLINK_WINDOW)
        self.blink_times      = deque()
        self.eye_was_closed   = False
        self.eye_closed_since = None

    def analyse_result(self, mp_result, w: int, h: int):
        """
        Analyze FaceLandmarker result for blink/eye metrics.

        `mp_result` is the output of FaceLandmarker.detect().
        Returns (is_tired: bool, details: dict).
        """
        # Guard: face landmarks must be present and sufficiently detailed
        if not mp_result.face_landmarks or len(mp_result.face_landmarks[0]) < 400:
            return False, {}

        lms = mp_result.face_landmarks[0]
        now = time.time()

        ear_r   = _ear(lms, config.RIGHT_EYE_EAR, w, h)
        ear_l   = _ear(lms, config.LEFT_EYE_EAR,  w, h)
        avg_ear = (ear_r + ear_l) / 2.0

        # Smooth EAR to reduce frame-to-frame noise
        self.ear_buffer.append(avg_ear)
        smooth_ear = _smooth(self.ear_buffer)

        eye_closed = smooth_ear < config.EAR_CLOSED_THRESHOLD

        if eye_closed and not self.eye_was_closed:
            self.eye_closed_since = now
        elif not eye_closed and self.eye_was_closed:
            self.blink_times.append(now)

        self.eye_was_closed = eye_closed

        cutoff = now - config.BLINK_WINDOW
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()

        # Compute blink rate in blinks per minute using the sliding window
        blink_rate = len(self.blink_times) / (config.BLINK_WINDOW / 60.0)

        long_closure = (eye_closed
                        and self.eye_closed_since is not None
                        and (now - self.eye_closed_since) >= config.CLOSED_DURATION_MAX)

        self.slow_buffer.append(1 if smooth_ear < config.EAR_SLOW_THRESHOLD else 0)
        slow_ratio = sum(self.slow_buffer) / len(self.slow_buffer) \
                 if self.slow_buffer else 0

        # Decision rule for 'tired' combines long eye closures, slow-blink ratio, and high blink rate
        tired = (long_closure
             or slow_ratio > config.SLOW_BLINK_RATIO
             or blink_rate  > config.BLINK_RATE_TIRED)

        details = {
            "ear":          round(smooth_ear, 3),
            "slow_ratio":   round(slow_ratio, 2),
            "blink_rate":   round(blink_rate, 1),
            "long_closure": long_closure,
        }
        return tired, details


# ──────────────────────────────────────────────────────────────────────────────
# STATE MACHINE
# ──────────────────────────────────────────────────────────────────────────────

class BanpeiStateMachine:
    def __init__(self):
        self.current         = "ok"
        self.candidate       = "ok"
        self.candidate_since = None
        self.confirm_durations = {
            "ok":    max(config.POSTURE_GOOD_DURATION, config.TIRED_RECOVER_DURATION),
            "tired": config.TIRED_DURATION,
            "angry": config.POSTURE_BAD_DURATION,
        }

    def update(self, raw: str) -> str:
        # Debounce transient state changes: require the candidate state to persist
        # for a configured duration before committing it as the current state.
        now = time.time()
        if raw != self.candidate:
            self.candidate       = raw
            self.candidate_since = now
        if self.candidate != self.current:
            elapsed = now - (self.candidate_since or now)
            needed  = self.confirm_durations.get(self.candidate, 2.0)
            if elapsed >= needed:
                print(f"[BANPEI] {self.current} → {self.candidate}")
                self.current         = self.candidate
                self.candidate_since = now
        return self.current

    @property
    def progress(self) -> float:
        if self.candidate == self.current:
            return 1.0
        now     = time.time()
        elapsed = now - (self.candidate_since or now)
        needed  = self.confirm_durations.get(self.candidate, 2.0)
        return min(elapsed / needed, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# FACADE — single object to instantiate from the main script
# ──────────────────────────────────────────────────────────────────────────────

class BanpeiMonitor:
    """
    Usage from gaze_media_controller.py:

        from banpei_core import BanpeiMonitor
        banpei = BanpeiMonitor()

        # In the main loop, after computing mp_result and frame_rgb:
        banpei_state = banpei.update(frame_rgb, mp_result, w, h)

        # To render debug overlays on the OpenCV frame:
        banpei.draw_debug(frame, show=show_banpei_debug)
    """

    COLORS = {
        "ok":    (60, 200, 60),
        "tired": (180, 100, 255),
        "angry": (0, 80, 255),
    }

    def __init__(self):
        self.posture  = PostureDetector()
        self.fatigue  = FatigueDetector()
        self.sm       = BanpeiStateMachine()

        self._last_written   = "ok"
        self._posture_details = {}
        self._fatigue_details = {}
        self._raw             = "ok"
        self.current_state    = "ok"

        write_state("ok")

    def update(self, frame_rgb: np.ndarray, mp_result, w: int, h: int) -> str:
        """
        Call once per frame.
        `frame_rgb`: RGB numpy frame (already available in main script)
        `mp_result`: FaceLandmarker.detect() result
        Returns the current Banpei state: "ok" | "tired" | "angry"
        """
        bad_posture, self._posture_details = self.posture.analyse(frame_rgb)
        tired,       self._fatigue_details = self.fatigue.analyse_result(mp_result, w, h)

        # Priority: bad posture > fatigue > ok
        if bad_posture:
            self._raw = "angry"
        elif tired:
            self._raw = "tired"
        else:
            self._raw = "ok"

        self.current_state = self.sm.update(self._raw)

        if self.current_state != self._last_written:
            write_state(self.current_state)
            self._last_written = self.current_state

        return self.current_state

    def draw_debug(self, frame: np.ndarray, show: bool = True,
                   frame_h: int = 480):
        """Draw Banpei metrics at the bottom of the OpenCV frame."""
        if not show:
            return

        col = self.COLORS.get(self.current_state, (200, 200, 200))
        pd  = self._posture_details
        fd  = self._fatigue_details

        # Posture line
        if pd and "tilt" in pd:
            bad = pd.get("bad_tilt") or pd.get("bad_head")
            cv2.putText(frame,
                        f"[Banpei] Posture  tilt={pd['tilt']:.3f}  "
                        f"head={pd['head_ratio']:.2f}  "
                        f"{'BAD' if bad else 'OK'}",
                        (10, frame_h - 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (0, 80, 255) if bad else (60, 180, 60), 1)

        # Fatigue line
        if fd and "ear" in fd:
            tired = (fd.get("long_closure")
                     or fd.get("slow_ratio", 0) > config.SLOW_BLINK_RATIO
                     or fd.get("blink_rate", 0) > config.BLINK_RATE_TIRED)
            cv2.putText(frame,
                        f"[Banpei] Fatigue  EAR={fd['ear']:.3f}  "
                        f"slow={fd['slow_ratio']:.2f}  "
                        f"blinks={fd['blink_rate']:.0f}/min  "
                        f"{'TIRED' if tired else 'OK'}",
                        (10, frame_h - 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (180, 80, 255) if tired else (60, 180, 60), 1)

        # Current state + progress towards the next confirmed state
        prog_txt = ""
        if self.sm.candidate != self.current_state:
            prog_txt = f"  →  {self.sm.candidate} {self.sm.progress*100:.0f}%"
            bar_w = int(self.sm.progress * frame.shape[1])
            cv2.rectangle(frame, (0, frame_h - 4),
                          (frame.shape[1], frame_h), (40, 40, 40), -1)
            cv2.rectangle(frame, (0, frame_h - 4),
                          (bar_w, frame_h), col, -1)

        cv2.putText(frame,
                    f"[Banpei] {self.current_state.upper()}{prog_txt}",
                    (10, frame_h - 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)

    def close(self):
        self.posture.close()
        write_state("ok")