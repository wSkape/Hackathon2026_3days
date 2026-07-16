"""
gaze_media_controller.py v7.1

============================

Five coordinated features:

┌──────────────────────────────────────────────────────────────────┐
│  MEDIA PAUSE     → MediaPipe iris tracking                       │
│  Gaze away       → pause after LOOK_AWAY_THRESHOLD seconds       │
├──────────────────────────────────────────────────────────────────┤
│  SMART LOCK      → YOLO11n                                       │
│  Absent → media pause (cascade) → OS lock                        │
├──────────────────────────────────────────────────────────────────┤
│  AUTO-PLAY       after smart lock unlock                         │
│  YOLO detects return → play after UNLOCK_RESUME_DELAY seconds    │
├──────────────────────────────────────────────────────────────────┤
│  PRIVACY SHIELD  → YOLO11n  (≥ 2 people detected)                │
│  Windows → Win+D  |  macOS → osascript  |  Linux → Super+D       │
├──────────────────────────────────────────────────────────────────┤
│  BANPEI          → MediaPipe Pose + Face Landmarker              │
│  Avatar always-on-top (banpei.py) updated in real-time:          │
│    "ok"    → normal posture and fatigue                          │
│    "tired" → low EAR / slow blinks                               │
│    "angry" → uneven shoulders / head too low                     │
│  Reuses the gaze mp_result → no duplicate detections.            │
└──────────────────────────────────────────────────────────────────┘

Installation:
    pip install mediapipe opencv-python pynput numpy ultralytics pillow

Required files in the same folder:
    banpei_core.py      (shared module)
    banpei.py           (avatar window — launch separately)
    banpei_ok.png / banpei_tired.png / banpei_angry.png

Shortcuts:
    L  →  Smart Lock ON/OFF
    P  →  Privacy Shield ON/OFF
    B  →  Banpei debug ON/OFF
    Q  →  Quit
"""
import os
from pathlib import Path
import config  # import constants and OS/utilities
import mp_constantes  # indices and helper functions for MediaPipe
import cv2
import math
import time
import platform
import subprocess
import sys
import threading
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pynput.keyboard import Controller, Key
from ultralytics import YOLO


# Banpei — avatar posture/fatigue (optional: disabled if module absent)
try:
    from banpei_core import BanpeiMonitor
    BANPEI_AVAILABLE = True
except ImportError:
    BANPEI_AVAILABLE = False
    print("[WARN] banpei_core.py introuvable — Banpei désactivé.")


MP_MODEL_PATH = "face_landmarker.task"
MP_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ──────────────────────────────────────────────────────────────────────────────
# OS ACTIONS
# ──────────────────────────────────────────────────────────────────────────────

def lock_screen():
    OS = platform.system()
    print(f"[SMART LOCK] LOCKDOWN ({OS})…")
    try:
        if OS == "Windows":
            import ctypes
            ctypes.windll.user32.LockWorkStation()
        elif OS == "Darwin":
            subprocess.run(["pmset", "displaysleepnow"], check=False)
        else:
            if subprocess.run(["loginctl", "lock-session"],
                              capture_output=True).returncode != 0:
                subprocess.run(["xdg-screensaver", "lock"], check=False)
    except Exception as e:
        print(f"[SMART LOCK] Error : {e}")


def show_desktop():
    """
    Show the desktop (hide all windows).
    Windows : Win+D
    macOS   : Mission Control → desktop (Fn+F11 or Exposé)
    Linux   : Super+D (GNOME/KDE) or wmctrl
    """
    OS = platform.system()
    print(f"[PRIVACY] Displaying desktop ({OS})…")
    kb = Controller()
    try:
        if OS == "Windows":
            # Win + D
            with kb.pressed(Key.cmd):
                kb.press("d")
                kb.release("d")
        elif OS == "Darwin":
            # Ctrl + F3 = afficher le bureau sur macOS standard
            # ou via osascript pour être sûr
            subprocess.run([
                "osascript", "-e",
                'tell application "Finder" to set miniaturized of windows of '
                '(every application process whose visible is true) to true'
            ], check=False)
        else:
            # GNOME / KDE : Super+D
            try:
                with kb.pressed(Key.cmd):
                    kb.press("d")
                    kb.release("d")
            except Exception:
                subprocess.run(["wmctrl", "-k", "on"], check=False)
    except Exception as e:
        print(f"[PRIVACY] Error show_desktop : {e}")


def restore_desktop():
    """
    Restore windows (Win+D a second time = toggle).
    macOS : un-minimize all windows via osascript.
    Linux : wmctrl -k off
    """
    OS = platform.system()
    print(f"[PRIVACY] Restoring desktop ({OS})…")
    kb = Controller()
    try:
        if OS == "Windows":
            with kb.pressed(Key.cmd):
                kb.press("d")
                kb.release("d")
        elif OS == "Darwin":
            subprocess.run([
                "osascript", "-e",
                'tell application "Finder" to set miniaturized of windows of '
                '(every application process whose visible is true) to false'
            ], check=False)
        else:
            try:
                with kb.pressed(Key.cmd):
                    kb.press("d")
                    kb.release("d")
            except Exception:
                subprocess.run(["wmctrl", "-k", "off"], check=False)
    except Exception as e:
        print(f"[PRIVACY] Error restore_desktop : {e}")


# ──────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def download_mp_model(path, url):
    if not os.path.exists(path):
        print(f"[INFO] Downloading MediaPipe model → {path}")
        urllib.request.urlretrieve(url, path)
        print("[INFO] MediaPipe model ready.")


def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [np.array(lm_px(landmarks[i], w, h)) for i in eye_indices[:6]]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def iris_offset_ratio(landmarks, iris_idx, inner_idx, outer_idx,
                      top_idx, bot_idx, w, h):
    iris  = np.array(lm_px(landmarks[iris_idx],  w, h), float)
    inner = np.array(lm_px(landmarks[inner_idx], w, h), float)
    outer = np.array(lm_px(landmarks[outer_idx], w, h), float)
    top   = np.array(lm_px(landmarks[top_idx],   w, h), float)
    bot   = np.array(lm_px(landmarks[bot_idx],   w, h), float)
    ch  = (inner + outer) / 2.0
    cv_ = (top + bot) / 2.0
    hw  = np.linalg.norm(inner - outer) / 2.0 + 1e-6
    hh  = np.linalg.norm(top - bot)     / 2.0 + 1e-6
    return (iris[0] - ch[0]) / hw, (iris[1] - cv_[1]) / hh


def head_yaw_pitch(landmarks, w, h):
    img_pts = np.array(
        [lm_px(landmarks[i], w, h) for i in mp_constantes.HEAD_PNP_INDICES],
        dtype=np.float64
    )
    cam  = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(mp_constantes.HEAD_PNP_3D, img_pts, cam, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    rot, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rot[0][0]**2 + rot[1][0]**2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2( rot[2][1], rot[2][2]))
        yaw   = math.degrees(math.atan2(-rot[2][0], sy))
    else:
        pitch = math.degrees(math.atan2(-rot[1][2], rot[1][1]))
        yaw   = math.degrees(math.atan2(-rot[2][0], sy))
    return yaw, pitch


# ──────────────────────────────────────────────────────────────────────────────
# MEDIA CONTROL
# ──────────────────────────────────────────────────────────────────────────────

class MediaController:
    def __init__(self):
        self.kb      = Controller()
        self._paused = False
        self._lock   = threading.Lock()

    def _tap(self):
        self.kb.press(Key.media_play_pause)
        self.kb.release(Key.media_play_pause)

    def pause(self):
        with self._lock:
            if not self._paused:
                print("[MEDIA] ⏸  Pause")
                self._tap()
                self._paused = True

    def play(self):
        with self._lock:
            if self._paused:
                print("[MEDIA] ▶  Play")
                self._tap()
                self._paused = False

    @property
    def is_paused(self):
        return self._paused


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    download_mp_model(MP_MODEL_PATH, MP_MODEL_URL)

    # ── Lancement automatique de banpei.py ────────────────────────────────────
    banpei_proc = None
    if BANPEI_AVAILABLE and config.BANPEI_ENABLED_DEFAULT:
        banpei_script = Path(__file__).parent / "banpei.py"
        if banpei_script.exists():
            banpei_proc = subprocess.Popen(
                [sys.executable, str(banpei_script)],
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )
            print(f"[BANPEI] Avatar launched (PID {banpei_proc.pid})")
        else:
            print(f"[BANPEI] banpei.py not found in {banpei_script.parent}")

    print("[INFO] Loading YOLO11n…")
    yolo_model = YOLO(config.YOLO_MODEL)
    print("[INFO] YOLO ready.")

    mp_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=config.DETECTION_CONFIDENCE,
        min_face_presence_confidence=config.PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.TRACKING_CONFIDENCE,
    )
    mp_detector = mp_vision.FaceLandmarker.create_from_options(mp_opts)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {config.CAMERA_INDEX} inaccessible.")

    media = MediaController()

    # ── Banpei ────────────────────────────────────────────────────────────────
    banpei       = BanpeiMonitor() if (BANPEI_AVAILABLE and config.BANPEI_ENABLED_DEFAULT) else None
    banpei_debug = config.BANPEI_DEBUG_DEFAULT

    # ── State: media pause ────────────────────────────────────────────────────
    look_away_since = None
    return_since    = None

    # ── State: Smart Lock ─────────────────────────────────────────────────────
    smart_lock_on  = config.SMART_LOCK_ENABLED_DEFAULT
    absent_since   = None
    lock_triggered = False

    # ── State: auto-play post-lock ────────────────────────────────────────────
    waiting_for_unlock_play = False
    returned_after_lock_at  = None

    # ── State: Privacy Shield ─────────────────────────────────────────────────
    privacy_on             = config.PRIVACY_SHIELD_ENABLED_DEFAULT
    privacy_triggered      = False   # desktop already shown for this intrusion
    multi_person_since     = None    # timestamp start for ≥2 people
    single_person_since    = None    # timestamp for return to <2 people

    # ── YOLO cache ────────────────────────────────────────────────────────────
    # Increase num_classes for Privacy Shield: we need the exact count
    person_count   = 1   # optimiste
    person_present = True
    yolo_boxes     = []
    frame_counter  = 0

    print("[INFO] Started  |  'L' Smart Lock  |  'P' Privacy Shield  |  'B' Banpei debug  |  'Q' Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()
        frame_counter += 1

        # ── YOLO ──────────────────────────────────────────────────────────────
        if frame_counter % config.YOLO_EVERY_N_FRAMES == 0:
            res   = yolo_model(frame,
                               classes=[config.COCO_PERSON_CLASS],
                               conf=config.YOLO_PERSON_CONFIDENCE,
                               verbose=False)
            boxes = res[0].boxes
            valid = []
            if boxes is not None and len(boxes):
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if (x2 - x1) * (y2 - y1) / (w * h) > 0.02:
                        valid.append((int(x1), int(y1), int(x2), int(y2),
                                      float(box.conf[0])))
            person_count   = len(valid)
            person_present = person_count >= 1
            yolo_boxes     = valid

        # ── MediaPipe : gaze ──────────────────────────────────────────────────
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = mp_detector.detect(mp_img)

        # ── Banpei : posture + fatigue (reuses rgb and mp_result) ──────────
        if banpei:
            banpei.update(rgb, mp_result, w, h)

        looking  = False
        head_ok  = False
        iris_ok  = False
        avg_h = avg_v = 0.0
        yaw = pitch   = 0.0
        debug_gaze    = []

        if (mp_result.face_landmarks and
                len(mp_result.face_landmarks[0]) > mp_constantes.LEFT_IRIS_CENTER):
            lms = mp_result.face_landmarks[0]

            yaw, pitch = head_yaw_pitch(lms, w, h)
            if yaw is None:
                yaw, pitch = 0.0, 0.0
            head_ok = abs(yaw) <= config.HEAD_YAW_MAX_DEG and abs(pitch) <= config.HEAD_PITCH_MAX_DEG

            ear_r     = eye_aspect_ratio(lms, mp_constantes.RIGHT_EYE, w, h)
            ear_l     = eye_aspect_ratio(lms, mp_constantes.LEFT_EYE,  w, h)
            eyes_open = ear_r > config.EAR_MIN or ear_l > config.EAR_MIN

            rh, rv = iris_offset_ratio(lms, mp_constantes.RIGHT_IRIS_CENTER,
                                       mp_constantes.RIGHT_EYE_INNER, mp_constantes.RIGHT_EYE_OUTER,
                                       mp_constantes.RIGHT_EYE_TOP,   mp_constantes.RIGHT_EYE_BOT, w, h)
            lh, lv = iris_offset_ratio(lms, mp_constantes.LEFT_IRIS_CENTER,
                                       mp_constantes.LEFT_EYE_INNER,  mp_constantes.LEFT_EYE_OUTER,
                                       mp_constantes.LEFT_EYE_TOP,    mp_constantes.LEFT_EYE_BOT,  w, h)

            if eyes_open:
                avg_h   = (abs(rh) + abs(lh)) / 2.0
                avg_v   = (abs(rv) + abs(lv)) / 2.0
                iris_ok = avg_h <= config.IRIS_H_THRESHOLD and avg_v <= config.IRIS_V_THRESHOLD
            else:
                iris_ok = True

            looking = head_ok or iris_ok

            for idx in [mp_constantes.RIGHT_IRIS_CENTER, mp_constantes.LEFT_IRIS_CENTER]:
                cx, cy = lm_px(lms[idx], w, h)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 180), 2)

            debug_gaze = [
                f"Tete  yaw={yaw:+.0f}  pitch={pitch:+.0f}  {'OK' if head_ok else '--'}",
                f"Iris  h={avg_h:.2f}  v={avg_v:.2f}  {'OK' if iris_ok else '--'}",
            ]

        # ═════════════════════════════════════════════════════════════════════=
        # LOGIC
        # ═════════════════════════════════════════════════════════════════════=

        # ── 1. Privacy Shield ─────────────────────────────────────────────────
        privacy_countdown = None   # seconds before triggering (None = inactive)

        if privacy_on:
            if person_count >= config.PRIVACY_PERSONS_THRESHOLD:
                single_person_since = None   # reset the restore timer

                if multi_person_since is None:
                    multi_person_since = now
                    print(f"[PRIVACY] {person_count} detected persons !")

                elapsed_multi = now - multi_person_since
                remaining     = max(0.0, config.PRIVACY_TRIGGER_DELAY - elapsed_multi)
                privacy_countdown = remaining

                if elapsed_multi >= config.PRIVACY_TRIGGER_DELAY and not privacy_triggered:
                    privacy_triggered = True
                    media.pause()
                    threading.Thread(target=show_desktop, daemon=True).start()

            else:
                # Less than 2 people → can we restore?
                multi_person_since = None

                if privacy_triggered:
                    if single_person_since is None:
                        single_person_since = now
                    elif now - single_person_since >= config.PRIVACY_RESTORE_DELAY:
                        print("[PRIVACY] Return to single person — restoring.")
                        privacy_triggered   = False
                        single_person_since = None
                        threading.Thread(target=restore_desktop, daemon=True).start()
                        # NB : on ne fait PAS play ici — c'est le moteur regard
                        #      ou auto-play post-lock qui s'en chargera.
                else:
                    single_person_since = None

        # ── 2. Smart Lock ─────────────────────────────────────────────────────
        lock_countdown = None

        if smart_lock_on:
            if not person_present:
                if absent_since is None:
                    absent_since = now
                    print("[SMART LOCK] Absence detected.")

                elapsed = now - absent_since

                if elapsed >= config.ABSENT_MEDIA_DELAY:
                    media.pause()

                remaining      = config.ABSENT_LOCK_DELAY - elapsed
                lock_countdown = max(0.0, remaining)

                if elapsed >= config.ABSENT_LOCK_DELAY and not lock_triggered:
                    lock_triggered          = True
                    waiting_for_unlock_play = True
                    returned_after_lock_at  = None
                    absent_since            = None
                    threading.Thread(target=lock_screen, daemon=True).start()

                look_away_since = None
                return_since    = None

            else:
                if absent_since is not None:
                    print("[SMART LOCK] Return — cancelled.")
                absent_since   = None
                lock_triggered = False

                if waiting_for_unlock_play:
                    if returned_after_lock_at is None:
                        returned_after_lock_at = now
                        print(f"[AUTO-PLAY] Return detected, play in {config.UNLOCK_RESUME_DELAY:.0f}s…")
                    elif now - returned_after_lock_at >= config.UNLOCK_RESUME_DELAY:
                        media.play()
                        waiting_for_unlock_play = False
                        returned_after_lock_at  = None

        # ── 3. Gaze-based media pause ─────────────────────────────────────────
        # Disabled if Privacy Shield took over OR auto-play is pending
        gaze_active = (person_present
                   and not waiting_for_unlock_play
                   and not privacy_triggered)

        if gaze_active:
            if not looking:
                return_since = None
                if look_away_since is None:
                    look_away_since = now
                elif now - look_away_since >= config.LOOK_AWAY_THRESHOLD:
                    media.pause()
            else:
                look_away_since = None
                if media.is_paused:
                    if return_since is None:
                        return_since = now
                    elif now - return_since >= config.RESUME_DELAY:
                        media.play()
                        return_since = None
                else:
                    return_since = None
        elif not person_present:
            look_away_since = None
            return_since    = None

        # ═════════════════════════════════════════════════════════════════════=
        # HUD
        # ═════════════════════════════════════════════════════════════════════=

        # YOLO boxes — color per person
        BOX_COLORS = [(0, 200, 80), (0, 80, 255), (255, 160, 0), (180, 0, 255)]
        for i, (x1, y1, x2, y2, conf) in enumerate(yolo_boxes):
            col = BOX_COLORS[i % len(BOX_COLORS)]
            # If Privacy Shield active, red boxes for intruders
            if privacy_triggered and i >= 1:
                col = (0, 0, 230)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"P{i+1} {conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

        # Red overlay if Privacy Shield triggered
        if privacy_triggered:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # Top band
        cv2.rectangle(frame, (0, 0), (w, 42), (18, 18, 18), -1)

        pres_col = (60, 220, 60) if person_present else (50, 50, 220)
        pres_txt = (f"{person_count} people{'s' if person_count > 1 else ''}"
                    if person_present else "Absent")
        cv2.putText(frame, pres_txt, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, pres_col, 2)

        if person_present and mp_result.face_landmarks:
            gaze_col = (80, 200, 255) if looking else (50, 100, 255)
            cv2.putText(frame, "Looking" if looking else "Averted",
                        (185, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, gaze_col, 2)

        media_col = (130, 100, 255) if media.is_paused else (100, 255, 100)
        cv2.putText(frame, "PAUSE" if media.is_paused else "PLAY",
                    (w - 115, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, media_col, 2)

        # Badges band
        cv2.rectangle(frame, (0, 42), (w, 62), (28, 28, 28), -1)

        sl_col = (30, 180, 255) if smart_lock_on else (70, 70, 70)
        cv2.putText(frame, f"[L] LOCK {'ON' if smart_lock_on else 'OFF'}",
                    (10, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.44, sl_col, 1)

        pv_col = (0, 200, 200) if privacy_on else (70, 70, 70)
        pv_state = "ACTIF" if privacy_triggered else ("ON" if privacy_on else "OFF")
        cv2.putText(frame, f"[P] PRIVACY {pv_state}",
                    (185, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.44, pv_col, 1)

        # Privacy Shield warning (countdown)
        if privacy_countdown is not None and not privacy_triggered:
            ratio_p = 1.0 - (privacy_countdown / config.PRIVACY_TRIGGER_DELAY)
            bar_w_p = int(ratio_p * w)
            # Fond orange clignotant
            pulse   = abs(math.sin(now * 6))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 80, 220), -1)
            cv2.addWeighted(overlay, 0.08 + 0.08 * pulse, frame,
                            1 - (0.08 + 0.08 * pulse), 0, frame)

            warn = f"⚠  {person_count} people detected — office in {privacy_countdown:.1f}s"
            (tw, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            tx = (w - tw) // 2
            cv2.putText(frame, warn, (tx+1, h//2+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
            cv2.putText(frame, warn, (tx, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 180, 255), 2)

            cv2.rectangle(frame, (0, h - 10), (w, h),      (50, 50, 50), -1)
            cv2.rectangle(frame, (0, h - 10), (bar_w_p, h),(0, 120, 255), -1)

        # Privacy Shield triggered warning
        elif privacy_triggered:
            msg = "PRIVACY SHIELD ACTIF — Office displayed"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            tx = (w - tw) // 2
            cv2.putText(frame, msg, (tx+1, h//2+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            cv2.putText(frame, msg, (tx, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2)

        # Auto-play pending
        elif waiting_for_unlock_play and returned_after_lock_at:
            remaining_ap = max(0.0, config.UNLOCK_RESUME_DELAY - (now - returned_after_lock_at))
            ap_txt = f"▶  Auto-play in {remaining_ap:.1f}s…"
            (tw, _), _ = cv2.getTextSize(ap_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, ap_txt, ((w - tw)//2, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        # Gaze pause bar
        elif look_away_since and not media.is_paused and gaze_active:
            elapsed   = now - look_away_since
            ratio     = min(elapsed / config.LOOK_AWAY_THRESHOLD, 1.0)
            bar_w     = int(ratio * w)
            remaining = max(0.0, config.LOOK_AWAY_THRESHOLD - elapsed)
            cv2.rectangle(frame, (0, h - 10), (w, h),      (50, 50, 50), -1)
            r2 = int(50 + 205 * ratio);  g2 = int(160 * (1 - ratio))
            cv2.rectangle(frame, (0, h - 10), (bar_w, h),  (0, g2, r2), -1)
            cv2.putText(frame, f"Eye break in {remaining:.1f}s",
                        (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # Smart Lock bar
        elif lock_countdown is not None and smart_lock_on:
            ratio_l = 1.0 - (lock_countdown / config.ABSENT_LOCK_DELAY)
            bar_w_l = int(ratio_l * w)

            if lock_countdown <= config.ABSENT_WARNING_BEFORE:
                pulse   = abs(math.sin(now * 3.5))
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
                alpha = 0.10 + 0.10 * pulse
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                warn = f"Locking in  {lock_countdown:.0f}s"
                fs   = 0.88
                (tw, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
                tx = (w - tw) // 2;  ty = h // 2
                cv2.putText(frame, warn, (tx+2, ty+2),
                            cv2.FONT_HERSHEY_DUPLEX, fs, (0, 0, 0), 3)
                cv2.putText(frame, warn, (tx, ty),
                            cv2.FONT_HERSHEY_DUPLEX, fs, (60, 60, 255), 2)

            cv2.rectangle(frame, (0, h - 12), (w, h), (40, 40, 40), -1)
            r3 = int(50 + 205 * ratio_l);  g3 = int(150 * (1 - ratio_l))
            cv2.rectangle(frame, (0, h - 12), (bar_w_l, h), (0, g3, r3), -1)
            absent_el = config.ABSENT_LOCK_DELAY - lock_countdown
            cv2.putText(frame,
                        f"Absent {absent_el:.0f}s/{config.ABSENT_LOCK_DELAY:.0f}s  |  "
                        f"Pause in {max(0.0, config.ABSENT_MEDIA_DELAY - absent_el):.1f}s",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Debug gaze
        for i, line in enumerate(debug_gaze):
            cv2.putText(frame, line, (10, h - 48 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (140, 140, 140), 1)

        cv2.putText(frame, "[Q] Quit", (w - 110, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (65, 65, 65), 1)

        # ── Banpei debug overlay ──────────────────────────────────────────────
        if banpei:
            banpei.draw_debug(frame, show=banpei_debug, frame_h=h)
            # Badge état Banpei dans la bande de badges (toujours visible)
            BANPEI_COLORS = {"ok": (60, 200, 60), "tired": (180, 100, 255), "angry": (0, 80, 255)}
            b_col = BANPEI_COLORS.get(banpei.current_state, (200, 200, 200))
            b_icon = {"ok": "😊", "tired": "😴", "angry": "😠"}.get(banpei.current_state, "?")
            cv2.putText(frame,
                        f"[B] Banpei: {banpei.current_state.upper()}",
                        (w // 2 + 10, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_col, 1)

        cv2.imshow("Gaze Media Controller v7.1", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            smart_lock_on           = not smart_lock_on
            absent_since            = None
            waiting_for_unlock_play = False
            returned_after_lock_at  = None
            print(f"[SMART LOCK] {'ON' if smart_lock_on else 'OFF'}")
        elif key == ord("p"):
            privacy_on         = not privacy_on
            privacy_triggered  = False
            multi_person_since = None
            single_person_since= None
            if not privacy_on:
                threading.Thread(target=restore_desktop, daemon=True).start()
            print(f"[PRIVACY SHIELD] {'ON' if privacy_on else 'OFF'}")
        elif key == ord("b"):
            if banpei:
                banpei_debug = not banpei_debug
                print(f"[BANPEI] Debug {'ON' if banpei_debug else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    mp_detector.close()
    if banpei:
        banpei.close()
    if banpei_proc and banpei_proc.poll() is None:
        banpei_proc.terminate()
        print("[BANPEI] Avatar terminated.")
    print("[INFO] Proper shutdown.")


if __name__ == "__main__":
    main()