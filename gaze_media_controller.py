"""
gaze_media_controller.py v5
============================

TROIS moteurs coordonnés :

┌──────────────────────────────────────────────────────────────────┐
│  PAUSE MÉDIA  → MediaPipe Face Landmarker + iris tracking        │
│  Si le REGARD est détourné → pause après LOOK_AWAY_THRESHOLD s   │
├──────────────────────────────────────────────────────────────────┤
│  SMART LOCK   → YOLO11n                                          │
│  Si ABSENT → déclenche aussi la pause média (cascade)            │
│             → verrouille l'OS après ABSENT_LOCK_DELAY s          │
├──────────────────────────────────────────────────────────────────┤
│  AUTO-PLAY après déverrouillage                                  │
│  Dès que YOLO détecte le retour de l'utilisateur après un        │
│  smart lock, le script attend UNLOCK_RESUME_DELAY s puis         │
│  envoie Play automatiquement.                                    │
└──────────────────────────────────────────────────────────────────┘

Installation :
    pip install mediapipe opencv-python pynput numpy ultralytics

Raccourcis :
    L  →  activer / désactiver Smart Lock
    Q  →  quitter
"""

import cv2
import math
import time
import platform
import subprocess
import threading
import urllib.request
import os

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pynput.keyboard import Controller, Key
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# ── Pause média (regard) ──────────────────────────────────────────────────────
LOOK_AWAY_THRESHOLD = 1.5      # s : regard détourné → pause
RESUME_DELAY        = 0.3      # s : retour du regard → reprise

# ── Smart Lock (présence) ─────────────────────────────────────────────────────
SMART_LOCK_ENABLED_DEFAULT = True
ABSENT_LOCK_DELAY          = 10.0   # s d'absence → verrou OS
ABSENT_MEDIA_DELAY         = 1.5    # s d'absence → pause média (avant le verrou)
ABSENT_WARNING_BEFORE      = 3.0    # s avant verrou → avertissement visuel

# ── Auto-play après smart lock ────────────────────────────────────────────────
# Quand YOLO re-détecte l'utilisateur après un smart lock,
# on attend ce délai (le temps qu'il déverrouille l'écran) avant de faire play.
UNLOCK_RESUME_DELAY = 2.0      # s après retour → play automatique

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_MODEL             = "yolo11n.pt"
COCO_PERSON_CLASS      = 0
YOLO_PERSON_CONFIDENCE = 0.45
YOLO_EVERY_N_FRAMES    = 5     # 1 = chaque frame, 3 = une frame sur 3

# ── Caméra ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Seuils tête ───────────────────────────────────────────────────────────────
HEAD_YAW_MAX_DEG   = 55
HEAD_PITCH_MAX_DEG = 45

# ── Seuils iris ───────────────────────────────────────────────────────────────
IRIS_H_THRESHOLD = 0.38
IRIS_V_THRESHOLD = 0.42
EAR_MIN          = 0.15

# ── MediaPipe ─────────────────────────────────────────────────────────────────
DETECTION_CONFIDENCE = 0.50
PRESENCE_CONFIDENCE  = 0.50
TRACKING_CONFIDENCE  = 0.50

MP_MODEL_PATH = "face_landmarker.task"
MP_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ──────────────────────────────────────────────────────────────────────────────
# INDICES LANDMARKS MEDIAPIPE
# ──────────────────────────────────────────────────────────────────────────────

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE_INNER = 133;  RIGHT_EYE_OUTER = 33
LEFT_EYE_INNER  = 362;  LEFT_EYE_OUTER  = 263
RIGHT_EYE_TOP   = 159;  RIGHT_EYE_BOT   = 145
LEFT_EYE_TOP    = 386;  LEFT_EYE_BOT    = 374
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473

HEAD_PNP_INDICES = [1, 152, 263, 33, 287, 57]
HEAD_PNP_3D = np.array([
    ( 0.0,    0.0,    0.0),
    ( 0.0,  -63.6,  -12.5),
    (-43.3,  32.7,  -26.0),
    ( 43.3,  32.7,  -26.0),
    (-28.9, -28.9,  -24.1),
    ( 28.9, -28.9,  -24.1),
], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# VERROUILLAGE OS
# ──────────────────────────────────────────────────────────────────────────────

def lock_screen():
    OS = platform.system()
    print(f"[SMART LOCK] Verrouillage ({OS})…")
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
        print(f"[SMART LOCK] Erreur : {e}")


# ──────────────────────────────────────────────────────────────────────────────
# UTILITAIRES MEDIAPIPE
# ──────────────────────────────────────────────────────────────────────────────

def download_mp_model(path, url):
    if not os.path.exists(path):
        print(f"[INFO] Téléchargement MediaPipe model → {path}")
        urllib.request.urlretrieve(url, path)
        print("[INFO] Modèle MediaPipe prêt.")


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
        [lm_px(landmarks[i], w, h) for i in HEAD_PNP_INDICES],
        dtype=np.float64
    )
    cam  = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(HEAD_PNP_3D, img_pts, cam, dist,
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
# CONTRÔLE MÉDIA
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
                print("[MEDIA] ▶  Reprise")
                self._tap()
                self._paused = False

    @property
    def is_paused(self):
        return self._paused


# ──────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    download_mp_model(MP_MODEL_PATH, MP_MODEL_URL)

    print("[INFO] Chargement YOLO11n… (téléchargement auto si absent)")
    yolo_model = YOLO(YOLO_MODEL)
    print("[INFO] YOLO prêt.")

    mp_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=DETECTION_CONFIDENCE,
        min_face_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    mp_detector = mp_vision.FaceLandmarker.create_from_options(mp_opts)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Caméra {CAMERA_INDEX} inaccessible.")

    media = MediaController()

    # ── État : pause média (regard) ───────────────────────────────────────────
    look_away_since = None   # timestamp début regard détourné
    return_since    = None   # timestamp retour du regard

    # ── État : Smart Lock ─────────────────────────────────────────────────────
    smart_lock_on  = SMART_LOCK_ENABLED_DEFAULT
    absent_since   = None    # timestamp début absence YOLO
    lock_triggered = False   # verrou déjà envoyé pour cette absence

    # ── État : auto-play après déverrouillage ─────────────────────────────────
    # Après un smart lock, on note le moment où l'utilisateur réapparaît.
    # On attend UNLOCK_RESUME_DELAY secondes puis on fait play.
    waiting_for_unlock_play = False  # True = on attend que l'utilisateur déverrouille
    returned_after_lock_at  = None   # timestamp retour post-smart-lock

    # ── Cache YOLO ────────────────────────────────────────────────────────────
    person_present = True    # optimiste au démarrage
    yolo_boxes     = []
    frame_counter  = 0

    print("[INFO] Démarré  |  'L' Smart Lock  |  'Q' Quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()
        frame_counter += 1

        # ── YOLO : détection de présence ──────────────────────────────────────
        if frame_counter % YOLO_EVERY_N_FRAMES == 0:
            res   = yolo_model(frame,
                               classes=[COCO_PERSON_CLASS],
                               conf=YOLO_PERSON_CONFIDENCE,
                               verbose=False)
            boxes = res[0].boxes
            valid = []
            if boxes is not None and len(boxes):
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if (x2 - x1) * (y2 - y1) / (w * h) > 0.02:
                        valid.append((int(x1), int(y1), int(x2), int(y2)))
            person_present = len(valid) > 0
            yolo_boxes     = valid

        # ── MediaPipe : gaze / iris ───────────────────────────────────────────
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_result = mp_detector.detect(mp_img)

        looking  = False
        head_ok  = False
        iris_ok  = False
        avg_h = avg_v = 0.0
        yaw = pitch   = 0.0
        debug_gaze    = []

        if (mp_result.face_landmarks and
                len(mp_result.face_landmarks[0]) > LEFT_IRIS_CENTER):
            lms = mp_result.face_landmarks[0]

            yaw, pitch = head_yaw_pitch(lms, w, h)
            if yaw is None:
                yaw, pitch = 0.0, 0.0
            head_ok = abs(yaw) <= HEAD_YAW_MAX_DEG and abs(pitch) <= HEAD_PITCH_MAX_DEG

            ear_r     = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            ear_l     = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            eyes_open = ear_r > EAR_MIN or ear_l > EAR_MIN

            rh, rv = iris_offset_ratio(lms, RIGHT_IRIS_CENTER,
                                       RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
                                       RIGHT_EYE_TOP,   RIGHT_EYE_BOT, w, h)
            lh, lv = iris_offset_ratio(lms, LEFT_IRIS_CENTER,
                                       LEFT_EYE_INNER,  LEFT_EYE_OUTER,
                                       LEFT_EYE_TOP,    LEFT_EYE_BOT,  w, h)

            if eyes_open:
                avg_h   = (abs(rh) + abs(lh)) / 2.0
                avg_v   = (abs(rv) + abs(lv)) / 2.0
                iris_ok = avg_h <= IRIS_H_THRESHOLD and avg_v <= IRIS_V_THRESHOLD
            else:
                iris_ok = True

            looking = head_ok or iris_ok

            for idx in [RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER]:
                cx, cy = lm_px(lms[idx], w, h)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 180), 2)

            debug_gaze = [
                f"Tete  yaw={yaw:+.0f}  pitch={pitch:+.0f}  {'OK' if head_ok else '--'}",
                f"Iris  h={avg_h:.2f}  v={avg_v:.2f}  {'OK' if iris_ok else '--'}",
            ]

        # ══════════════════════════════════════════════════════════════════════
        # LOGIQUE PRINCIPALE
        # ══════════════════════════════════════════════════════════════════════

        # ── Smart Lock ────────────────────────────────────────────────────────
        lock_countdown = None

        if smart_lock_on:
            if not person_present:
                # L'utilisateur est absent
                if absent_since is None:
                    absent_since = now
                    print("[SMART LOCK] Absence détectée.")

                elapsed = now - absent_since

                # 1) Cascade : pause média dès ABSENT_MEDIA_DELAY s d'absence
                if elapsed >= ABSENT_MEDIA_DELAY:
                    media.pause()

                # 2) Compte à rebours vers le verrou
                remaining      = ABSENT_LOCK_DELAY - elapsed
                lock_countdown = max(0.0, remaining)

                # 3) Verrouillage OS
                if elapsed >= ABSENT_LOCK_DELAY and not lock_triggered:
                    lock_triggered          = True
                    waiting_for_unlock_play = True   # arme l'auto-play
                    returned_after_lock_at  = None
                    absent_since            = None
                    threading.Thread(target=lock_screen, daemon=True).start()

                # Reset timers du regard (inutile si absent)
                look_away_since = None
                return_since    = None

            else:
                # L'utilisateur est de retour
                if absent_since is not None:
                    print("[SMART LOCK] Retour détecté — compte à rebours annulé.")
                absent_since = None

                # Auto-play post-smart-lock
                if waiting_for_unlock_play:
                    if returned_after_lock_at is None:
                        returned_after_lock_at = now
                        print(f"[AUTO-PLAY] Retour détecté, play dans {UNLOCK_RESUME_DELAY:.0f}s…")
                    elif now - returned_after_lock_at >= UNLOCK_RESUME_DELAY:
                        print("[AUTO-PLAY] ▶  Reprise automatique après déverrouillage.")
                        media.play()
                        waiting_for_unlock_play = False
                        returned_after_lock_at  = None

        # ── Pause média (regard) ──────────────────────────────────────────────
        # Actif seulement si la personne est présente ET que l'auto-play n'est
        # pas en attente (évite de pauser juste après avoir repris).
        if person_present and not waiting_for_unlock_play:
            if not looking:
                return_since = None
                if look_away_since is None:
                    look_away_since = now
                elif now - look_away_since >= LOOK_AWAY_THRESHOLD:
                    media.pause()
            else:
                look_away_since = None
                if media.is_paused:
                    if return_since is None:
                        return_since = now
                    elif now - return_since >= RESUME_DELAY:
                        media.play()
                        return_since = None
                else:
                    return_since = None
        elif not person_present:
            look_away_since = None
            return_since    = None

        # ══════════════════════════════════════════════════════════════════════
        # HUD
        # ══════════════════════════════════════════════════════════════════════

        # Boîtes YOLO
        for (x1, y1, x2, y2) in yolo_boxes:
            col = (0, 200, 80) if person_present else (60, 60, 60)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, "person", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

        # Bande top
        cv2.rectangle(frame, (0, 0), (w, 42), (18, 18, 18), -1)

        # Présence
        pres_col = (60, 220, 60) if person_present else (50, 50, 220)
        cv2.putText(frame, "Présent" if person_present else "Absent",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, pres_col, 2)

        # Regard
        if person_present and mp_result.face_landmarks:
            gaze_col = (80, 200, 255) if looking else (50, 100, 255)
            cv2.putText(frame, "Regarde" if looking else "Détourné",
                        (130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gaze_col, 2)

        # Média
        media_col = (130, 100, 255) if media.is_paused else (100, 255, 100)
        cv2.putText(frame, "PAUSE" if media.is_paused else "LECTURE",
                    (w - 115, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, media_col, 2)

        # Bande Smart Lock
        cv2.rectangle(frame, (0, 42), (w, 62), (28, 28, 28), -1)
        sl_col = (30, 180, 255) if smart_lock_on else (80, 80, 80)
        cv2.putText(frame, f"[L] SMART LOCK {'ON' if smart_lock_on else 'OFF'}",
                    (10, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.46, sl_col, 1)

        # Auto-play en attente
        if waiting_for_unlock_play and returned_after_lock_at:
            remaining_ap = max(0.0, UNLOCK_RESUME_DELAY - (now - returned_after_lock_at))
            ap_txt = f"▶  Auto-play dans {remaining_ap:.1f}s…"
            (tw, _), _ = cv2.getTextSize(ap_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, ap_txt, ((w - tw) // 2, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        # Debug gaze
        for i, line in enumerate(debug_gaze):
            cv2.putText(frame, line, (10, h - 48 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)

        cv2.putText(frame, "[Q] Quitter", (w - 110, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1)

        # ── Barre : pause média (regard) ──────────────────────────────────────
        if look_away_since and not media.is_paused and person_present:
            elapsed   = now - look_away_since
            ratio     = min(elapsed / LOOK_AWAY_THRESHOLD, 1.0)
            bar_w     = int(ratio * w)
            remaining = max(0.0, LOOK_AWAY_THRESHOLD - elapsed)
            cv2.rectangle(frame, (0, h - 10), (w, h),      (50, 50, 50), -1)
            r2 = int(50 + 205 * ratio);  g2 = int(160 * (1 - ratio))
            cv2.rectangle(frame, (0, h - 10), (bar_w, h),  (0, g2, r2), -1)
            cv2.putText(frame, f"Pause regard dans {remaining:.1f}s",
                        (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # ── Compte à rebours Smart Lock ───────────────────────────────────────
        elif lock_countdown is not None and smart_lock_on:
            ratio_l = 1.0 - (lock_countdown / ABSENT_LOCK_DELAY)
            bar_w_l = int(ratio_l * w)

            if lock_countdown <= ABSENT_WARNING_BEFORE:
                pulse   = abs(math.sin(now * 3.5))
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
                alpha = 0.10 + 0.10 * pulse
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                warn = f"Verrouillage dans  {lock_countdown:.0f}s"
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
            absent_elapsed = ABSENT_LOCK_DELAY - lock_countdown
            cv2.putText(frame,
                        f"Absent {absent_elapsed:.0f}s / {ABSENT_LOCK_DELAY:.0f}s  |  "
                        f"Pause média dans {max(0.0, ABSENT_MEDIA_DELAY - absent_elapsed):.1f}s",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow("Gaze Media Controller v5", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            smart_lock_on           = not smart_lock_on
            absent_since            = None
            waiting_for_unlock_play = False
            returned_after_lock_at  = None
            print(f"[SMART LOCK] {'ON' if smart_lock_on else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    mp_detector.close()
    print("[INFO] Arrêt propre.")


if __name__ == "__main__":
    main()