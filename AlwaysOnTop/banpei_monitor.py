"""
banpei_monitor.py
=================
Détecte la posture et la fatigue via MediaPipe Pose + Face Landmarker.
Écrit l'état dans banpei_state.json, lu par banpei.py.

États possibles :
    "ok"     → posture correcte, pas de fatigue
    "tired"  → EAR faible / clignements trop lents / yeux fermés longtemps
    "angry"  → mauvaise posture (épaules déséquilibrées, tête trop basse)

Priorité : angry > tired > ok

Dépendances :
    pip install mediapipe opencv-python numpy

'p' dans la fenêtre OpenCV → afficher/masquer le debug
'q' → quitter
"""

import cv2
import json
import math
import time
import urllib.request
import os
from collections import deque
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

STATE_FILE   = "banpei_state.json"
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Fatigue (EAR) ─────────────────────────────────────────────────────────────
EAR_CLOSED_THRESHOLD = 0.20      # en dessous = œil considéré fermé
EAR_SLOW_THRESHOLD   = 0.23      # en dessous = clignement lent
CLOSED_DURATION_MAX  = 0.8       # s : yeux fermés > 0.8s → fatigué
BLINK_WINDOW         = 20.0      # s glissante pour calculer le taux de clignement
BLINK_RATE_TIRED     = 22        # clignements/min au-dessus → fatigué (clignements rapides = fatigue)
SLOW_BLINK_RATIO     = 0.35      # ratio de frames "œil lent" sur fenêtre courte → fatigué
SLOW_BLINK_WINDOW    = 60        # nb de frames pour le ratio lent
EAR_SMOOTH_N         = 5         # moyenne mobile EAR

# ── Posture ───────────────────────────────────────────────────────────────────
# Épaules : différence de hauteur Y (normalisé 0-1, Y croît vers le bas)
SHOULDER_TILT_THRESHOLD  = 0.055   # différence Y épaules → déséquilibre
# Tête basse : ratio distance nez→ligne épaules / largeur épaules
HEAD_LOW_RATIO           = 0.25    # en dessous → tête trop basse / courbé
# Lissage temporel pour éviter les faux positifs
POSTURE_BAD_DURATION     = 2.5     # s de mauvaise posture continue → angry
POSTURE_GOOD_DURATION    = 3.0     # s de bonne posture continue → retour ok

# ── Fatigue : durée avant changement d'état ───────────────────────────────────
TIRED_DURATION           = 3.0     # s de signe de fatigue → tired
TIRED_RECOVER_DURATION   = 8.0     # s sans fatigue → retour ok

# ── MediaPipe ─────────────────────────────────────────────────────────────────
DETECTION_CONFIDENCE = 0.55
PRESENCE_CONFIDENCE  = 0.55
TRACKING_CONFIDENCE  = 0.55

FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ── Pose (BlazePose via ancienne API pour la simplicité) ─────────────────────
# On utilise mp.solutions.pose (API legacy, toujours incluse dans mediapipe)

# ──────────────────────────────────────────────────────────────────────────────
# INDICES LANDMARKS MEDIAPIPE FACE MESH
# ──────────────────────────────────────────────────────────────────────────────

# EAR : 6 points par œil (contour vertical + horizontal)
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]

# Nez (tip)
NOSE_TIP = 4


# ──────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────

def download_model_if_needed(path, url):
    if not os.path.exists(path):
        print(f"[INFO] Téléchargement → {path}")
        urllib.request.urlretrieve(url, path)
        print("[INFO] Prêt.")


def write_state(state: str):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"state": state, "ts": time.time()}, f)
    except Exception as e:
        print(f"[WARN] Écriture state : {e}")


def lm(landmarks, idx, w, h):
    p = landmarks[idx]
    return p.x * w, p.y * h


def ear(landmarks, indices, w, h):
    """Eye Aspect Ratio sur 6 points."""
    pts = [np.array(lm(landmarks, i, w, h)) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def smooth(deq: deque) -> float:
    return sum(deq) / len(deq) if deq else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# DÉTECTEUR DE POSTURE (BlazePose)
# ──────────────────────────────────────────────────────────────────────────────

class PostureDetector:
    # Indices BlazePose
    L_SHOULDER = 11
    R_SHOULDER = 12
    NOSE       = 0

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def analyse(self, frame_rgb):
        """
        Retourne (is_bad_posture: bool, details: dict)
        """
        result = self.pose.process(frame_rgb)
        if not result.pose_landmarks:
            return False, {}

        lms = result.pose_landmarks.landmark
        h, w = 1, 1   # coordonnées normalisées suffisent ici

        ls = lms[self.L_SHOULDER]
        rs = lms[self.R_SHOULDER]
        ns = lms[self.NOSE]

        # Épaules visibles ?
        if ls.visibility < 0.4 or rs.visibility < 0.4:
            return False, {"visibility": "low"}

        # 1. Déséquilibre des épaules (axe Y, 0=haut)
        tilt = abs(ls.y - rs.y)

        # 2. Tête trop basse : distance nez → milieu des épaules
        mid_shoulder_y = (ls.y + rs.y) / 2.0
        shoulder_width = abs(ls.x - rs.x)
        head_height    = mid_shoulder_y - ns.y   # positif = nez au-dessus des épaules

        if shoulder_width < 1e-4:
            ratio = 99.0
        else:
            ratio = head_height / shoulder_width

        bad_tilt = tilt > SHOULDER_TILT_THRESHOLD
        bad_head = ratio < HEAD_LOW_RATIO

        details = {
            "tilt":      round(tilt, 4),
            "head_ratio": round(ratio, 3),
            "bad_tilt":  bad_tilt,
            "bad_head":  bad_head,
        }
        return (bad_tilt or bad_head), details

    def draw(self, frame, details: dict):
        if not details:
            return
        color = (0, 60, 255) if (details.get("bad_tilt") or details.get("bad_head")) \
                else (60, 200, 60)
        cv2.putText(frame,
                    f"Tilt={details.get('tilt', 0):.3f}  "
                    f"Head={details.get('head_ratio', 0):.2f}",
                    (10, FRAME_HEIGHT - 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def close(self):
        self.pose.close()


# ──────────────────────────────────────────────────────────────────────────────
# DÉTECTEUR DE FATIGUE (Face Landmarker)
# ──────────────────────────────────────────────────────────────────────────────

class FatigueDetector:
    def __init__(self, face_detector):
        self.detector      = face_detector
        self.ear_buffer    = deque(maxlen=EAR_SMOOTH_N)
        self.slow_buffer   = deque(maxlen=SLOW_BLINK_WINDOW)

        # Clignements
        self.blink_times   = deque()   # timestamps des clignements
        self.eye_was_closed = False
        self.eye_closed_since = None

    def analyse(self, mp_img, w, h):
        """
        Retourne (is_tired: bool, details: dict)
        """
        result = self.detector.detect(mp_img)
        if not result.face_landmarks or len(result.face_landmarks[0]) < 400:
            return False, {}

        lms = result.face_landmarks[0]
        now = time.time()

        ear_r = ear(lms, RIGHT_EYE_EAR, w, h)
        ear_l = ear(lms, LEFT_EYE_EAR,  w, h)
        avg_ear = (ear_r + ear_l) / 2.0

        self.ear_buffer.append(avg_ear)
        smooth_ear = smooth(self.ear_buffer)

        # ── Détection clignement ──────────────────────────────────────────────
        eye_closed = smooth_ear < EAR_CLOSED_THRESHOLD

        if eye_closed and not self.eye_was_closed:
            # Début fermeture
            self.eye_closed_since = now
        elif not eye_closed and self.eye_was_closed:
            # Fin fermeture = un clignement
            self.blink_times.append(now)

        self.eye_was_closed = eye_closed

        # Nettoyer les vieux clignements
        cutoff = now - BLINK_WINDOW
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()

        blink_rate = len(self.blink_times) / (BLINK_WINDOW / 60.0)  # par minute

        # ── Critères fatigue ──────────────────────────────────────────────────
        # 1. Yeux fermés trop longtemps
        long_closure = (eye_closed and self.eye_closed_since is not None
                        and (now - self.eye_closed_since) >= CLOSED_DURATION_MAX)

        # 2. EAR moyen faible (clignements lents / yeux mi-clos)
        self.slow_buffer.append(1 if smooth_ear < EAR_SLOW_THRESHOLD else 0)
        slow_ratio = sum(self.slow_buffer) / len(self.slow_buffer) if self.slow_buffer else 0

        # 3. Taux de clignement élevé (fatigue → hyperclignement)
        high_blink_rate = blink_rate > BLINK_RATE_TIRED

        tired = (long_closure
                 or slow_ratio > SLOW_BLINK_RATIO
                 or high_blink_rate)

        details = {
            "ear":          round(smooth_ear, 3),
            "slow_ratio":   round(slow_ratio, 2),
            "blink_rate":   round(blink_rate, 1),
            "long_closure": long_closure,
        }
        return tired, details

    def draw(self, frame, details: dict):
        if not details:
            return
        tired = (details.get("long_closure")
                 or details.get("slow_ratio", 0) > SLOW_BLINK_RATIO
                 or details.get("blink_rate", 0) > BLINK_RATE_TIRED)
        color = (0, 60, 255) if tired else (60, 200, 60)
        cv2.putText(frame,
                    f"EAR={details.get('ear', 0):.3f}  "
                    f"Slow={details.get('slow_ratio', 0):.2f}  "
                    f"Blinks={details.get('blink_rate', 0):.0f}/min",
                    (10, FRAME_HEIGHT - 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ──────────────────────────────────────────────────────────────────────────────
# MACHINE À ÉTATS (lissage temporel)
# ──────────────────────────────────────────────────────────────────────────────

class StateMachine:
    """
    Évite les changements d'état trop rapides.
    Un état ne change que si le signal est maintenu pendant une durée minimale.
    """
    def __init__(self):
        self.current        = "ok"
        self.candidate      = "ok"
        self.candidate_since = None

        # Durées de confirmation (s)
        self.confirm_durations = {
            "ok":    max(POSTURE_GOOD_DURATION, TIRED_RECOVER_DURATION),
            "tired": TIRED_DURATION,
            "angry": POSTURE_BAD_DURATION,
        }

    def update(self, raw_state: str) -> str:
        """Propose un état brut, retourne l'état lissé."""
        now = time.time()

        if raw_state != self.candidate:
            self.candidate       = raw_state
            self.candidate_since = now

        if self.candidate != self.current:
            elapsed = now - (self.candidate_since or now)
            needed  = self.confirm_durations.get(self.candidate, 2.0)
            if elapsed >= needed:
                print(f"[STATE] {self.current} → {self.candidate}")
                self.current         = self.candidate
                self.candidate_since = now

        return self.current

    @property
    def progress(self) -> float:
        """Progression (0-1) vers le prochain changement d'état."""
        if self.candidate == self.current:
            return 1.0
        now     = time.time()
        elapsed = now - (self.candidate_since or now)
        needed  = self.confirm_durations.get(self.candidate, 2.0)
        return min(elapsed / needed, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    download_model_if_needed(FACE_MODEL_PATH, FACE_MODEL_URL)

    # Face Landmarker
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=DETECTION_CONFIDENCE,
        min_face_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    face_detector = mp_vision.FaceLandmarker.create_from_options(face_opts)

    posture_det = PostureDetector()
    fatigue_det = FatigueDetector(face_detector)
    state_machine = StateMachine()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Caméra {CAMERA_INDEX} inaccessible.")

    # État initial
    write_state("ok")

    show_debug  = True
    last_state  = "ok"

    print("[INFO] Démarré — 'p' debug  |  'q' quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Posture ───────────────────────────────────────────────────────────
        bad_posture, posture_details = posture_det.analyse(rgb)

        # ── Fatigue ───────────────────────────────────────────────────────────
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        tired, fatigue_details = fatigue_det.analyse(mp_img, w, h)

        # ── État brut (priorité angry > tired > ok) ───────────────────────────
        if bad_posture:
            raw = "angry"
        elif tired:
            raw = "tired"
        else:
            raw = "ok"

        # ── Lissage temporel ──────────────────────────────────────────────────
        current_state = state_machine.update(raw)

        # ── Écriture fichier ──────────────────────────────────────────────────
        if current_state != last_state:
            write_state(current_state)
            last_state = current_state

        # ── HUD ───────────────────────────────────────────────────────────────
        if show_debug:
            STATE_COLORS = {
                "ok":    (60, 200, 60),
                "tired": (180, 100, 255),
                "angry": (0, 60, 255),
            }
            col = STATE_COLORS.get(current_state, (200, 200, 200))

            # Bande top
            cv2.rectangle(frame, (0, 0), (w, 42), (18, 18, 18), -1)
            cv2.putText(frame, f"Banpei : {current_state.upper()}",
                        (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

            # Candidat et progression
            if state_machine.candidate != current_state:
                prog  = state_machine.progress
                bar_w = int(prog * w)
                cv2.rectangle(frame, (0, h - 8), (w, h),     (50, 50, 50), -1)
                cv2.rectangle(frame, (0, h - 8), (bar_w, h), col,          -1)
                cv2.putText(frame,
                            f"→ {state_machine.candidate}  {prog*100:.0f}%",
                            (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

            # Détails
            posture_det.draw(frame, posture_details)
            fatigue_det.draw(frame, fatigue_details)

            # Légende
            cv2.putText(frame, "Posture ↑   Fatigue ↑",
                        (10, h - 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)

            cv2.putText(frame, f"[P] debug  [Q] quitter  |  raw={raw}",
                        (10, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 80), 1)

            # Draw pose landmarks (optionnel)
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                None,   # on dessine via pose_detector, pas ici
            )

        cv2.imshow("Banpei Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()
    posture_det.close()
    write_state("ok")
    print("[INFO] Arrêt.")


if __name__ == "__main__":
    main()