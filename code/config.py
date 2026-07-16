
"""
All constants for Banpei
"""

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# ── Media pause (gaze) ──────────────────────────────────────────────────────
LOOK_AWAY_THRESHOLD = 2.0
RESUME_DELAY        = 0.3

# ── Smart Lock ────────────────────────────────────────────────────────────────
SMART_LOCK_ENABLED_DEFAULT   = True
ABSENT_LOCK_DELAY            = 10.0   # s d'absence → verrou OS
ABSENT_MEDIA_DELAY           = 2.0    # seconds of absence → media pause
ABSENT_WARNING_BEFORE        = 3.0    # s avant verrou → avertissement

# ── Auto-play after smart lock ───────────────────────────────────────────────
UNLOCK_RESUME_DELAY = 2.0

# ── Privacy Shield ────────────────────────────────────────────────────────────
PRIVACY_SHIELD_ENABLED_DEFAULT = True
PRIVACY_PERSONS_THRESHOLD      = 2     # number of people to trigger
PRIVACY_TRIGGER_DELAY          = 1.0   # seconds with ≥2 people before triggering
PRIVACY_RESTORE_DELAY          = 2.0   # seconds with <2 people before restoring

# ── Banpei ────────────────────────────────────────────────────────────────────
# BANPEI defaults
BANPEI_ENABLED_DEFAULT = True    # False = disabled even if banpei_core is present
BANPEI_DEBUG_DEFAULT   = False   # show metrics in the OpenCV window

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_MODEL             = "yolo11n.pt"
COCO_PERSON_CLASS      = 0
YOLO_PERSON_CONFIDENCE = 0.45
YOLO_EVERY_N_FRAMES    = 3

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Head / iris thresholds ───────────────────────────────────────────────────
HEAD_YAW_MAX_DEG   = 55
HEAD_PITCH_MAX_DEG = 45
IRIS_H_THRESHOLD   = 0.38
IRIS_V_THRESHOLD   = 0.42
EAR_MIN            = 0.15

# ── MediaPipe ─────────────────────────────────────────────────────────────────
DETECTION_CONFIDENCE = 0.50
PRESENCE_CONFIDENCE  = 0.50
TRACKING_CONFIDENCE  = 0.50

# ── Fatigue (EAR) ─────────────────────────────────────────────────────────────
EAR_CLOSED_THRESHOLD = 0.20
EAR_SLOW_THRESHOLD   = 0.23
CLOSED_DURATION_MAX  = 0.8
BLINK_WINDOW         = 20.0
BLINK_RATE_TIRED     = 22
SLOW_BLINK_RATIO     = 0.35
SLOW_BLINK_WINDOW    = 60
EAR_SMOOTH_N         = 5

# ── Posture ───────────────────────────────────────────────────────────────────
SHOULDER_TILT_THRESHOLD = 0.055
HEAD_LOW_RATIO          = 0.25
POSTURE_BAD_DURATION    = 2.5
POSTURE_GOOD_DURATION   = 3.0

# ── Fatigue : duration ──────────────────────────────────────────────────────────
TIRED_DURATION         = 3.0
TIRED_RECOVER_DURATION = 8.0

# ── EAR indices (Face Mesh 478 pts) ───────────────────────────────────────────
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]