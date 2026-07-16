# Banpei

A webcam-based desktop assistant built with MediaPipe, YOLO11, and OpenCV. It watches your posture, attention, and presence in real time and reacts automatically - no extra hardware required.

## Features

**Media Pause (gaze tracking) :** pauses your media when you look away, resumes when you look back.
**Smart Lock :** locks your session automatically when you've been absent too long (with a media pause as an intermediate step).
**Auto-Play :** resumes playback a few seconds after you return and unlock.
**Privacy Shield :** detects when 2+ people are in frame and instantly shows the desktop to hide sensitive content.
**Banpei :** a floating always-on-top avatar that reacts to your posture and fatigue (`ok` / `tired` / `angry`), running as a separate always-on-top window.

## Shortcut	Action
| :-----: | :-----: |
| `L` | Toggle Smart Lock|
| `P`	| Toggle Privacy Shield|
| `B`	| Toggle Banpei debug overlay|
| `Q` |	Quit|

## Quick start (Windows)

*Download/clone the repository.
*Double-click `launch.bat`

The launcher will :

*create `.venv` automatically (first run),
*install the latest dependencies from `requirements.txt`

On first launch, the app will also automatically download the required MediaPipe/YOLO model files (`face_landmarker.task`, `pose_landmarker_lite.task`, `yolo11n.pt`) into the project folder.

## Manual run (optional)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python gaze_media_controller.py

```

# Project structure

| :-----: | :-----: |
|gaze_media_controller.py   | # main loop: camera, YOLO, MediaPipe, feature logic, HUD|
|banpei_core.py             | # posture/fatigue detectors + state machine (imported by the controller)|
|banpei.py                  | # standalone always-on-top avatar window (launched automatically)|
|config.py                  | # tunable thresholds and constants|
|mp_constantes.py           | # MediaPipe landmark indices used for gaze/head-pose estimation|

## Requirements

*Windows (uses OS-level lock and Win+D shortcuts; partial support for macOS/Linux)
*A webcam
*Python 3.10+ (installed automatically via the launcher if using `launch.bat`)
