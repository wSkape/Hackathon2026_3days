"""
banpei.py
=========
Always-on-top window that displays the Banpei avatar.
Reads state from banpei_state.json (written by banpei_monitor.py).

Dependencies:
    pip install pillow

Expected images in the same folder:
    banpei_ok.png
    banpei_tired.png
    banpei_angry.png
"""

import tkinter as tk
from tkinter import ttk
import json
import os
import sys
import time
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    print("[ERREUR] Pillow requis : pip install pillow")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

STATE_FILE    = "banpei_state.json"   # shared file with banpei_monitor.py
POLL_INTERVAL = 300                   # ms between file polls
AVATAR_SIZE   = (160, 160)            # avatar display size
WINDOW_W      = 180
WINDOW_H      = 210
# CORNER_RADIUS is simulated by drawing arcs on a Canvas
CORNER_RADIUS = 16                    # visual rounding (simulated)
BG_COLOR      = "#1a1a2e"
ACCENT_COLORS = {
    "ok":     "#00e5a0",
    "tired":  "#7b68ee",
    "angry":  "#ff4757",
}
LABEL_TEXTS = {
    "ok":     "Tout va bien !",
    "tired":  "Tu sembles fatigué…",
    "angry":  "Redresse-toi !",
}

IMAGES = {
    "ok":    "banpei_ok.png",
    "tired": "banpei_tired.png",
    "angry": "banpei_angry.png",
}

# Initial position (bottom-right corner, adjusted at startup)
MARGIN = 20

# ──────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

class BanpeiApp:
    def __init__(self, root: tk.Tk):
        self.root       = root
        self.state      = "ok"
        self.prev_state = None
        self._drag_x    = 0
        self._drag_y    = 0

        self._setup_window()
        self._load_images()
        self._build_ui()
        self._position_window()
        self._poll()

    # ── Window setup

    def _setup_window(self):
        root = self.root
        root.title("Banpei")
        root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        root.resizable(False, False)
        root.overrideredirect(True)          # remove title bar
        root.attributes("-topmost", True)    # always on top
        root.attributes("-alpha", 0.95)
        root.configure(bg=BG_COLOR)

        # Try to enable transparent color where supported (Windows/macOS)
        try:
            root.attributes("-transparentcolor", "")
        except Exception:
            pass

        # Bind mouse events for drag-to-move behavior
        root.bind("<ButtonPress-1>",   self._on_drag_start)
        root.bind("<B1-Motion>",       self._on_drag_motion)
        root.bind("<Double-Button-1>", self._on_double_click)

    def _position_window(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = sw - WINDOW_W - MARGIN
        y  = sh - WINDOW_H - MARGIN - 48   # 48 ≈ hauteur barre des tâches
        self.root.geometry(f"+{x}+{y}")

    # ── Images ────────────────────────────────────────────────────────────────

    def _load_images(self):
        self.photos = {}
        base = Path(__file__).parent
        for state, filename in IMAGES.items():
            path = base / filename
            if path.exists():
                img = Image.open(path).convert("RGBA")
                img = img.resize(AVATAR_SIZE, Image.LANCZOS)
                self.photos[state] = ImageTk.PhotoImage(img)
            else:
                # Colored placeholder if image is missing
                placeholder = Image.new("RGBA", AVATAR_SIZE,
                                        (*self._hex_to_rgb(ACCENT_COLORS[state]), 60))
                self.photos[state] = ImageTk.PhotoImage(placeholder)
                print(f"[WARN] Image manquante : {path}")

    @staticmethod
    def _hex_to_rgb(hex_color: str):
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Colored border drawn on a Canvas to simulate rounded corners
        self.border_canvas = tk.Canvas(
            self.root, width=WINDOW_W, height=WINDOW_H,
            bg=BG_COLOR, highlightthickness=0
        )
        self.border_canvas.place(x=0, y=0)

        self._draw_border("ok")

        # Avatar image label
        self.avatar_label = tk.Label(
            self.root, image=self.photos["ok"],
            bg=BG_COLOR, cursor="hand2"
        )
        self.avatar_label.place(
            x=(WINDOW_W - AVATAR_SIZE[0]) // 2,
            y=14
        )

        # State text label
        self.state_label = tk.Label(
            self.root,
            text=LABEL_TEXTS["ok"],
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT_COLORS["ok"],
            bg=BG_COLOR,
            wraplength=WINDOW_W - 16,
            justify="center",
        )
        self.state_label.place(
            x=0, y=AVATAR_SIZE[1] + 22,
            width=WINDOW_W
        )

        # Close button (×)
        close_btn = tk.Label(
            self.root, text="×", font=("Segoe UI", 13),
            fg="#555577", bg=BG_COLOR, cursor="hand2"
        )
        close_btn.place(x=WINDOW_W - 22, y=4)
        close_btn.bind("<Button-1>", lambda e: self.root.destroy())
        close_btn.bind("<Enter>",    lambda e: close_btn.config(fg="#ff4757"))
        close_btn.bind("<Leave>",    lambda e: close_btn.config(fg="#555577"))

        # Live indicator dot
        self.dot = tk.Label(self.root, text="●", font=("Segoe UI", 7),
                            fg="#00e5a0", bg=BG_COLOR)
        self.dot.place(x=8, y=8)

    def _draw_border(self, state: str):
        c = self.border_canvas
        c.delete("border")
        color = ACCENT_COLORS[state]
        r = CORNER_RADIUS
        w, h = WINDOW_W, WINDOW_H
        # Rectangle arrondi
        c.create_arc((0, 0, r*2, r*2),         start=90,  extent=90,  outline=color, width=2, style="arc", tags="border")
        c.create_arc((w-r*2, 0, w, r*2),        start=0,   extent=90,  outline=color, width=2, style="arc", tags="border")
        c.create_arc((0, h-r*2, r*2, h),         start=180, extent=90,  outline=color, width=2, style="arc", tags="border")
        c.create_arc((w-r*2, h-r*2, w, h),       start=270, extent=90,  outline=color, width=2, style="arc", tags="border")
        c.create_line(r, 0, w-r, 0,         fill=color, width=2, tags="border")
        c.create_line(r, h, w-r, h,         fill=color, width=2, tags="border")
        c.create_line(0, r, 0, h-r,         fill=color, width=2, tags="border")
        c.create_line(w, r, w, h-r,         fill=color, width=2, tags="border")

    # ── Animated transition

    def _transition_to(self, new_state: str):
        """Cross-fade / UI update between two states."""
        if new_state == self.state:
            return
        self.state = new_state

        self.avatar_label.config(image=self.photos[new_state])
        self.state_label.config(
            text=LABEL_TEXTS[new_state],
            fg=ACCENT_COLORS[new_state],
        )
        self._draw_border(new_state)

        # Flash the live indicator dot briefly
        self.dot.config(fg=ACCENT_COLORS[new_state])
        self.root.after(800, lambda: self.dot.config(fg="#00e5a0"))

        # Small bounce animation (move Y)
        self._bounce(0)

    def _bounce(self, step: int):
        STEPS = [(-3, 40), (-5, 40), (-3, 40), (0, 40), (3, 40), (5, 40),
                 (3, 40), (0, 0)]
        if step >= len(STEPS):
            return
        dy, delay = STEPS[step]
        try:
            geo = self.root.geometry()          # WxH+X+Y
            parts = geo.replace("-", "+-").split("+")
            x, y  = int(parts[1]), int(parts[2])
            self.root.geometry(f"+{x}+{y + dy}")
            self.root.after(delay, lambda: self._bounce(step + 1))
        except Exception:
            pass

    # ── State polling

    def _poll(self):
        try:
            # Read shared JSON state file and update UI when it changes
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                new_state = data.get("state", "ok")
                if new_state in IMAGES and new_state != self.state:
                    self._transition_to(new_state)
        except Exception:
            # Ignore transient read/parse errors (file may be written concurrently)
            pass
        # Schedule next poll
        self.root.after(POLL_INTERVAL, self._poll)

    # ── Drag ──────────────────────────────────────────────────────────────────

    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag_motion(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _on_double_click(self, event):
        """Double-click: force an immediate state re-read."""
        self._poll()


# ──────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app  = BanpeiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()