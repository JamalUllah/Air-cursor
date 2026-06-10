"""
hand_cursor.py  —  Dual-hand gesture mouse controller
======================================================
Compatible with mediapipe 0.10.30+ (Tasks API).

RIGHT HAND — cursor control only
─────────────────────────────────
  Fist (all curled)        →  IDLE: cursor freezes, reposition freely
  Index finger up          →  MOVE: cursor follows index fingertip
  Index+Middle+Ring up     →  SCROLL: move hand up/down to scroll

LEFT HAND — all clicks
──────────────────────
  Index+Thumb snap (quick) →  LEFT CLICK
  Index+Thumb held         →  DRAG (hold mouse down, move right hand)
  Middle+Thumb snap        →  RIGHT CLICK

SENSITIVITY
───────────
  Increased: MARGIN expanded to 0.15 for easier corner reach,
  EMA alpha raised to 0.65, and SCROLL_SPEED raised significantly.

GUI FREEZE FIX
──────────────
  Preview window runs in a separate thread so OpenCV imshow never
  blocks the main detection loop — no more freezing when you move
  the window.

Run:   python3 hand_cursor.py
Quit:  press Q in the preview window
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time, math, os, urllib.request, threading
from collections import deque
from enum import Enum, auto

# ── Safety ────────────────────────────────────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# ── Tuning ────────────────────────────────────────────────────────────────────
CAM_INDEX         = 0
MOVE_SMOOTH       = 0.65    # higher = more responsive (was 0.35)
PINCH_DIST        = 0.058   # normalised distance: fingers "touching"
PINCH_OPEN        = 0.11    # normalised distance: fingers "released"
CLICK_COOLDOWN    = 0.38    # seconds min between clicks
HOLD_TIME         = 0.12    # seconds pinch must be held before drag starts
SCROLL_SPEED      = 100     # scroll units (increased significantly)
MARGIN            = 0.15    # edge dead-zone — larger = easier to reach corners

# ── Landmark IDs ──────────────────────────────────────────────────────────────
WRIST=0; THUMB_TIP=4; THUMB_IP=3; THUMB_MCP=2
INDEX_TIP=8;  INDEX_MCP=5
MIDDLE_TIP=12; MIDDLE_MCP=9
RING_TIP=16;   RING_MCP=13
PINKY_TIP=20;  PINKY_MCP=17

# ── Skeleton connections ───────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
TIP_IDS    = {4, 8, 12, 16, 20}
TIP_LABELS = {4:"THUMB", 8:"INDEX", 12:"MID", 16:"RING", 20:"PINKY"}


# ═══════════════════════════════════════════════════════════════════════════════
# States
# ═══════════════════════════════════════════════════════════════════════════════

class RState(Enum):
    """Right hand states."""
    IDLE   = auto()   # fist — cursor frozen
    MOVE   = auto()   # index up — move cursor
    SCROLL = auto()   # 3 fingers up — scroll

class LState(Enum):
    """Left hand states."""
    IDLE        = auto()   # nothing / neutral
    LCLICK      = auto()   # index+thumb pinch → left click
    LHOLD       = auto()   # index+thumb held  → drag
    RCLICK      = auto()   # middle+thumb snap → right click


# BGR colours
RCOL = {
    RState.IDLE:   (100, 100, 100),
    RState.MOVE:   (50,  220,  50),
    RState.SCROLL: (200,  80, 255),
}
LCOL = {
    LState.IDLE:   (100, 100, 100),
    LState.LCLICK: (0,   200, 255),
    LState.LHOLD:  (0,    80, 255),
    LState.RCLICK: (80,   80, 255),
}

RLABEL = {
    RState.IDLE:   "RIGHT: IDLE (fist)",
    RState.MOVE:   "RIGHT: MOVE",
    RState.SCROLL: "RIGHT: SCROLL",
}
LLABEL = {
    LState.IDLE:   "LEFT: idle",
    LState.LCLICK: "LEFT: LEFT CLICK",
    LState.LHOLD:  "LEFT: DRAG HOLD",
    LState.RCLICK: "LEFT: RIGHT CLICK",
}

# Active tips per state (for highlighting)
R_ACTIVE = {
    RState.IDLE:   set(),
    RState.MOVE:   {8},
    RState.SCROLL: {8, 12, 16},
}
L_ACTIVE = {
    LState.IDLE:   set(),
    LState.LCLICK: {8, 4},
    LState.LHOLD:  {8, 4},
    LState.RCLICK: {12, 4},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class LM:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z): self.x=x; self.y=y; self.z=z

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def finger_up(lm, tip, mcp):
    return lm[tip].y < lm[mcp].y

def lm_px(lm, i, w, h):
    return int(lm[i].x * w), int(lm[i].y * h)

def map_screen(x, y):
    xc = (x - MARGIN) / (1 - 2 * MARGIN)
    yc = (y - MARGIN) / (1 - 2 * MARGIN)
    return (int(np.clip(xc, 0, 1) * SCREEN_W),
            int(np.clip(yc, 0, 1) * SCREEN_H))


class EMA:
    def __init__(self, alpha=MOVE_SMOOTH):
        self.alpha = alpha
        self.x = self.y = None

    def update(self, x, y):
        if self.x is None:
            self.x, self.y = float(x), float(y)
        else:
            self.x += self.alpha * (x - self.x)
            self.y += self.alpha * (y - self.y)
        return int(self.x), int(self.y)

    def reset(self):
        self.x = self.y = None


# ═══════════════════════════════════════════════════════════════════════════════
# Gesture classifiers  (history-smoothed with majority vote)
# ═══════════════════════════════════════════════════════════════════════════════

class RightClassifier:
    HISTORY = 4

    def __init__(self):
        self.buf = deque(maxlen=self.HISTORY)

    def _raw(self, lm):
        idx   = finger_up(lm, INDEX_TIP,  INDEX_MCP)
        mid   = finger_up(lm, MIDDLE_TIP, MIDDLE_MCP)
        ring  = finger_up(lm, RING_TIP,   RING_MCP)
        pinky = finger_up(lm, PINKY_TIP,  PINKY_MCP)

        # Scroll: index + middle + ring up
        if idx and mid and ring:
            return RState.SCROLL
        # Move: index only
        if idx and not mid and not ring and not pinky:
            return RState.MOVE
        # Everything else (fist, unknown) → IDLE
        return RState.IDLE

    def classify(self, lm):
        self.buf.append(self._raw(lm))
        counts = {}
        for s in self.buf:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.get)


class LeftClassifier:
    HISTORY = 4

    def __init__(self):
        self.buf = deque(maxlen=self.HISTORY)

    def _raw(self, lm):
        d_it = dist(lm[INDEX_TIP],  lm[THUMB_TIP])   # index–thumb
        d_mt = dist(lm[MIDDLE_TIP], lm[THUMB_TIP])   # middle–thumb

        # Right click: middle + thumb pinch
        if d_mt < PINCH_DIST:
            return LState.RCLICK

        # Left click / hold: index + thumb pinch
        if d_it < PINCH_DIST:
            return LState.LCLICK   # caller upgrades to LHOLD based on time

        return LState.IDLE

    def classify(self, lm):
        self.buf.append(self._raw(lm))
        counts = {}
        for s in self.buf:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.get)


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hand(frame, lm, colour, active_tips, w, h, label=""):
    """Draw skeleton + coloured tips for one hand."""
    up_map = {
        4:  lm[4].y  < lm[3].y,
        8:  lm[8].y  < lm[5].y,
        12: lm[12].y < lm[9].y,
        16: lm[16].y < lm[13].y,
        20: lm[20].y < lm[17].y,
    }

    # Skeleton lines
    for a, b in CONNECTIONS:
        pa = lm_px(lm, a, w, h)
        pb = lm_px(lm, b, w, h)
        hot = a in active_tips or b in active_tips
        cv2.line(frame, pa, pb,
                 colour if hot else (70, 70, 70),
                 3 if hot else 1, cv2.LINE_AA)

    # Dots
    for i in range(21):
        px = lm_px(lm, i, w, h)
        if i not in TIP_IDS:
            cv2.circle(frame, px, 4, (160, 160, 160), 1, cv2.LINE_AA)
            continue
        if i in active_tips:
            cv2.circle(frame, px, 13, colour,        -1, cv2.LINE_AA)
            cv2.circle(frame, px, 13, (255,255,255),  1, cv2.LINE_AA)
        elif up_map.get(i):
            cv2.circle(frame, px, 9,  (40, 200, 60), -1, cv2.LINE_AA)
            cv2.circle(frame, px, 9,  (180,255,180),  1, cv2.LINE_AA)
        else:
            cv2.circle(frame, px, 5,  (180,180,180), -1, cv2.LINE_AA)

    # Pinch line between active pair
    if len(active_tips) == 2:
        tips = list(active_tips)
        p1 = lm_px(lm, tips[0], w, h)
        p2 = lm_px(lm, tips[1], w, h)
        cv2.line(frame, p1, p2, colour, 2, cv2.LINE_AA)
        for p in [p1, p2]:
            cv2.circle(frame, p, 17, colour, 1, cv2.LINE_AA)

    # Cursor cross on index tip when moving
    if 8 in active_tips and len(active_tips) == 1:
        ix = lm_px(lm, INDEX_TIP, w, h)
        cv2.drawMarker(frame, ix, (255,255,255),
                       cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)

    # Tip labels
    for tid, name in TIP_LABELS.items():
        px = lm_px(lm, tid, w, h)
        col = colour if tid in active_tips else (120, 120, 120)
        cv2.putText(frame, name, (px[0]+8, px[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)

    # Hand label near wrist
    if label:
        wp = lm_px(lm, WRIST, w, h)
        cv2.circle(frame, wp, 5, (180,180,180), -1, cv2.LINE_AA)
        cv2.putText(frame, label, (wp[0]+8, wp[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def draw_hud(frame, rstate, lstate, dragging, scroll_dir, w, h):
    """Top HUD bar + bottom cheat-sheet."""
    # Semi-transparent top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w, 54), (12,12,12), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    # Right hand status
    rc = RCOL.get(rstate, (120,120,120))
    cv2.putText(frame, RLABEL.get(rstate,""), (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, rc, 2, cv2.LINE_AA)

    # Left hand status
    lc = LCOL.get(lstate, (120,120,120))
    cv2.putText(frame, LLABEL.get(lstate,""), (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, lc, 2, cv2.LINE_AA)

    # DRAGGING badge
    if dragging:
        bx = w - 175
        cv2.rectangle(frame, (bx,6),(w-6,50),(0,40,180),-1)
        cv2.rectangle(frame, (bx,6),(w-6,50),(80,130,255),1)
        cv2.putText(frame, "DRAGGING", (bx+6,38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (160,210,255), 2, cv2.LINE_AA)

    # Scroll indicator
    if scroll_dir:
        arrow = "SCROLL ▲" if scroll_dir == 'up' else "SCROLL ▼"
        cv2.putText(frame, arrow, (w//2 - 70, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200,80,255), 2, cv2.LINE_AA)

    # Bottom cheat-sheet
    ph = 138
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h-ph), (360, h), (12,12,12), -1)
    cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)

    guide = [
        ("RIGHT fist",           "Freeze cursor",          (100,100,100)),
        ("RIGHT index up",       "Move cursor",            (50, 220, 50)),
        ("RIGHT 3 fingers up",   "Scroll up/down",         (200, 80,255)),
        ("LEFT index+thumb snap","Left click",             (0,  200,255)),
        ("LEFT index+thumb hold","Drag (hold)",            (0,   80,255)),
        ("LEFT middle+thumb",    "Right click",            (80,  80,255)),
    ]
    y0 = h - ph + 18
    for g, a, col in guide:
        cv2.putText(frame, f"{g}: {a}", (8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1, cv2.LINE_AA)
        y0 += 20


# ═══════════════════════════════════════════════════════════════════════════════
# Model download
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_model():
    if os.path.exists(MODEL_PATH): return
    print(f"Downloading hand landmarker model (~9 MB) ...\n  → {MODEL_PATH}\n")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.\n")
    except Exception as e:
        print(f"ERROR: {e}\nDownload manually:\n  {MODEL_URL}")
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Display thread  ← fixes the GUI freeze problem
# ═══════════════════════════════════════════════════════════════════════════════

class DisplayThread(threading.Thread):
    """
    Runs cv2.imshow + waitKey in its own thread so moving the window
    never blocks the main detection loop.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self._lock  = threading.Lock()
        self._frame = None
        self._quit  = False
        self.want_quit = False

    def update(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def run(self):
        cv2.namedWindow("Hand Cursor  [Q to quit]", cv2.WINDOW_NORMAL)
        while not self._quit:
            with self._lock:
                frame = self._frame
            if frame is not None:
                cv2.imshow("Hand Cursor  [Q to quit]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.want_quit = True
                self._quit = True
        cv2.destroyAllWindows()

    def stop(self):
        self._quit = True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_model()

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode           = mp.tasks.vision.RunningMode

    # Detect up to 2 hands
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.50,
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ema            = EMA()
    r_classifier   = RightClassifier()
    l_classifier   = LeftClassifier()
    frame_ts       = 0

    # Mouse state
    dragging        = False
    l_pinch_open    = True    # True when left index+thumb is open
    r_pinch_open    = True    # True when left middle+thumb is open
    lclick_armed    = True
    rclick_armed    = True
    last_click_t    = 0.0
    pinch_start_t   = None   # when the left pinch started (for hold detection)
    prev_scroll_y   = None
    scroll_dir      = None

    # Start display thread
    display = DisplayThread()
    display.start()

    print("═" * 58)
    print("  Dual-Hand Cursor — active")
    print("═" * 58)
    print("  RIGHT hand controls cursor:")
    print("    Fist           → freeze cursor (reposition freely)")
    print("    Index finger   → move cursor")
    print("    3 fingers up   → scroll (move hand up/down)")
    print()
    print("  LEFT hand fires clicks:")
    print("    Index+Thumb snap  → LEFT CLICK")
    print("    Index+Thumb hold  → DRAG (hold mouse button)")
    print("    Middle+Thumb snap → RIGHT CLICK")
    print("═" * 58)
    print("  Press Q in the preview window to quit\n")

    with HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened() and not display.want_quit:
            ok, frame = cap.read()
            if not ok:
                break

            frame    = cv2.flip(frame, 1)
            h, w, _  = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33
            result   = detector.detect_for_video(mp_img, frame_ts)

            rstate     = RState.IDLE
            lstate     = LState.IDLE
            scroll_dir = None
            r_lm       = None
            l_lm       = None

            # ── Separate hands by handedness ──────────────────────────────
            if result.hand_landmarks and result.handedness:
                for lm_list, handed in zip(result.hand_landmarks,
                                           result.handedness):
                    label = handed[0].category_name
                    lm = [LM(p.x, p.y, p.z) for p in lm_list]
                    # Mirrored frame: "Left" label = physical right hand
                    if label == "Left":
                        r_lm = lm
                    else:
                        l_lm = lm

            # ── RIGHT HAND: cursor movement ───────────────────────────────
            if r_lm:
                rstate = r_classifier.classify(r_lm)

                sx, sy = map_screen(r_lm[INDEX_TIP].x, r_lm[INDEX_TIP].y)
                sx, sy = ema.update(sx, sy)

                if rstate == RState.IDLE:
                    # Cursor freezes — good for repositioning hand
                    pass

                elif rstate == RState.MOVE:
                    pyautogui.moveTo(sx, sy, _pause=False)

                elif rstate == RState.SCROLL:
                    cur_y = r_lm[INDEX_TIP].y
                    if prev_scroll_y is not None:
                        delta  = cur_y - prev_scroll_y
                        clicks = int(delta * SCROLL_SPEED * 14)
                        if clicks != 0:
                            pyautogui.scroll(-clicks)
                            scroll_dir = 'down' if clicks > 0 else 'up'
                    prev_scroll_y = cur_y

                if rstate != RState.SCROLL:
                    prev_scroll_y = None

                draw_hand(frame, r_lm,
                          RCOL[rstate],
                          R_ACTIVE[rstate],
                          w, h, "RIGHT")
            else:
                # No right hand → reset EMA so cursor doesn't jump on reappearance
                ema.reset()

            # ── LEFT HAND: clicks ─────────────────────────────────────────
            if l_lm:
                raw_l = l_classifier.classify(l_lm)
                now   = time.time()

                # ── Index+Thumb pinch → left click or drag ─────────────────
                if raw_l == LState.LCLICK:
                    lstate = LState.LCLICK

                    if l_pinch_open:
                        # Pinch just closed — start timer
                        pinch_start_t = now
                        l_pinch_open  = False

                    held = now - (pinch_start_t or now)

                    if held >= HOLD_TIME and not dragging:
                        # Held long enough → start drag
                        pyautogui.mouseDown()
                        dragging = True
                        lstate   = LState.LHOLD

                    elif dragging:
                        lstate = LState.LHOLD

                else:
                    # Pinch released
                    if not l_pinch_open:
                        # Was pinching — fire click if it was a short tap
                        held = now - (pinch_start_t or now)
                        if held < HOLD_TIME and not dragging:
                            if lclick_armed and (now - last_click_t) > CLICK_COOLDOWN:
                                pyautogui.click()
                                last_click_t = now
                                lclick_armed = False
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False
                        pinch_start_t = None
                        l_pinch_open  = True
                        lclick_armed  = True

                # ── Middle+Thumb pinch → right click ───────────────────────
                if raw_l == LState.RCLICK:
                    lstate = LState.RCLICK
                    if rclick_armed and (now - last_click_t) > CLICK_COOLDOWN:
                        pyautogui.click(button='right')
                        last_click_t = now
                        rclick_armed = False
                else:
                    rclick_armed = True

                draw_hand(frame, l_lm,
                          LCOL.get(lstate, (100,100,100)),
                          L_ACTIVE.get(lstate, set()),
                          w, h, "LEFT")

            else:
                # No left hand — release any held state safely
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                l_pinch_open  = True
                pinch_start_t = None
                lclick_armed  = True
                rclick_armed  = True

            # ── Render ────────────────────────────────────────────────────
            draw_hud(frame, rstate, lstate, dragging, scroll_dir, w, h)
            display.update(frame)   # non-blocking — thread handles imshow

    # Cleanup
    if dragging:
        pyautogui.mouseUp()
    display.stop()
    cap.release()
    print("Hand cursor stopped.")


if __name__ == "__main__":
    main()
