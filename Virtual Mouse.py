#!/usr/bin/env python3
"""
Virtual Mouse.py
Main script to map hand gestures to system mouse actions.

Usage:
  python "Virtual Mouse.py" [--no-mouse] [--debug] [--safe] [--cam N]

Note: file name includes a space; call it with quotes on some shells/windows.
"""

import argparse
import sys
import time
import traceback
from threading import Thread
from typing import Optional, Tuple

import cv2
import numpy as np

from HandTracking import HandDetector  # import matches file HandTracking.py

# Try mouse backends (optional)
MOUSE_BACKEND = None
try:
    import autopy

    MOUSE_BACKEND = "autopy"
except Exception:
    try:
        import pyautogui

        MOUSE_BACKEND = "pyautogui"
    except Exception:
        MOUSE_BACKEND = None

# Camera & mapping settings (tweakable)
CAM_INDEX = 0
W_CAM, H_CAM = 1280, 720
FRAME_MARGIN_DEFAULT = 100
SMOOTH_ALPHA = 0.75  # smoothing factor for cursor interpolation (0..1)
INERTIA = 0.6  # how much previous velocity influences next step
DRAG_THRESHOLD = 40
CLICK_COOLDOWN = 0.25

# Globals (controlled via CLI)
mirror = True
overlay = True
no_mouse = False

# small helper video grabber thread to reduce camera stalls
class VideoGet:
    def __init__(self, src=0, width=W_CAM, height=H_CAM):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stopped = False
        self.grabbed = False
        self.frame = None
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            try:
                grabbed, frame = self.cap.read()
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass


# mouse actions with optional backends and cooldown
_last_click = 0.0


def _now():
    return time.perf_counter()


def move_mouse(x: float, y: float):
    if no_mouse:
        return
    try:
        if MOUSE_BACKEND == "autopy":
            autopy.mouse.move(int(x), int(y))
        elif MOUSE_BACKEND == "pyautogui":
            pyautogui.moveTo(int(x), int(y))
    except Exception:
        pass


def _click(kind: str = "left"):
    global _last_click
    now = _now()
    if now - _last_click < CLICK_COOLDOWN:
        return
    _last_click = now
    if no_mouse:
        return
    try:
        if MOUSE_BACKEND == "autopy":
            if kind == "left":
                autopy.mouse.click()
            else:
                autopy.mouse.click(autopy.mouse.Button.RIGHT)
        elif MOUSE_BACKEND == "pyautogui":
            if kind == "left":
                pyautogui.click()
            else:
                pyautogui.click(button="right")
    except Exception:
        pass


def left_click():
    _click("left")


def right_click():
    _click("right")


def mouse_down():
    if no_mouse:
        return
    try:
        if MOUSE_BACKEND == "autopy":
            autopy.mouse.toggle(True)
        elif MOUSE_BACKEND == "pyautogui":
            pyautogui.mouseDown()
    except Exception:
        pass


def mouse_up():
    if no_mouse:
        return
    try:
        if MOUSE_BACKEND == "autopy":
            autopy.mouse.toggle(False)
        elif MOUSE_BACKEND == "pyautogui":
            pyautogui.mouseUp()
    except Exception:
        pass


def get_screen_size() -> Tuple[int, int]:
    try:
        import screeninfo

        m = screeninfo.get_monitors()[0]
        return m.width, m.height
    except Exception:
        try:
            import pyautogui

            return pyautogui.size()
        except Exception:
            return 1366, 768


def safe_len(x):
    return 0 if x is None else len(x)


def simple_calibrate(det: HandDetector, vg: VideoGet, samples: int = 20, previous_margin: int = FRAME_MARGIN_DEFAULT) -> int:
    print("Calibration: center an open hand, press s repeatedly to sample frames (or wait to auto-collect).")
    collected = []
    tries = 0
    max_tries = samples * 10
    while len(collected) < samples and tries < max_tries:
        tries += 1
        g, f = vg.read()
        if not g or f is None:
            time.sleep(0.01)
            continue
        frame = cv2.flip(f, 1)
        frame = det.findHands(frame, draw=False)
        lm, b = det.findPosition(frame, draw=False)
        if b and len(b) >= 4:
            xmin, ymin, xmax, ymax = b
            collected.append(xmax - xmin)
            cv2.putText(frame, f"Samples {len(collected)}/{samples}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("calibrate", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            pass

    cv2.destroyWindow("calibrate")
    if collected:
        med = int(np.median(collected))
        new_margin = max(40, min(240, med // 2))
        print("Calibration done ->", new_margin)
        return new_margin
    else:
        print("Calibration failed (no hand detected). Keeping old margin.")
        return previous_margin


def main():
    global mirror, overlay, no_mouse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mouse", action="store_true", help="demo: do not move system mouse")
    parser.add_argument("--debug", action="store_true", help="draw debug info (slower)")
    parser.add_argument("--safe", action="store_true", help="extra checks (slower, more stable)")
    parser.add_argument("--cam", type=int, default=CAM_INDEX, help="camera index")
    args = parser.parse_args()

    no_mouse = args.no_mouse
    det = HandDetector(
        maxHands=1,
        detectionCon=0.7,
        trackCon=0.6,
        model_complexity=0,
        smooth_alpha=0.6,
        process_every=2,
        debug=args.debug,
    )

    if MOUSE_BACKEND is None and not no_mouse:
        print("No mouse backend found; running demo (--no-mouse)")
        no_mouse = True

    vg = VideoGet(src=args.cam, width=W_CAM, height=H_CAM)
    screen_w, screen_h = get_screen_size()
    print("screen", screen_w, screen_h)

    time.sleep(0.2)  # allow camera warmup

    # local state for smoothing & velocity
    prev_x = prev_y = 0.0
    vel_x = vel_y = 0.0
    dragging = False
    frame_margin = FRAME_MARGIN_DEFAULT
    pTime = 0.0

    try:
        while True:
            try:
                grabbed, frame = vg.read()
                if not grabbed or frame is None:
                    time.sleep(0.002)
                    continue

                img = frame.copy()
                if mirror:
                    img = cv2.flip(img, 1)

                # process hands (may skip frames internally)
                img = det.findHands(img, draw=args.debug)
                lmList, bbox = det.findPosition(img, draw=args.debug)

                overlay_x = overlay_y = None
                hand_size = 0
                if bbox and safe_len(bbox) >= 4:
                    try:
                        xmin, ymin, xmax, ymax = bbox
                        hand_size = max(1, xmax - xmin)
                    except Exception:
                        hand_size = 0

                adaptive_drag = DRAG_THRESHOLD
                if hand_size:
                    try:
                        adaptive_drag = max(18, int(DRAG_THRESHOLD * (120 / max(1, hand_size))))
                    except Exception:
                        adaptive_drag = DRAG_THRESHOLD

                if lmList and len(lmList) > 12:
                    try:
                        x_idx, y_idx = lmList[8][1], lmList[8][2]
                        x_mid, y_mid = lmList[12][1], lmList[12][2]
                    except Exception:
                        if args.debug:
                            print("Landmarks malformed, skipping frame")
                        continue

                    fingers = det.fingersUp()

                    # move cursor: index up, middle down (index=1,middle=0)
                    if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 0:
                        # map camera coords -> screen coords
                        try:
                            x3 = np.interp(x_idx, (frame_margin, W_CAM - frame_margin), (0, screen_w))
                            y3 = np.interp(y_idx, (frame_margin, H_CAM - frame_margin), (0, screen_h))
                        except Exception:
                            x3 = (x_idx / float(W_CAM)) * screen_w
                            y3 = (y_idx / float(H_CAM)) * screen_h

                        # smoothing
                        smx = prev_x * (1 - SMOOTH_ALPHA) + x3 * SMOOTH_ALPHA
                        smy = prev_y * (1 - SMOOTH_ALPHA) + y3 * SMOOTH_ALPHA

                        vel_x = INERTIA * (smx - prev_x)
                        vel_y = INERTIA * (smy - prev_y)
                        target_x = prev_x + vel_x
                        target_y = prev_y + vel_y

                        prev_x, prev_y = target_x, target_y
                        overlay_x, overlay_y = int(target_x), int(target_y)
                        move_mouse(target_x, target_y)

                        if args.debug:
                            cv2.circle(img, (x_idx, y_idx), 8, (0, 200, 255), cv2.FILLED)

                    # left click: index+middle up
                    if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1:
                        left_click()
                        if args.debug:
                            cv2.circle(img, (x_mid, y_mid), 7, (0, 255, 0), cv2.FILLED)

                    # right click: index + thumb + middle
                    if len(fingers) >= 3 and fingers[1] == 1 and fingers[0] == 1 and fingers[2] == 1:
                        right_click()
                        if args.debug:
                            try:
                                cv2.circle(img, (lmList[4][1], lmList[4][2]), 7, (255, 0, 0), cv2.FILLED)
                            except Exception:
                                pass

                    # drag detection via findDistance (safe)
                    try:
                        distance_pair = det.findDistance(8, 4)
                        if isinstance(distance_pair, tuple):
                            length = distance_pair[0]
                        else:
                            length = None
                    except Exception:
                        length = None

                    if length is not None:
                        try:
                            if length < adaptive_drag:
                                if not dragging:
                                    dragging = True
                                    mouse_down()
                            else:
                                if dragging:
                                    dragging = False
                                    mouse_up()
                        except Exception:
                            pass

                    # scroll: four fingers up (index,middle,ring,pinky)
                    if len(fingers) >= 5 and fingers[:5] == [0, 1, 1, 1, 1]:
                        try:
                            scroll_amt = int((y_idx - H_CAM / 2) / 50)
                        except Exception:
                            scroll_amt = 0
                        try:
                            if not no_mouse and MOUSE_BACKEND == "pyautogui":
                                pyautogui.scroll(-scroll_amt)
                        except Exception:
                            pass

                # draw overlay cursor mapped into display image coords
                if overlay and overlay_x is not None and overlay_y is not None:
                    try:
                        dx = int(overlay_x * (img.shape[1] / float(screen_w)))
                        dy = int(overlay_y * (img.shape[0] / float(screen_h)))
                        cv2.circle(img, (dx, dy), 12, (0, 255, 0), 2)
                        if args.debug:
                            cv2.putText(img, "CURSOR", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except Exception:
                        pass

                # fps display
                cTime_local = time.perf_counter()
                fps = int(1.0 / (cTime_local - pTime)) if pTime else 0
                pTime = cTime_local
                cv2.putText(img, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

                if not args.debug:
                    cv2.rectangle(img, (0, img.shape[0] - 28), (380, img.shape[0]), (0, 0, 0), -1)
                    mode = "DEMO" if no_mouse else "LIVE"
                    cv2.putText(img, f"{mode} q:quit m:mirror c:cal o:overlay", (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                cv2.imshow("VirtualMouse", img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("m"):
                    mirror = not mirror
                if k == ord("c"):
                    frame_margin = simple_calibrate(det, vg, previous_margin=frame_margin)
                if k == ord("o"):
                    overlay = not overlay
            except Exception as e:
                print("Frame processing error (continuing):", str(e))
                traceback.print_exc()
                time.sleep(0.02)
                continue
    except KeyboardInterrupt:
        pass
    finally:
        vg.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
