#!/usr/bin/env python3

import argparse
import sys
import time
from threading import Thread
import cv2
import numpy as np
from HandTracking import HandDetector

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

CAM_INDEX = 0
W_CAM, H_CAM = 1280, 720
FRAME_MARGIN_DEFAULT = 100
SMOOTH_ALPHA = 0.88
INERTIA = 0.72
DRAG_THRESHOLD = 30
CLICK_COOLDOWN = 0.15

mirror = True
overlay = True
no_mouse = False

class VideoGet:
    def __init__(self, src=0, width=W_CAM, height=H_CAM):
        self.cap = cv2.VideoCapture(
            src, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stopped = False
        self.grabbed = False
        self.frame = None
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            self.grabbed = grabbed
            if grabbed:
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

_last_click = 0.0

def _now():
    return time.perf_counter()

def move_mouse(x, y):
    if no_mouse:
        return
    try:
        if MOUSE_BACKEND == "autopy":
            autopy.mouse.move(int(x), int(y))
        elif MOUSE_BACKEND == "pyautogui":
            pyautogui.moveTo(int(x), int(y))
    except Exception:
        pass

def _click(kind="left"):
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

def get_screen_size():
    try:
        import screeninfo
        m = screeninfo.get_monitors()[0]
        return m.width, m.height
    except Exception:
        import pyautogui
        return pyautogui.size()

def main():
    global mirror, overlay, no_mouse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mouse", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cam", type=int, default=CAM_INDEX)
    args = parser.parse_args()

    no_mouse = args.no_mouse

    det = HandDetector(
        maxHands=1,
        smooth_alpha=0.65,
        process_every=1,
        debug=args.debug,
    )

    if MOUSE_BACKEND is None and not no_mouse:
        no_mouse = True

    vg = VideoGet(src=args.cam)
    screen_w, screen_h = get_screen_size()

    prev_x = prev_y = 0.0
    dragging = False
    pTime = 0.0

    try:
        while True:
            grabbed, frame = vg.read()
            if not grabbed:
                continue

            img = frame.copy()
            if mirror:
                img = cv2.flip(img, 1)

            img = det.findHands(img)
            lmList, bbox = det.findPosition(img)

            if lmList and len(lmList) > 12:
                x_idx, y_idx = lmList[8][1], lmList[8][2]
                fingers = det.fingersUp()

                if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 0:
                    x3 = np.interp(x_idx, (FRAME_MARGIN_DEFAULT, W_CAM - FRAME_MARGIN_DEFAULT), (0, screen_w))
                    y3 = np.interp(y_idx, (FRAME_MARGIN_DEFAULT, H_CAM - FRAME_MARGIN_DEFAULT), (0, screen_h))

                    smx = prev_x + (x3 - prev_x) * SMOOTH_ALPHA
                    smy = prev_y + (y3 - prev_y) * SMOOTH_ALPHA

                    vel_x = (smx - prev_x) * INERTIA
                    vel_y = (smy - prev_y) * INERTIA

                    target_x = prev_x + vel_x
                    target_y = prev_y + vel_y

                    prev_x, prev_y = target_x, target_y
                    move_mouse(target_x, target_y)

                if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1:
                    left_click()

                if len(fingers) >= 3 and fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1:
                    right_click()

                dist, _ = det.findDistance(8, 4)
                if dist is not None:
                    if dist < DRAG_THRESHOLD:
                        if not dragging:
                            dragging = True
                            mouse_down()
                    else:
                        if dragging:
                            dragging = False
                            mouse_up()

            cTime = time.perf_counter()
            fps = int(1 / (cTime - pTime)) if pTime else 0
            pTime = cTime
            cv2.putText(img, f"FPS: {fps}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.imshow("VirtualMouse", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        vg.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
