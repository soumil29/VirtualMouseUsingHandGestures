#!/usr/bin/env python3
"""
HandTracking.py
MediaPipe Hands wrapper with smoothing, gesture helpers, and defensive coding.
"""

import math
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 1,
        detectionCon: float = 0.6,
        trackCon: float = 0.5,
        model_complexity: int = 0,
        smooth_alpha: float = 0.6,
        process_every: int = 1,
        debug: bool = False,
        motion_buf: int = 8,
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lm = None  # integer pixel landmarks: list of tuples (id,x,y)
        self.norm_lm = None  # normalized landmarks in [0,1]
        self.hand_box = None
        self.handedness = None
        self.prev_norm = None
        self.alpha = float(smooth_alpha)
        self.frame_idx = 0
        self.process_every = max(1, int(process_every))
        self.last_results = None
        self.debug = debug
        self.motion_buf = int(motion_buf)
        self.idx_buf = deque(maxlen=self.motion_buf)
        self.time_buf = deque(maxlen=self.motion_buf)

    def findHands(self, img, draw: bool = False):
        """Run MediaPipe on the image (may skip frames depending on process_every)."""
        try:
            self.frame_idx += 1
            do_proc = (self.frame_idx % self.process_every) == 0
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if do_proc:
                self.last_results = self.hands.process(imgRGB)
            if self.last_results and draw and self.debug and getattr(self.last_results, "multi_hand_landmarks", None):
                for h in self.last_results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, h, self.mpHands.HAND_CONNECTIONS)
        except Exception as e:
            if self.debug:
                print("findHands error:", e)
        return img

    def findPosition(self, img, handNo: int = 0, draw: bool = False) -> Tuple[Optional[List[Tuple[int, int, int]]], Optional[Tuple[int, int, int, int]]]:
        """Return (lm, hand_box) where lm is list of (id,x,y) integer pixels or (None,None)."""
        try:
            h, w, _ = img.shape
            self.lm = None
            self.norm_lm = None
            self.hand_box = None
            self.handedness = None
            res = self.last_results
            if not res or not getattr(res, "multi_hand_landmarks", None):
                return None, None
            if handNo >= len(res.multi_hand_landmarks):
                return None, None
            hand = res.multi_hand_landmarks[handNo]
            pts = []
            norm_pts = []
            for id, lm in enumerate(hand.landmark):
                nx, ny = float(lm.x), float(lm.y)
                cx, cy = int(nx * w), int(ny * h)
                pts.append((id, cx, cy))
                norm_pts.append((nx, ny))
            if not pts:
                return None, None
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            self.hand_box = (xmin, ymin, xmax, ymax)
            try:
                self.handedness = res.multi_handedness[handNo].classification[0].label
            except Exception:
                self.handedness = None

            # Smooth normalized landmarks
            cur = [(nx, ny) for nx, ny in norm_pts]
            if self.prev_norm and len(self.prev_norm) == len(cur):
                sm = []
                for (px, py), (cx_, cy_) in zip(self.prev_norm, cur):
                    sx = px * self.alpha + cx_ * (1 - self.alpha)
                    sy = py * self.alpha + cy_ * (1 - self.alpha)
                    sm.append((sx, sy))
                self.norm_lm = sm
            else:
                self.norm_lm = cur
            self.prev_norm = list(self.norm_lm)

            # update motion buffer (index tip) in normalized coords
            if len(self.norm_lm) > 8:
                ix, iy = self.norm_lm[8]
                self.idx_buf.append((ix, iy))
                self.time_buf.append(time.time())

            if draw and self.debug:
                cv2.rectangle(img, (xmin - 12, ymin - 12), (xmax + 12, ymax + 12), (0, 255, 0), 1)
                for id, cx, cy in pts:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), -1)

            self.lm = [(i, int(self.norm_lm[i][0] * w), int(self.norm_lm[i][1] * h)) for i in range(len(self.norm_lm))]
            return self.lm, self.hand_box
        except Exception as e:
            if self.debug:
                print("findPosition error:", e)
            return None, None

    def fingersUp(self) -> list:
        """Return a list of 5 ints (thumb, index, middle, ring, pinky) or [] if not available."""
        try:
            if not self.lm or not self.norm_lm:
                return []
            fingers = []
            # thumb: compare tip x vs ip x, fallback safely
            try:
                tip_x = self.lm[4][1]
                ip_x = self.lm[3][1]
                if self.handedness == "Right":
                    fingers.append(1 if tip_x < ip_x else 0)
                else:
                    fingers.append(1 if tip_x > ip_x else 0)
            except Exception:
                fingers.append(0)
            # other fingers: angle test with fallback
            for i in [8, 12, 16, 20]:
                try:
                    tip = self.norm_lm[i]
                    pip = self.norm_lm[i - 2]
                    mcp = self.norm_lm[i - 3] if (i - 3) >= 0 else pip
                    bax = mcp[0] - pip[0]
                    bay = mcp[1] - pip[1]
                    bcx = tip[0] - pip[0]
                    bcy = tip[1] - pip[1]
                    denom = (bax * bax + bay * bay) * (bcx * bcx + bcy * bcy)
                    angle = 180.0
                    if denom > 1e-8:
                        cosv = (bax * bcx + bay * bcy) / math.sqrt(denom)
                        cosv = max(-1.0, min(1.0, cosv))
                        angle = math.degrees(math.acos(cosv))
                    fingers.append(1 if angle > 150 else 0)
                except Exception:
                    try:
                        fingers.append(1 if self.lm[i][2] < self.lm[i - 2][2] else 0)
                    except Exception:
                        fingers.append(0)
            return fingers
        except Exception as e:
            if self.debug:
                print("fingersUp error:", e)
            return []

    def findDistance(self, p1: int, p2: int, img=None, draw: bool = False):
        """Return (px_distance, (cx,cy)) or (None,None).
        Accepts optional img/draw for backward compatibility.
        """
        try:
            if not self.lm:
                return None, None
            if p1 >= len(self.lm) or p2 >= len(self.lm):
                return None, None
            x1, y1 = self.lm[p1][1], self.lm[p1][2]
            x2, y2 = self.lm[p2][1], self.lm[p2][2]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if draw and img is not None and self.debug:
                cv2.circle(img, (x1, y1), 6, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 6, (0, 255, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return dist, (cx, cy)
        except Exception as e:
            if self.debug:
                print("findDistance error:", e)
            return None, None

    def detect_motion(self) -> Optional[str]:
        """Lightweight motion detector returning 'swipe_left/right' or 'wave' or None."""
        try:
            if len(self.idx_buf) < 4:
                return None
            dx = self.idx_buf[-1][0] - self.idx_buf[0][0]
            dy = self.idx_buf[-1][1] - self.idx_buf[0][1]
            dt = max(1e-3, self.time_buf[-1] - self.time_buf[0])
            vx = dx / dt
            vy = dy / dt
            if abs(vx) > 1.0 and abs(vx) > abs(vy):
                return "swipe_right" if vx > 0 else "swipe_left"
            xs = [p[0] for p in self.idx_buf]
            changes = sum(1 for i in range(2, len(xs)) if (xs[i] - xs[i - 1]) * (xs[i - 1] - xs[i - 2]) < 0)
            if changes >= 2:
                return "wave"
            return None
        except Exception as e:
            if self.debug:
                print("detect_motion error:", e)
            return None

    def detect_thumb_vertical(self) -> Optional[str]:
        """Return 'thumbs_up'/'thumbs_down' or None based on normalized y positions."""
        try:
            if not self.norm_lm or len(self.norm_lm) <= 8:
                return None
            thumb_y = self.norm_lm[4][1]
            wrist_y = self.norm_lm[0][1]
            index_mcp_y = self.norm_lm[5][1]
            if thumb_y < wrist_y - 0.06 and thumb_y < index_mcp_y - 0.04:
                return "thumbs_up"
            if thumb_y > wrist_y + 0.06 and thumb_y > index_mcp_y + 0.04:
                return "thumbs_down"
            return None
        except Exception as e:
            if self.debug:
                print("detect_thumb_vertical error:", e)
            return None

    def detect_ok(self) -> bool:
        try:
            if not self.norm_lm or len(self.norm_lm) <= 8:
                return False
            tx, ty = self.norm_lm[4]
            ix, iy = self.norm_lm[8]
            d = math.hypot(tx - ix, ty - iy)
            fingers = self.fingersUp()
            return d < 0.045 and sum(fingers) >= 3
        except Exception as e:
            if self.debug:
                print("detect_ok error:", e)
            return False

    def detect_gesture(self) -> Optional[str]:
        """Return stable gesture label or None. Fully defensively coded."""
        try:
            if not self.lm:
                return None
            fingers = self.fingersUp()
            if sum(fingers) == 0:
                return "fist"
            if fingers == [1, 1, 1, 1, 1]:
                return "open_palm"
            if self.detect_ok():
                return "ok_sign"
            if fingers == [0, 1, 0, 0, 0]:
                return "point"
            if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                return "index_middle"
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and sum(fingers) == 3:
                return "three_fingers"
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                return "four_fingers"
            dist_pair = self.findDistance(8, 4)
            if isinstance(dist_pair, tuple):
                dist = dist_pair[0]
            else:
                dist = None
            if dist is not None and dist < 40 and fingers[1] == 1 and fingers[0] == 1:
                return "thumb_index_pinch"
            t = self.detect_thumb_vertical()
            if t:
                return t
            m = self.detect_motion()
            if m:
                return m
            return None
        except Exception as e:
            if self.debug:
                print("detect_gesture error:", e)
            return None


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    det = HandDetector(debug=True)
    p = time.perf_counter()
    while True:
        s, img = cap.read()
        if not s:
            time.sleep(0.01)
            continue
        img = det.findHands(img, draw=True)
        lm, b = det.findPosition(img, draw=True)
        if lm:
            try:
                print("gesture:", det.detect_gesture())
            except Exception:
                pass
        now = time.perf_counter()
        fps = int(1 / (now - p)) if p else 0
        p = now
        cv2.putText(img, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("HandTracking", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
