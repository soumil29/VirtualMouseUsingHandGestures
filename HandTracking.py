#!/usr/bin/env python3
"""
HandTracking.py
MediaPipe Hands wrapper (Tasks API compatible, optimized)
"""

import math
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(
        self,
        mode=False,
        maxHands=1,
        detectionCon=0.6,
        trackCon=0.5,
        model_complexity=0,
        smooth_alpha=0.65,
        process_every=1,
        debug=False,
        motion_buf=6,
    ):
        self.maxHands = maxHands
        self.alpha = float(smooth_alpha)
        self.debug = debug

        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=maxHands,
        )

        self.hands = HandLandmarker.create_from_options(options)

        self.lm = None
        self.norm_lm = None
        self.hand_box = None
        self.prev_norm = None
        self.last_results = None

        self.idx_buf = deque(maxlen=motion_buf)
        self.time_buf = deque(maxlen=motion_buf)

    def findHands(self, img, draw=False):
        try:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

            self.last_results = self.hands.detect_for_video(
                mp_image, int(time.time() * 1000)
            )
        except Exception as e:
            if self.debug:
                print("findHands error:", e)
        return img

    def findPosition(self, img, handNo=0, draw=False):
        try:
            h, w, _ = img.shape
            self.lm = None
            self.norm_lm = None
            self.hand_box = None

            if not self.last_results or not self.last_results.hand_landmarks:
                return None, None

            if handNo >= len(self.last_results.hand_landmarks):
                return None, None

            hand = self.last_results.hand_landmarks[handNo]

            pts = []
            norm_pts = []

            for id, lm in enumerate(hand):
                nx, ny = float(lm.x), float(lm.y)
                cx, cy = int(nx * w), int(ny * h)
                pts.append((id, cx, cy))
                norm_pts.append((nx, ny))

            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            self.hand_box = (xmin, ymin, xmax, ymax)

            # Exponential smoothing
            if self.prev_norm and len(self.prev_norm) == len(norm_pts):
                sm = []
                for (px, py), (cx_, cy_) in zip(self.prev_norm, norm_pts):
                    sx = px * self.alpha + cx_ * (1 - self.alpha)
                    sy = py * self.alpha + cy_ * (1 - self.alpha)
                    sm.append((sx, sy))
                self.norm_lm = sm
            else:
                self.norm_lm = norm_pts

            self.prev_norm = list(self.norm_lm)

            self.lm = [
                (i, int(self.norm_lm[i][0] * w), int(self.norm_lm[i][1] * h))
                for i in range(len(self.norm_lm))
            ]

            return self.lm, self.hand_box

        except Exception as e:
            if self.debug:
                print("findPosition error:", e)
            return None, None

    def fingersUp(self):
        try:
            if not self.lm:
                return []

            fingers = []

            # Thumb (adaptive tolerance)
            fingers.append(1 if abs(self.lm[4][1] - self.lm[3][1]) > 15 else 0)

            # Other fingers (vertical check with tolerance)
            for tip in [8, 12, 16, 20]:
                fingers.append(
                    1 if self.lm[tip][2] < self.lm[tip - 2][2] - 12 else 0
                )

            return fingers
        except:
            return []

    def findDistance(self, p1, p2):
        try:
            if not self.lm:
                return None, None

            x1, y1 = self.lm[p1][1], self.lm[p1][2]
            x2, y2 = self.lm[p2][1], self.lm[p2][2]

            dist = math.hypot(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            return dist, (cx, cy)
        except:
            return None, None
