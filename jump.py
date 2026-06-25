"""
Controls
--------
  q / Esc  quit
  c        recalibrate the standing baseline to where you are now
  [ / ]    lower / raise sensitivity (smaller threshold = more sensitive)

Requirements
------------
  pip install opencv-python mediapipe pynput
  plus 'pose_landmarker_lite.task' next to this script (see download URL below).

macOS note: sending keystrokes needs Accessibility permission (System Settings
-> Privacy & Security -> Accessibility) and Camera permission. Run Geometry Dash
in windowed/borderless mode so injected keystrokes reach it.
"""


import os
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from pynput.keyboard import Controller, Key

JUMP_SPEED = 0.45
JUMP_RISE = 0.001
REARM_RISE = 0.004
BASELINE_SMOOTHING = 0.04
COOLDOWN_S = 0.20
POS_EMA = 0.85
MIN_VISIBILITY = 0.5

CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 60

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_lite.task")

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


class CameraStream:

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit("Could not open the webcam (index 0).")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._running = True
        self._last_served = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.time()

    def read_fresh(self, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._stamp != self._last_served:
                    self._last_served = self._stamp
                    return self._frame.copy(), self._stamp
            time.sleep(0.001)
        return None, 0.0

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def make_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Model not found: {MODEL_PATH}\n"
            "Download pose_landmarker_lite.task:\n"
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def torso_center_y(landmarks):
    pts = [landmarks[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    visible = [p for p in pts if getattr(p, "visibility", 1.0) >= MIN_VISIBILITY]
    if len(visible) < 2:
        return None
    return sum(p.y for p in visible) / len(visible)


def main():
    keyboard = Controller()
    landmarker = make_landmarker()
    cam = CameraStream(0)

    smooth_y = None
    baseline_y = None
    history = deque(maxlen=2)
    armed = True
    last_jump_t = 0.0
    flash_until = 0.0
    jump_speed = JUMP_SPEED
    start = time.time()

    print("Pose jump detector running. Jump to press SPACE.")
    print("Keys: q quit | c recalibrate | [ more sensitive | ] less sensitive")

    while True:
        frame, stamp = cam.read_fresh()
        if frame is None:
            print("Camera stalled; exiting.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = stamp
        ts_ms = int((now - start) * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        status = "no person"
        speed_up = 0.0

        if result.pose_landmarks:
            cy = torso_center_y(result.pose_landmarks[0])
            if cy is not None:
                if smooth_y is None:
                    smooth_y = cy
                    baseline_y = cy
                else:
                    smooth_y = POS_EMA * cy + (1 - POS_EMA) * smooth_y
                history.append((now, smooth_y))

                if len(history) >= 2:
                    (t0, y0), (t1, y1) = history[0], history[-1]
                    dt = t1 - t0
                    speed_up = (y0 - y1) / dt if dt > 1e-3 else 0.0

                rise = baseline_y - smooth_y

                if not armed and rise <= REARM_RISE:
                    armed = True

                is_jump = (
                    armed
                    and speed_up >= jump_speed
                    and rise >= JUMP_RISE
                    and (now - last_jump_t) >= COOLDOWN_S
                )

                if is_jump:
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)
                    armed = False
                    last_jump_t = now
                    flash_until = time.time() + 0.25
                    print(f"JUMP -> SPACE  (up={speed_up:.2f} fh/s, rise={rise:.3f})")

                if armed and abs(rise) < JUMP_RISE and speed_up < jump_speed * 0.5:
                    baseline_y = (
                        BASELINE_SMOOTHING * smooth_y
                        + (1 - BASELINE_SMOOTHING) * baseline_y
                    )

                state = "ARMED" if armed else "AIR"
                status = f"{state} up={speed_up:+.2f} rise={rise:+.3f} thr={jump_speed:.2f}"

                py = int(smooth_y * h)
                by = int(baseline_y * h)
                colour = (0, 255, 0) if armed else (0, 165, 255)
                cv2.circle(frame, (w // 2, py), 8, colour, -1)
                cv2.line(frame, (0, by), (w, by), (255, 180, 0), 1)
        else:
            history.clear()

        if time.time() < flash_until:
            cv2.putText(frame, "JUMP!", (w // 2 - 90, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(frame, "q quit | c recalibrate | [ ] sensitivity",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Jump -> Space", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c") and smooth_y is not None:
            baseline_y = smooth_y
            armed = True
            print("Baseline recalibrated.")
        elif key == ord("["):
            jump_speed = max(0.15, jump_speed - 0.05)
            print(f"Sensitivity up -> threshold {jump_speed:.2f}")
        elif key == ord("]"):
            jump_speed = min(2.0, jump_speed + 0.05)
            print(f"Sensitivity down -> threshold {jump_speed:.2f}")

    cam.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()

# python3 jump.py to run
