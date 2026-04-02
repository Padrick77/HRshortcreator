"""
Havoc Rundown Shorts Creator
============================
Converts a landscape video to a 1080×1920 portrait video by dynamically
tracking a user-selected object and smoothly panning a 9:16 crop window
to keep it centered.

Supports two tracking modes:
  - YOLO mode:   click a YOLO-detected object (works for people, cars, etc.)
  - Manual mode:  draw a box around ANY object (uses OpenCV CSRT tracker)

Dependencies:
    pip install opencv-python ultralytics opencv-contrib-python

Usage:
    python smart_cropper.py --input video.mp4 --output portrait.mp4
"""

import argparse
import collections
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Smoothing Filter
# ---------------------------------------------------------------------------

class SmoothingFilter:
    """Simple Moving Average filter for stabilizing the crop X-coordinate."""

    def __init__(self, window_size: int = 15):
        self.window_size = max(1, window_size)
        self._buffer: collections.deque = collections.deque(maxlen=self.window_size)
        self._last_value: float | None = None

    def update(self, x: float) -> float:
        """Push a new X value and return the smoothed result."""
        self._buffer.append(x)
        self._last_value = float(np.mean(self._buffer))
        return self._last_value

    def hold(self) -> float | None:
        """Return the last smoothed value without updating (object lost)."""
        return self._last_value


# ---------------------------------------------------------------------------
# Interactive Object Selector
# ---------------------------------------------------------------------------

class _ObjectSelector:
    """OpenCV mouse-callback helper for interactive target selection."""

    def __init__(self):
        self.selected_index: int | None = None
        self._boxes = []

    def set_boxes(self, boxes):
        self._boxes = boxes

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Find which box the click falls inside (prefer smallest area)
        best_idx = None
        best_area = float("inf")
        for i, (x1, y1, x2, y2) in enumerate(self._boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best_idx = i
        if best_idx is not None:
            self.selected_index = best_idx


# ---------------------------------------------------------------------------
# Smart Cropper
# ---------------------------------------------------------------------------

class SmartCropper:
    """Main engine: tracks a selected object and produces a portrait crop."""

    # Tracking mode constants
    MODE_YOLO = "yolo"
    MODE_MANUAL = "manual"

    def __init__(
        self,
        input_path: str,
        output_path: str,
        model_name: str = "yolo26n.pt",
        smoothing_window: int = 15,
        confidence: float = 0.3,
        auto_select: bool = False,
        start_time: float = 0.0,
        duration: float = 0.0,
    ):
        # Paths
        self.input_path = input_path
        self.output_path = output_path
        self.auto_select = auto_select

        # Model
        print(f"[INFO] Loading model '{model_name}' ...")
        self.model = YOLO(model_name)
        self.confidence = confidence

        # Video capture
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        self.src_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Source: {self.src_width}x{self.src_height} @ {self.fps:.1f} FPS, "
              f"{self.total_frames} frames")

        # Time range
        self.start_time = start_time
        self.duration = duration
        self._start_frame = int(self.start_time * self.fps) if self.start_time > 0 else 0
        if self.duration > 0:
            self._max_frames = int(self.duration * self.fps)
        else:
            self._max_frames = self.total_frames - self._start_frame

        if self._start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
            print(f"[INFO] Starting at {self.start_time:.1f}s (frame {self._start_frame})")
        if self.duration > 0:
            print(f"[INFO] Processing {self.duration:.1f}s ({self._max_frames} frames)")

        # Crop geometry (9:16)
        self.crop_w, self.out_w, self.out_h = self._compute_crop_geometry()
        print(f"[INFO] Crop window: {self.crop_w}x{self.src_height} → "
              f"output {self.out_w}x{self.out_h}")

        # Smoothing
        self.smoother = SmoothingFilter(smoothing_window)

        # State
        self._target_track_id: int | None = None
        self._tracking_mode: str = self.MODE_YOLO
        self._opencv_tracker = None  # Used in manual mode

        # Template re-acquisition state
        self._templates: collections.deque = collections.deque(maxlen=10)
        self._template_bbox: tuple | None = None  # (x, y, w, h) of last good track
        self._template_save_interval = 10  # save a template every N good frames
        self._frames_tracked_ok = 0
        self._reacquire_threshold = 0.35  # template match confidence

        # Drift detection state
        self._ref_histogram = None    # HSV histogram of the original target
        self._drift_threshold = 0.30  # min histogram correlation (lenient)
        self._drift_tmpl_threshold = 0.20  # min template match (lenient)
        self._prev_center = None  # (cx, cy) of previous frame for jump detection
        self._max_jump_ratio = 0.6  # max center movement as fraction of box size
        self._drift_cooldown = 0  # skip drift checks for N frames after re-acquire

    # ----- geometry --------------------------------------------------------

    def _compute_crop_geometry(self):
        """Compute the 9:16 crop width from source height."""
        crop_w = int(self.src_height * 9 / 16)  # 607 for 1080p, 1215 for 4K
        out_w = 1080
        out_h = 1920
        return crop_w, out_w, out_h

    # ----- interactive selection -------------------------------------------

    def _select_target_interactive(self, frame, results) -> tuple[str, int | None]:
        """Display detections and let the user click one, or press 'm' for manual.

        Returns (mode, track_id_or_None).
        """
        has_detections = (
            results[0].boxes.id is not None
            and len(results[0].boxes.id) > 0
        )

        if has_detections:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            names = results[0].names
        else:
            boxes_xyxy = np.array([])
            track_ids = np.array([])
            class_ids = np.array([])
            names = {}

        # Auto-select mode (for testing): pick largest box
        if self.auto_select:
            if not has_detections:
                print("[AUTO] No detections — using center crop")
                return self.MODE_MANUAL, None
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes_xyxy]
            best = int(np.argmax(areas))
            tid = int(track_ids[best])
            label = names[int(class_ids[best])]
            print(f"[AUTO] Selected object #{best}: '{label}' (track ID {tid})")
            return self.MODE_YOLO, tid

        # Scale display to fit screen (max 1280x720)
        max_disp_w, max_disp_h = 1280, 720
        disp_scale = min(max_disp_w / frame.shape[1], max_disp_h / frame.shape[0], 1.0)
        display = cv2.resize(frame, None, fx=disp_scale, fy=disp_scale)
        selector = _ObjectSelector()

        if has_detections:
            # Scale box coords to match display size
            scaled_boxes = []
            for (x1, y1, x2, y2) in boxes_xyxy:
                sx1 = int(x1 * disp_scale)
                sy1 = int(y1 * disp_scale)
                sx2 = int(x2 * disp_scale)
                sy2 = int(y2 * disp_scale)
                scaled_boxes.append((sx1, sy1, sx2, sy2))
            selector.set_boxes(scaled_boxes)

            for i, (sx1, sy1, sx2, sy2) in enumerate(scaled_boxes):
                label = f"#{i} {names[int(class_ids[i])]} [ID:{track_ids[i]}]"
                cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                font_scale = 0.5 if disp_scale < 0.5 else 0.6
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(display, (sx1, sy1 - th - 8), (sx1 + tw + 4, sy1),
                              (0, 255, 0), -1)
                cv2.putText(display, label, (sx1 + 2, sy1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        # Instructions
        line1 = "CLICK a green box to track it"
        line2 = "Press 'M' to manually draw a box around any object"
        line3 = "Press 'Q' to quit"
        cv2.putText(display, line1, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(display, line2, (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(display, line3, (20, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2, cv2.LINE_AA)

        if not has_detections:
            cv2.putText(display, "No objects auto-detected — press 'M' to select manually",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        win_name = "Havoc Rundown - Select Target"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_name, selector.mouse_callback)
        cv2.imshow(win_name, display)

        print("[INFO] Click on a detected object, or press 'M' to draw a manual box.")
        print("       Press 'Q' to cancel.")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q") or key == ord("Q"):
                cv2.destroyAllWindows()
                raise SystemExit("User cancelled selection.")
            if key == ord("m") or key == ord("M"):
                # Switch to manual ROI selection
                cv2.destroyAllWindows()
                return self._manual_roi_select(frame)
            if selector.selected_index is not None:
                break

        cv2.destroyAllWindows()

        idx = selector.selected_index
        tid = int(track_ids[idx])
        label = names[int(class_ids[idx])]
        print(f"[INFO] Selected object #{idx}: '{label}' (track ID {tid})")
        return self.MODE_YOLO, tid

    def _manual_roi_select(self, frame) -> tuple[str, None]:
        """Let the user draw a rectangle around any object to track."""
        print("[INFO] Draw a rectangle around the object you want to track.")
        print("       Press ENTER/SPACE to confirm, or 'C' to cancel.")

        # Scale down for selection if frame is very large
        max_display = 1280
        scale = 1.0
        display_frame = frame
        if frame.shape[1] > max_display:
            scale = max_display / frame.shape[1]
            display_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        roi = cv2.selectROI("Draw box around target — ENTER to confirm",
                            display_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            raise SystemExit("User cancelled manual selection.")

        # Scale ROI back to original frame size
        x, y, w, h = roi
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        print(f"[INFO] Manual ROI selected: x={x}, y={y}, w={w}, h={h}")

        # Initialize OpenCV CSRT tracker
        self._init_csrt_tracker(frame, (x, y, w, h))

        # Save the initial template for re-acquisition
        self._save_template(frame, (x, y, w, h))

        return self.MODE_MANUAL, None

    def _init_csrt_tracker(self, frame, bbox):
        """Create and initialize a CSRT (or KCF fallback) tracker."""
        tracker = None
        # Try CSRT first (best quality), then KCF fallback.
        # Newer OpenCV versions moved contrib trackers into cv2.legacy.
        for factory_name in [
            "cv2.TrackerCSRT.create",
            "cv2.legacy.TrackerCSRT_create",
            "cv2.TrackerKCF.create",
            "cv2.legacy.TrackerKCF_create",
        ]:
            parts = factory_name.split(".")
            obj = __import__(parts[0])
            try:
                for attr in parts[1:]:
                    obj = getattr(obj, attr)
                tracker = obj()
                print(f"[INFO] Using tracker: {factory_name}")
                break
            except AttributeError:
                continue

        if tracker is None:
            raise RuntimeError(
                "No compatible OpenCV tracker found. "
                "Install opencv-contrib-python: pip install opencv-contrib-python"
            )
        self._opencv_tracker = tracker
        self._opencv_tracker.init(frame, bbox)
        self._template_bbox = bbox
        self._manual_last_cx = float(bbox[0] + bbox[2] / 2)

        # Save reference histogram for drift detection
        if self._ref_histogram is None:
            self._ref_histogram = self._compute_histogram(frame, bbox)

    def _save_template(self, frame, bbox):
        """Save a cropped template of the tracked object for re-acquisition."""
        x, y, w, h = [int(v) for v in bbox]
        # Pad the template slightly for better matching
        pad = int(max(w, h) * 0.15)
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)
        template = frame[y1:y2, x1:x2].copy()
        if template.size > 0:
            self._templates.append(template)

        # Update reference histogram with a blend so it adapts gradually
        new_hist = self._compute_histogram(frame, bbox)
        if new_hist is not None and self._ref_histogram is not None:
            # 80% old + 20% new to adapt slowly
            self._ref_histogram = 0.8 * self._ref_histogram + 0.2 * new_hist

    def _compute_histogram(self, frame, bbox):
        """Compute an HSV color histogram for the region inside bbox."""
        x, y, w, h = [int(v) for v in bbox]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return None
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _check_drift(self, frame, bbox) -> bool:
        """Return True if the tracked region has drifted away from the target.

        Requires at least 2 out of 3 checks to fail before declaring drift.
        This prevents false positives from rotation, hits, or lighting changes.
        """
        # Cooldown after re-acquisition — let tracker stabilize
        if self._drift_cooldown > 0:
            self._drift_cooldown -= 1
            return False

        x, y, w, h = [int(v) for v in bbox]
        cx, cy = x + w / 2, y + h / 2
        fail_count = 0
        fail_reasons = []

        # --- Check 1: Position jump ---
        if self._prev_center is not None:
            px, py = self._prev_center
            dx = abs(cx - px)
            dy = abs(cy - py)
            max_jump = max(w, h) * self._max_jump_ratio
            if dx > max_jump or dy > max_jump:
                fail_count += 1
                fail_reasons.append(f"jump=({dx:.0f},{dy:.0f})")
        self._prev_center = (cx, cy)

        # --- Check 2: Histogram color match ---
        hist_score = 1.0
        if self._ref_histogram is not None:
            current_hist = self._compute_histogram(frame, bbox)
            if current_hist is not None:
                hist_score = cv2.compareHist(
                    self._ref_histogram, current_hist, cv2.HISTCMP_CORREL)
                if hist_score < self._drift_threshold:
                    fail_count += 1
                    fail_reasons.append(f"color={hist_score:.2f}")

        # --- Check 3: Template structural match ---
        tmpl_score = 1.0
        if self._templates:
            region = frame[max(0, y):min(frame.shape[0], y + h),
                          max(0, x):min(frame.shape[1], x + w)]
            if region.size > 0:
                region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                best_tmpl_score = 0.0
                for tmpl in self._templates:
                    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                    tmpl_resized = cv2.resize(tmpl_gray,
                                              (region_gray.shape[1], region_gray.shape[0]))
                    result = cv2.matchTemplate(
                        region_gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
                    score = result[0][0] if result.size > 0 else 0
                    best_tmpl_score = max(best_tmpl_score, score)
                tmpl_score = best_tmpl_score
                if tmpl_score < self._drift_tmpl_threshold:
                    fail_count += 1
                    fail_reasons.append(f"struct={tmpl_score:.2f}")

        # Require at least 2 checks to fail
        if fail_count >= 2:
            print(f"[DRIFT] Detected ({fail_count}/3 failed): {', '.join(fail_reasons)}")
            self._prev_center = None
            return True

        return False

    def _try_reacquire(self, frame) -> tuple[float, float, float, float] | None:
        """Search NEAR THE LAST KNOWN POSITION for the lost object.

        Limits search to a region around where the object was last seen,
        preventing false matches on spectators, walls, etc.
        """
        if not self._templates or self._template_bbox is None:
            return None

        # Define search region: 4x the bbox size around last known position
        lx, ly, lw, lh = [int(v) for v in self._template_bbox]
        lcx, lcy = lx + lw // 2, ly + lh // 2
        search_radius_x = max(lw * 4, 400)
        search_radius_y = max(lh * 4, 400)

        sx1 = max(0, lcx - search_radius_x)
        sy1 = max(0, lcy - search_radius_y)
        sx2 = min(frame.shape[1], lcx + search_radius_x)
        sy2 = min(frame.shape[0], lcy + search_radius_y)

        search_region = frame[sy1:sy2, sx1:sx2]
        if search_region.size == 0:
            return None

        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

        best_val = 0.0
        best_loc = None
        best_scale = 1.0
        best_template = None

        scales = [0.7, 0.85, 1.0, 1.15, 1.3]

        for template in self._templates:
            tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            th, tw = tmpl_gray.shape[:2]

            for scale in scales:
                new_w = int(tw * scale)
                new_h = int(th * scale)
                if new_w < 10 or new_h < 10:
                    continue
                if new_w >= search_gray.shape[1] or new_h >= search_gray.shape[0]:
                    continue

                scaled_tmpl = cv2.resize(tmpl_gray, (new_w, new_h))
                result = cv2.matchTemplate(search_gray, scaled_tmpl,
                                           cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale
                    best_template = template

        if best_val >= 0.45 and best_loc is not None:
            th, tw = best_template.shape[:2]
            w = int(tw * best_scale)
            h = int(th * best_scale)
            # Map coordinates back to full frame
            x = best_loc[0] + sx1
            y = best_loc[1] + sy1
            print(f"[REACQUIRE] Object re-found! confidence={best_val:.2f} "
                  f"at ({x},{y}) scale={best_scale:.2f}")
            return (x, y, w, h)

        return None

    # ----- per-frame tracking ----------------------------------------------

    def _get_target_center_x_yolo(self, results) -> float | None:
        """Return the X-center of the YOLO-tracked object, or None if lost."""
        if results[0].boxes.id is None:
            return None

        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        mask = track_ids == self._target_track_id
        if not mask.any():
            return None

        x1, y1, x2, y2 = boxes[mask][0]
        return float((x1 + x2) / 2.0)

    def _get_target_center_x_manual(self, frame) -> float | None:
        """Update the OpenCV tracker and return center-x.

        Includes drift detection: if CSRT reports success but the tracked
        region no longer looks like the target, treat it as lost.
        """
        success, bbox = self._opencv_tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]

            # --- Drift detection: is CSRT still on the right object? ---
            self._frames_tracked_ok += 1
            if self._check_drift(frame, (x, y, w, h)):
                # CSRT drifted - treat as lost
                success = False
            else:
                cx = float(x + w / 2)
                self._manual_last_cx = cx
                self._template_bbox = (x, y, w, h)

                # Save templates while tracking is good
                if self._frames_tracked_ok % self._template_save_interval == 0:
                    self._save_template(frame, (x, y, w, h))

                return cx

        # --- Tracking lost or drifted: attempt re-acquisition ---
        self._frames_tracked_ok = 0
        reacq = self._try_reacquire(frame)

        if reacq is not None:
            self._init_csrt_tracker(frame, reacq)
            x, y, w, h = [int(v) for v in reacq]
            cx = float(x + w / 2)
            self._save_template(frame, reacq)
            self._drift_cooldown = 20  # let tracker stabilize
            return cx

        # Truly lost — hold position
        return None

    # ----- cropping --------------------------------------------------------

    def _crop_frame(self, frame: np.ndarray, center_x: float) -> np.ndarray:
        """Extract a 9:16 vertical slice centered on `center_x`, resize to output."""
        half_w = self.crop_w // 2

        cx = int(round(center_x))
        cx = max(half_w, min(self.src_width - half_w, cx))

        x1 = cx - half_w
        x2 = x1 + self.crop_w

        cropped = frame[:, x1:x2]
        resized = cv2.resize(cropped, (self.out_w, self.out_h),
                             interpolation=cv2.INTER_LINEAR)
        return resized

    # ----- audio muxing ----------------------------------------------------

    def _find_ffmpeg(self) -> str:
        """Locate the ffmpeg executable — PATH first, then imageio_ffmpeg bundle."""
        import shutil
        path = shutil.which("ffmpeg")
        if path:
            return path
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return ""

    def _mux_audio(self, original_input: str, silent_video: str, final_output: str):
        """Use ffmpeg to copy the audio from the original into the cropped video."""
        ffmpeg_path = self._find_ffmpeg()
        if not ffmpeg_path:
            print("[WARN] ffmpeg not found. Install: pip install imageio-ffmpeg")
            if os.path.exists(final_output):
                os.remove(final_output)
            os.rename(silent_video, final_output)
            return

        print(f"[INFO] Muxing audio with ffmpeg ({os.path.basename(ffmpeg_path)}) ...")

        # Build the input args for the original file (with time offset if needed)
        original_input_args = []
        if self.start_time > 0:
            original_input_args += ["-ss", str(self.start_time)]
        original_input_args += ["-i", original_input]
        if self.duration > 0:
            original_input_args += ["-t", str(self.duration)]

        cmd = [
            ffmpeg_path, "-y",
            "-i", silent_video,
            *original_input_args,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            final_output,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"[WARN] ffmpeg returned code {result.returncode}")
                print(result.stderr[-500:] if result.stderr else "")
                print("[WARN] Falling back to video-only output.")
                if os.path.exists(final_output):
                    os.remove(final_output)
                os.rename(silent_video, final_output)
            else:
                os.remove(silent_video)
                print("[INFO] Audio muxed successfully.")
        except FileNotFoundError:
            print("[WARN] ffmpeg binary not executable. Output will have no audio.")
            if os.path.exists(final_output):
                os.remove(final_output)
            os.rename(silent_video, final_output)

    # ----- main loop -------------------------------------------------------

    def process(self):
        """Run the full pipeline: detect → select → track → crop → write → mux."""
        t_start = time.time()

        base, ext = os.path.splitext(self.output_path)
        silent_path = f"{base}_silent{ext}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(silent_path, fourcc, self.fps,
                                 (self.out_w, self.out_h))

        frame_idx = 0
        lost_since = 0
        selection_done = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Stop if we've processed enough frames
            if frame_idx >= self._max_frames:
                break

            # --- First frame(s): select target ---
            if not selection_done:
                # Run YOLO to show available detections
                results = self.model.track(
                    frame, tracker="bytetrack.yaml", persist=True,
                    verbose=False, conf=self.confidence,
                )
                mode, tid = self._select_target_interactive(frame, results)
                self._tracking_mode = mode
                self._target_track_id = tid
                selection_done = True
                print(f"[INFO] Tracking mode: {self._tracking_mode.upper()}")

            # --- Per-frame tracking ---
            if self._tracking_mode == self.MODE_YOLO:
                results = self.model.track(
                    frame, tracker="bytetrack.yaml", persist=True,
                    verbose=False, conf=self.confidence,
                )
                cx = self._get_target_center_x_yolo(results)
            else:
                # Manual/CSRT tracking — no YOLO needed each frame
                cx = self._get_target_center_x_manual(frame)

            if cx is not None:
                smoothed_x = self.smoother.update(cx)
                lost_since = 0
            else:
                smoothed_x = self.smoother.hold()
                lost_since += 1
                if smoothed_x is None:
                    smoothed_x = self.src_width / 2.0

            # Crop and write
            cropped = self._crop_frame(frame, smoothed_x)
            writer.write(cropped)

            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == self._max_frames:
                elapsed = time.time() - t_start
                pct = frame_idx / max(self._max_frames, 1) * 100
                fps_proc = frame_idx / max(elapsed, 0.001)
                status = "TRACKING" if lost_since == 0 else f"LOST ({lost_since}f)"
                print(f"[PROGRESS] {frame_idx}/{self._max_frames} "
                      f"({pct:.0f}%) | {fps_proc:.1f} FPS | {status}")

        writer.release()
        self.cap.release()

        elapsed = time.time() - t_start
        print(f"\n[INFO] Video processing complete: {frame_idx} frames in {elapsed:.1f}s "
              f"({frame_idx / max(elapsed, 0.001):.1f} FPS)")

        self._mux_audio(self.input_path, silent_path, self.output_path)

        print(f"[DONE] Output saved to: {self.output_path}")
        return self.output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Havoc Rundown Shorts Creator — landscape to portrait with object tracking"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the input landscape video")
    parser.add_argument("--output", "-o", default="output_portrait.mp4",
                        help="Path for the output portrait video (default: output_portrait.mp4)")
    parser.add_argument("--model", "-m", default="yolo26n.pt",
                        help="YOLO model weights (default: yolo26n.pt)")
    parser.add_argument("--smoothing", "-s", type=int, default=15,
                        help="Smoothing window size in frames (default: 15)")
    parser.add_argument("--confidence", "-c", type=float, default=0.15,
                        help="Minimum detection confidence (default: 0.15)")
    parser.add_argument("--auto-select", action="store_true",
                        help="Auto-select the largest object (skip interactive selection)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", "-d", type=float, default=0.0,
                        help="Duration in seconds to process (default: full video)")

    args = parser.parse_args()

    cropper = SmartCropper(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        smoothing_window=args.smoothing,
        confidence=args.confidence,
        auto_select=args.auto_select,
        start_time=args.start,
        duration=args.duration,
    )
    cropper.process()


if __name__ == "__main__":
    main()
