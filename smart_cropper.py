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
import queue
import subprocess
import sys
import threading
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
# Threaded I/O Pipeline
# ---------------------------------------------------------------------------

class ThreadedFrameReader:
    """Reads video frames in a background thread into a bounded queue."""

    def __init__(self, cap: cv2.VideoCapture, max_frames: int,
                 queue_size: int = 128):
        self._cap = cap
        self._max_frames = max_frames
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _reader_loop(self):
        count = 0
        while not self._stopped and count < self._max_frames:
            ret, frame = self._cap.read()
            if not ret:
                break
            self._queue.put(frame)  # blocks if queue is full
            count += 1
        self._queue.put(None)  # sentinel

    def read(self):
        """Return the next frame, or None when finished."""
        frame = self._queue.get()
        return frame

    def stop(self):
        self._stopped = True


class ThreadedFrameWriter:
    """Writes video frames from a queue in a background thread."""

    def __init__(self, writer: cv2.VideoWriter, queue_size: int = 128):
        self._writer = writer
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _writer_loop(self):
        while True:
            frame = self._queue.get()
            if frame is None:  # sentinel
                break
            self._writer.write(frame)

    def write(self, frame):
        """Queue a frame for writing (blocks if queue is full)."""
        self._queue.put(frame)

    def stop(self):
        """Signal the writer to finish and wait for it to drain."""
        self._queue.put(None)  # sentinel
        self._thread.join()



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
        zoom: float = 1.0,
        auto_label: bool = True,
        training_only: bool = False,
    ):
        # Paths
        self.input_path = input_path
        self.output_path = output_path
        self.auto_select = auto_select

        # Model — resolve relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_name) if not os.path.isabs(model_name) else model_name
        print(f"[INFO] Loading model '{model_path}' ...")
        self.model = YOLO(model_path)
        self.confidence = confidence

        # Zoom
        self.zoom = max(1.0, float(zoom))

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

        # Crop geometry (9:16 with zoom)
        self.crop_w, self.crop_h, self.out_w, self.out_h = self._compute_crop_geometry()
        print(f"[INFO] Crop window: {self.crop_w}x{self.crop_h} "
              f"(zoom {self.zoom:.1f}x) → output {self.out_w}x{self.out_h}")

        # Smoothing (X and Y axes)
        self.smoother_x = SmoothingFilter(smoothing_window)
        self.smoother_y = SmoothingFilter(smoothing_window)

        # State
        self._target_track_id: int | None = None
        self._tracking_mode: str = self.MODE_YOLO
        self._opencv_tracker = None  # Used in manual mode

        # Template re-acquisition state
        self._templates: collections.deque = collections.deque(maxlen=20)
        self._template_bbox: tuple | None = None  # (x, y, w, h) of last good track
        self._original_bbox: tuple | None = None  # original selection for size reference
        self._template_save_interval = 5  # save a template every N good frames
        self._frames_tracked_ok = 0
        self._reacquire_threshold = 0.35  # template match confidence

        # Drift detection state
        self._ref_histogram = None    # HSV histogram of the original target
        self._drift_threshold = 0.30  # min histogram correlation (lenient)
        self._drift_tmpl_threshold = 0.20  # min template match (lenient)
        self._prev_center = None  # (cx, cy) of previous frame for jump detection
        self._max_jump_ratio = 0.6  # max center movement as fraction of box size
        self._drift_cooldown = 0  # skip drift checks for N frames after re-acquire

        # Velocity model for motion prediction
        self._velocity = (0.0, 0.0)  # (vx, vy) pixels per frame
        self._velocity_alpha = 0.3   # exponential smoothing factor

        # Optical flow point tracking (secondary tracker)
        self._flow_points = None     # tracked keypoints (Nx1x2 float32)
        self._flow_prev_gray = None  # previous frame grayscale for LK flow
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Padded tracker — CSRT works better with extra spatial context
        self._tracker_pad_ratio = 0.4  # pad each side by 40% of object size

        # Secondary tracker for the second bot (training labels only)
        self._secondary_tracker = None
        self._secondary_bbox: tuple | None = None  # (x, y, w, h)
        self._secondary_pad = None
        self._secondary_templates: collections.deque = collections.deque(maxlen=10)

        # Training-only mode: skip video output, just collect labels
        self.training_only = training_only

        # Auto-labeling: collect YOLO training data from confident tracking
        self.auto_label = auto_label
        self._label_save_interval = 15  # save every N good-tracking frames
        self._label_dataset_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "training_data")
        self._label_count = 0
        self._label_class_id = 0  # YOLO class index for "bot"

    # ----- geometry --------------------------------------------------------

    def _compute_crop_geometry(self):
        """Compute the 9:16 crop dimensions, scaled by zoom factor.

        At zoom=1.0 the crop is the full source height (original behaviour).
        At zoom=2.0 the crop is half-sized in each axis → subjects appear 2× bigger.
        """
        crop_h = int(self.src_height / self.zoom)
        crop_w = int(crop_h * 9 / 16)
        # Clamp to source dimensions
        crop_w = min(crop_w, self.src_width)
        crop_h = min(crop_h, self.src_height)
        out_w = 1080
        out_h = 1920
        return crop_w, crop_h, out_w, out_h

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
        """Let the user draw rectangles around both bots.

        First selection = primary bot (used for crop tracking).
        Second selection = secondary bot (training labels only, optional).
        """
        # Scale down for selection if frame is very large
        max_display = 1280
        scale = 1.0
        display_frame = frame
        if frame.shape[1] > max_display:
            scale = max_display / frame.shape[1]
            display_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # --- Bot 1: Primary (crop tracking) ---
        print("[INFO] Draw a rectangle around the PRIMARY bot (the one to follow).")
        print("       Press ENTER/SPACE to confirm, or 'C' to cancel.")
        roi1 = cv2.selectROI("BOT 1 (PRIMARY) — draw box, ENTER to confirm",
                             display_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi1 == (0, 0, 0, 0):
            raise SystemExit("User cancelled manual selection.")

        x, y, w, h = roi1
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)
        primary_bbox = (x, y, w, h)

        print(f"[INFO] Primary bot selected: x={x}, y={y}, w={w}, h={h}")

        # Initialize primary CSRT tracker
        self._init_csrt_tracker(frame, primary_bbox)
        self._save_template(frame, primary_bbox)

        # --- Bot 2: Secondary (training labels only) ---
        # Draw bot 1 on the display so user can see it
        disp2 = display_frame.copy()
        sx1, sy1 = int(x * scale), int(y * scale)
        sw, sh = int(w * scale), int(h * scale)
        cv2.rectangle(disp2, (sx1, sy1), (sx1 + sw, sy1 + sh), (0, 255, 0), 2)
        cv2.putText(disp2, "BOT 1 (PRIMARY)", (sx1, sy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        print("[INFO] Now draw a rectangle around the SECOND bot.")
        print("       Press ENTER/SPACE to confirm, or 'C' to skip.")
        roi2 = cv2.selectROI("BOT 2 (SECONDARY) — draw box, ENTER to confirm, C to skip",
                             disp2, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi2 != (0, 0, 0, 0):
            x2, y2, w2, h2 = roi2
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            w2 = int(w2 / scale)
            h2 = int(h2 / scale)
            secondary_bbox = (x2, y2, w2, h2)
            print(f"[INFO] Secondary bot selected: x={x2}, y={y2}, w={w2}, h={h2}")
            self._init_secondary_tracker(frame, secondary_bbox)
        else:
            print("[INFO] Secondary bot skipped.")

        return self.MODE_MANUAL, None

    def _init_secondary_tracker(self, frame, bbox):
        """Create a CSRT tracker for the second bot (training labels only)."""
        tracker = None
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
                break
            except AttributeError:
                continue

        if tracker is None:
            print("[WARN] Could not create secondary tracker.")
            return

        x, y, w, h = [int(v) for v in bbox]
        self._secondary_bbox = (x, y, w, h)

        # Pad for better tracking
        pad_x = int(w * self._tracker_pad_ratio)
        pad_y = int(h * self._tracker_pad_ratio)
        px = max(0, x - pad_x)
        py = max(0, y - pad_y)
        pw = min(frame.shape[1] - px, w + 2 * pad_x)
        ph = min(frame.shape[0] - py, h + 2 * pad_y)
        self._secondary_pad = (x - px, y - py, w, h)

        self._secondary_tracker = tracker
        self._secondary_tracker.init(frame, (px, py, pw, ph))
        print(f"[INFO] Secondary tracker initialized.")

        # Save initial template
        pad = int(max(w, h) * 0.15)
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)
        template = frame[y1:y2, x1:x2].copy()
        if template.size > 0:
            self._secondary_templates.append(template)

    def _update_secondary_tracker(self, frame) -> tuple | None:
        """Update the secondary bot tracker and return its bbox, or None."""
        if self._secondary_tracker is None:
            return None

        success, padded_bbox = self._secondary_tracker.update(frame)
        if not success:
            return self._secondary_bbox  # hold last position

        px, py, pw, ph = [int(v) for v in padded_bbox]
        if self._secondary_pad is not None:
            off_x, off_y, ow, oh = self._secondary_pad
            x = px + off_x
            y = py + off_y
            w = ow
            h = oh
        else:
            x, y, w, h = px, py, pw, ph

        self._secondary_bbox = (x, y, w, h)
        return (x, y, w, h)

    def _init_csrt_tracker(self, frame, bbox):
        """Create and initialize a CSRT (or KCF fallback) tracker.

        Uses a padded bounding box so CSRT has more spatial context,
        which significantly improves tracking of small objects.
        """
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

        # Store the true (unpadded) object bbox
        x, y, w, h = [int(v) for v in bbox]
        self._template_bbox = (x, y, w, h)
        self._manual_last_cx = float(x + w / 2)

        # Save original size reference on first init
        if self._original_bbox is None:
            self._original_bbox = (x, y, w, h)

        # Pad the tracker region for better context (helps small objects)
        pad_x = int(w * self._tracker_pad_ratio)
        pad_y = int(h * self._tracker_pad_ratio)
        px = max(0, x - pad_x)
        py = max(0, y - pad_y)
        pw = min(frame.shape[1] - px, w + 2 * pad_x)
        ph = min(frame.shape[0] - py, h + 2 * pad_y)
        self._tracker_pad = (x - px, y - py, w, h)  # offset to get true box from padded

        self._opencv_tracker = tracker
        self._opencv_tracker.init(frame, (px, py, pw, ph))

        # Initialize optical flow points inside the target region
        self._init_flow_points(frame, (x, y, w, h))

        # Save reference histogram for drift detection
        if self._ref_histogram is None:
            self._ref_histogram = self._compute_histogram(frame, bbox)

    def _init_flow_points(self, frame, bbox):
        """Detect good features to track inside the target for optical flow."""
        x, y, w, h = [int(v) for v in bbox]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._flow_prev_gray = gray

        # Create mask for the target region
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[max(0, y):min(gray.shape[0], y + h),
             max(0, x):min(gray.shape[1], x + w)] = 255

        # Detect corners/features inside the target
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=50, qualityLevel=0.05, minDistance=5, mask=mask)
        self._flow_points = points
        n = len(points) if points is not None else 0
        print(f"[FLOW] Initialized {n} feature points in target region")

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

    def _auto_label_save(self, frame, primary_bbox, secondary_bbox=None):
        """Save a frame + YOLO-format annotations for both bots.

        Only called during confident tracking (no drift detected).
        Annotations use normalised xywh format per YOLO spec.
        Each bot is a separate line in the label file.
        """
        if not self.auto_label:
            return

        img_h, img_w = frame.shape[:2]

        # Collect all valid bboxes
        bboxes = [primary_bbox]
        if secondary_bbox is not None:
            bboxes.append(secondary_bbox)

        # Ensure the dataset directory exists
        img_dir = os.path.join(self._label_dataset_dir, "images", "train")
        lbl_dir = os.path.join(self._label_dataset_dir, "labels", "train")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Generate a unique filename using video name + frame count
        video_stem = os.path.splitext(os.path.basename(self.input_path))[0]
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_stem)
        fname = f"{safe_stem}_{self._label_count:05d}"

        # Save image
        img_path = os.path.join(img_dir, f"{fname}.jpg")
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Save label (one line per bot)
        lbl_path = os.path.join(lbl_dir, f"{fname}.txt")
        with open(lbl_path, "w") as f:
            for bbox in bboxes:
                x, y, w, h = [int(v) for v in bbox]
                cx_norm = (x + w / 2) / img_w
                cy_norm = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                f.write(f"{self._label_class_id} {cx_norm:.6f} {cy_norm:.6f} "
                        f"{w_norm:.6f} {h_norm:.6f}\n")

        self._label_count += 1
        if self._label_count == 1:
            print(f"[AUTOLABEL] Saving training data to: {self._label_dataset_dir}")
            print(f"[AUTOLABEL] Labeling {len(bboxes)} bot(s) per frame")
        if self._label_count % 50 == 0:
            print(f"[AUTOLABEL] {self._label_count} labeled frames saved")

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

        Uses adaptive thresholds based on object size — small objects get
        more lenient thresholds since their patches are noisier.
        Requires at least 2 out of 3 checks to fail before declaring drift.
        """
        # Cooldown after re-acquisition — let tracker stabilize
        if self._drift_cooldown > 0:
            self._drift_cooldown -= 1
            return False

        x, y, w, h = [int(v) for v in bbox]
        cx, cy = x + w / 2, y + h / 2
        fail_count = 0
        fail_reasons = []

        # Adaptive thresholds: relax for small objects
        obj_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        area_ratio = obj_area / max(frame_area, 1)
        # Objects < 1% of frame area get substantially relaxed thresholds
        if area_ratio < 0.01:
            drift_thresh = self._drift_threshold * 0.5  # much more lenient
            tmpl_thresh = self._drift_tmpl_threshold * 0.3
            jump_ratio = self._max_jump_ratio * 1.5
        elif area_ratio < 0.03:
            drift_thresh = self._drift_threshold * 0.7
            tmpl_thresh = self._drift_tmpl_threshold * 0.6
            jump_ratio = self._max_jump_ratio * 1.2
        else:
            drift_thresh = self._drift_threshold
            tmpl_thresh = self._drift_tmpl_threshold
            jump_ratio = self._max_jump_ratio

        # --- Check 1: Position jump ---
        if self._prev_center is not None:
            px, py = self._prev_center
            dx = abs(cx - px)
            dy = abs(cy - py)
            max_jump = max(w, h) * jump_ratio
            if dx > max_jump or dy > max_jump:
                fail_count += 1
                fail_reasons.append(f"jump=({dx:.0f},{dy:.0f})")
        self._prev_center = (cx, cy)

        # Update velocity model
        if self._prev_center is not None and len(fail_reasons) == 0:
            old_vx, old_vy = self._velocity
            if hasattr(self, '_prev_center_for_vel') and self._prev_center_for_vel is not None:
                pvx, pvy = self._prev_center_for_vel
                new_vx = cx - pvx
                new_vy = cy - pvy
                a = self._velocity_alpha
                self._velocity = (a * new_vx + (1 - a) * old_vx,
                                  a * new_vy + (1 - a) * old_vy)
        self._prev_center_for_vel = (cx, cy)

        # --- Check 2: Histogram color match ---
        hist_score = 1.0
        if self._ref_histogram is not None:
            current_hist = self._compute_histogram(frame, bbox)
            if current_hist is not None:
                hist_score = cv2.compareHist(
                    self._ref_histogram, current_hist, cv2.HISTCMP_CORREL)
                if hist_score < drift_thresh:
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
                if tmpl_score < tmpl_thresh:
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

        Uses velocity prediction to centre the search on where the object
        is expected to be, then falls back to template matching + optical flow.
        """
        if not self._templates or self._template_bbox is None:
            return None

        # Predict position using velocity model
        lx, ly, lw, lh = [int(v) for v in self._template_bbox]
        lcx, lcy = lx + lw // 2, ly + lh // 2
        vx, vy = self._velocity
        # Project ahead a few frames (object may have moved since we lost it)
        pred_cx = int(lcx + vx * 3)
        pred_cy = int(lcy + vy * 3)
        pred_cx = max(0, min(frame.shape[1], pred_cx))
        pred_cy = max(0, min(frame.shape[0], pred_cy))

        # Define search region: wider for small objects
        obj_area = lw * lh
        frame_area = frame.shape[0] * frame.shape[1]
        if obj_area / max(frame_area, 1) < 0.01:
            # Small object — search much wider
            search_radius_x = max(lw * 8, 600)
            search_radius_y = max(lh * 8, 600)
        else:
            search_radius_x = max(lw * 5, 400)
            search_radius_y = max(lh * 5, 400)

        sx1 = max(0, pred_cx - search_radius_x)
        sy1 = max(0, pred_cy - search_radius_y)
        sx2 = min(frame.shape[1], pred_cx + search_radius_x)
        sy2 = min(frame.shape[0], pred_cy + search_radius_y)

        search_region = frame[sy1:sy2, sx1:sx2]
        if search_region.size == 0:
            return None

        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

        best_val = 0.0
        best_loc = None
        best_scale = 1.0
        best_template = None

        # Wider scale range for small objects that may change apparent size
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

        for template in self._templates:
            tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            th, tw = tmpl_gray.shape[:2]

            for scale in scales:
                new_w = int(tw * scale)
                new_h = int(th * scale)
                if new_w < 8 or new_h < 8:
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

        # Lower threshold for small objects (0.35 vs 0.45)
        reacq_threshold = 0.35 if obj_area / max(frame_area, 1) < 0.02 else 0.45

        if best_val >= reacq_threshold and best_loc is not None:
            th, tw = best_template.shape[:2]
            w = int(tw * best_scale)
            h = int(th * best_scale)
            # Map coordinates back to full frame
            x = best_loc[0] + sx1
            y = best_loc[1] + sy1
            print(f"[REACQUIRE] Template match! confidence={best_val:.2f} "
                  f"at ({x},{y}) scale={best_scale:.2f}")
            return (x, y, w, h)

        # --- Fallback: optical flow ---
        flow_result = self._try_flow_reacquire(frame)
        if flow_result is not None:
            return flow_result

        return None

    def _try_flow_reacquire(self, frame) -> tuple[float, float, float, float] | None:
        """Use optical flow tracked points to estimate where the target moved."""
        if (self._flow_points is None or self._flow_prev_gray is None
                or len(self._flow_points) < 3):
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._flow_prev_gray, gray, self._flow_points, None,
            **self._lk_params)

        if new_pts is None or status is None:
            return None

        # Keep only successfully tracked points
        good_mask = status.flatten() == 1
        if good_mask.sum() < 3:
            return None

        old_good = self._flow_points[good_mask]
        new_good = new_pts[good_mask]

        # Filter outliers using median displacement
        displacements = new_good.reshape(-1, 2) - old_good.reshape(-1, 2)
        med_dx = np.median(displacements[:, 0])
        med_dy = np.median(displacements[:, 1])
        distances = np.sqrt((displacements[:, 0] - med_dx) ** 2 +
                            (displacements[:, 1] - med_dy) ** 2)
        inlier_mask = distances < np.median(distances) * 3 + 1
        if inlier_mask.sum() < 3:
            return None

        inlier_pts = new_good[inlier_mask].reshape(-1, 2)

        # Compute bounding box of tracked points
        min_x, min_y = inlier_pts.min(axis=0)
        max_x, max_y = inlier_pts.max(axis=0)

        # Use original object size as reference (flow points may be sparse)
        if self._original_bbox is not None:
            _, _, ow, oh = self._original_bbox
        elif self._template_bbox is not None:
            _, _, ow, oh = self._template_bbox
        else:
            return None

        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        x = int(cx - ow / 2)
        y = int(cy - oh / 2)

        # Validate: the flow region should be roughly where we expect
        if self._template_bbox is not None:
            lx, ly, lw, lh = self._template_bbox
            dist = np.sqrt((cx - (lx + lw / 2)) ** 2 + (cy - (ly + lh / 2)) ** 2)
            max_dist = max(lw, lh) * 8
            if dist > max_dist:
                return None

        print(f"[REACQUIRE] Optical flow! {inlier_mask.sum()} inlier points, "
              f"center=({cx:.0f},{cy:.0f})")

        # Update flow state
        self._flow_points = inlier_pts.reshape(-1, 1, 2).astype(np.float32)
        self._flow_prev_gray = gray

        return (x, y, ow, oh)

    # ----- per-frame tracking ----------------------------------------------

    def _get_target_center_yolo(self, results) -> tuple[float, float] | None:
        """Return the (cx, cy) center of the YOLO-tracked object, or None if lost."""
        if results[0].boxes.id is None:
            return None

        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        mask = track_ids == self._target_track_id
        if not mask.any():
            return None

        x1, y1, x2, y2 = boxes[mask][0]
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    def _get_target_center_manual(self, frame) -> tuple[float, float] | None:
        """Update the OpenCV tracker and return (cx, cy).

        Uses padded tracking with drift detection. Falls back to optical
        flow and template re-acquisition when CSRT loses the target.
        """
        success, padded_bbox = self._opencv_tracker.update(frame)

        if success:
            # Unpad: extract the true object bbox from the padded tracker result
            px, py, pw, ph = [int(v) for v in padded_bbox]
            if hasattr(self, '_tracker_pad'):
                off_x, off_y, ow, oh = self._tracker_pad
                x = px + off_x
                y = py + off_y
                w = ow
                h = oh
            else:
                x, y, w, h = px, py, pw, ph

            # --- Drift detection: is CSRT still on the right object? ---
            self._frames_tracked_ok += 1
            if self._check_drift(frame, (x, y, w, h)):
                # CSRT drifted - treat as lost
                success = False
            else:
                cx = float(x + w / 2)
                cy = float(y + h / 2)
                self._manual_last_cx = cx
                self._template_bbox = (x, y, w, h)

                # Update optical flow tracking
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self._flow_points is not None and self._flow_prev_gray is not None:
                    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self._flow_prev_gray, gray, self._flow_points, None,
                        **self._lk_params)
                    if new_pts is not None and status is not None:
                        good = status.flatten() == 1
                        if good.sum() > 0:
                            self._flow_points = new_pts[good].reshape(-1, 1, 2)

                    # Refresh flow points every 30 frames to avoid depletion
                    if (self._frames_tracked_ok % 30 == 0
                            or (self._flow_points is not None
                                and len(self._flow_points) < 5)):
                        self._init_flow_points(frame, (x, y, w, h))
                    else:
                        self._flow_prev_gray = gray
                else:
                    self._init_flow_points(frame, (x, y, w, h))

                # Save templates while tracking is good (more frequently)
                if self._frames_tracked_ok % self._template_save_interval == 0:
                    self._save_template(frame, (x, y, w, h))

                # Update secondary tracker and auto-label both bots
                sec_bbox = self._update_secondary_tracker(frame)
                if self._frames_tracked_ok % self._label_save_interval == 0:
                    self._auto_label_save(frame, (x, y, w, h), sec_bbox)

                return cx, cy

        # --- Tracking lost or drifted: attempt re-acquisition ---
        self._frames_tracked_ok = 0
        reacq = self._try_reacquire(frame)

        if reacq is not None:
            self._init_csrt_tracker(frame, reacq)
            x, y, w, h = [int(v) for v in reacq]
            cx = float(x + w / 2)
            cy = float(y + h / 2)
            self._save_template(frame, reacq)
            self._drift_cooldown = 30  # longer cooldown after re-acquire
            return cx, cy

        # Truly lost — hold position
        return None

    # ----- cropping --------------------------------------------------------

    def _crop_frame(self, frame: np.ndarray, center_x: float,
                    center_y: float | None = None) -> np.ndarray:
        """Extract a 9:16 region centered on (center_x, center_y), resize to output.

        When zoom > 1.0 both width and height of the crop are reduced,
        making the tracked subject appear larger in the output.
        """
        half_w = self.crop_w // 2
        half_h = self.crop_h // 2

        # X axis — clamp to frame bounds
        cx = int(round(center_x))
        cx = max(half_w, min(self.src_width - half_w, cx))
        x1 = cx - half_w
        x2 = x1 + self.crop_w

        # Y axis — default to vertical center if not provided
        if center_y is None:
            center_y = self.src_height / 2.0
        cy = int(round(center_y))
        cy = max(half_h, min(self.src_height - half_h, cy))
        y1 = cy - half_h
        y2 = y1 + self.crop_h

        cropped = frame[y1:y2, x1:x2]
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
        """Run the full pipeline: detect → select → track → crop → write → mux.

        Uses a 3-thread pipeline to overlap I/O with processing:
          Thread 1 (reader):  decodes frames from disk into a queue
          Thread 2 (main):    tracking + cropping
          Thread 3 (writer):  encodes cropped frames to disk from a queue
        """
        t_start = time.time()

        base, ext = os.path.splitext(self.output_path)
        silent_path = f"{base}_silent{ext}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv_writer = cv2.VideoWriter(silent_path, fourcc, self.fps,
                                    (self.out_w, self.out_h))

        # --- First frame: read synchronously for target selection ---
        ret, first_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video.")

        # Run YOLO on first frame and let user select target
        results = self.model.track(
            first_frame, tracker="bytetrack.yaml", persist=True,
            verbose=False, conf=self.confidence,
        )
        mode, tid = self._select_target_interactive(first_frame, results)
        self._tracking_mode = mode
        self._target_track_id = tid
        print(f"[INFO] Tracking mode: {self._tracking_mode.upper()}")

        # Process the first frame
        if self._tracking_mode == self.MODE_YOLO:
            center = self._get_target_center_yolo(results)
        else:
            center = self._get_target_center_manual(first_frame)

        if center is not None:
            smoothed_x = self.smoother_x.update(center[0])
            smoothed_y = self.smoother_y.update(center[1])
        else:
            smoothed_x = self.src_width / 2.0
            smoothed_y = self.src_height / 2.0

        cropped = self._crop_frame(first_frame, smoothed_x, smoothed_y)
        cv_writer.write(cropped)
        frame_idx = 1
        lost_since = 0

        # --- Start threaded pipeline for remaining frames ---
        remaining = self._max_frames - 1
        reader = ThreadedFrameReader(self.cap, remaining).start()
        writer = ThreadedFrameWriter(cv_writer).start()

        print(f"[INFO] Threaded pipeline started (reader → processor → writer)")

        while True:
            frame = reader.read()
            if frame is None:
                break

            # --- Per-frame tracking ---
            if self._tracking_mode == self.MODE_YOLO:
                results = self.model.track(
                    frame, tracker="bytetrack.yaml", persist=True,
                    verbose=False, conf=self.confidence,
                )
                center = self._get_target_center_yolo(results)
            else:
                center = self._get_target_center_manual(frame)

            if center is not None:
                smoothed_x = self.smoother_x.update(center[0])
                smoothed_y = self.smoother_y.update(center[1])
                lost_since = 0
            else:
                smoothed_x = self.smoother_x.hold()
                smoothed_y = self.smoother_y.hold()
                lost_since += 1
                if smoothed_x is None:
                    smoothed_x = self.src_width / 2.0
                if smoothed_y is None:
                    smoothed_y = self.src_height / 2.0

            # Crop and queue for writing
            cropped = self._crop_frame(frame, smoothed_x, smoothed_y)
            writer.write(cropped)

            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == self._max_frames:
                elapsed = time.time() - t_start
                pct = frame_idx / max(self._max_frames, 1) * 100
                fps_proc = frame_idx / max(elapsed, 0.001)
                status = "TRACKING" if lost_since == 0 else f"LOST ({lost_since}f)"
                print(f"[PROGRESS] {frame_idx}/{self._max_frames} "
                      f"({pct:.0f}%) | {fps_proc:.1f} FPS | {status}")

        # Drain the writer queue and release resources
        reader.stop()
        writer.stop()
        cv_writer.release()
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
    parser.add_argument("--zoom", "-z", type=float, default=1.0,
                        help="Zoom level (1.0 = full frame, 2.0 = 2x zoom, etc.)")

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
        zoom=args.zoom,
    )
    cropper.process()


if __name__ == "__main__":
    main()
