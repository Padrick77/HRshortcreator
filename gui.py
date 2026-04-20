"""
Havoc Rundown Shorts Creator — GUI Launcher
============================================
Modern dark-themed desktop interface with side-by-side video previews.

Usage:
    python gui.py
"""

import os
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

C = {
    "bg":         "#0d0d0d",
    "card":       "#161625",
    "input":      "#1c1c35",
    "border":     "#2a2a42",
    "btn":        "#1e3a5f",
    "btn_hover":  "#27507a",
    "accent":     "#e94560",
    "accent2":    "#ff6b81",
    "text":       "#f0f0f0",
    "text2":      "#8a8a9a",
    "text3":      "#5a5a6a",
    "log_bg":     "#0a0a15",
    "green":      "#2ecc71",
    "yellow":     "#f39c12",
    "red":        "#e74c3c",
    "blue":       "#3498db",
    "preview_bg": "#0a0a12",
}

FONT = "Segoe UI"
F = {
    "title":   (FONT, 16, "bold"),
    "sub":     (FONT, 10),
    "label":   (FONT, 10),
    "lbl_b":   (FONT, 10, "bold"),
    "input":   (FONT, 10),
    "btn":     (FONT, 10, "bold"),
    "log":     ("Consolas", 9),
    "stat":    (FONT, 11, "bold"),
    "small":   (FONT, 9),
    "preview": (FONT, 9, "bold"),
}


# ---------------------------------------------------------------------------
# Video Player Widget
# ---------------------------------------------------------------------------

class VideoPreview:
    """Renders video frames onto a tk.Canvas with playback controls."""

    def __init__(self, parent, label_text="", width=480, height=270):
        self.frame = tk.Frame(parent, bg=C["card"], highlightbackground=C["border"],
                              highlightthickness=1)

        # Header
        tk.Label(self.frame, text=label_text, font=F["preview"], bg=C["card"],
                 fg=C["accent"]).pack(anchor="w", padx=10, pady=(8, 4))

        # Canvas
        self.canvas_width = width
        self.canvas_height = height
        self.canvas = tk.Canvas(self.frame, width=width, height=height,
                                bg=C["preview_bg"], highlightthickness=0)
        self.canvas.pack(padx=10, pady=(0, 8))

        # State
        self._cap = None
        self._photo = None  # prevent GC
        self._total_frames = 0
        self._fps = 30.0

        # Render scale tracking for coordinate mapping
        self._render_scale = 1.0
        self._render_offset_x = 0
        self._render_offset_y = 0
        self._source_w = 0
        self._source_h = 0

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def load_video(self, path: str):
        """Open a video file for preview."""
        self.release()
        if not path or not os.path.exists(path):
            self._show_placeholder("No video loaded")
            return False
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._show_placeholder("Cannot open video")
            return False
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.show_frame(0)
        return True

    def show_frame(self, frame_idx: int):
        """Display a specific frame by index."""
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            return
        self._render_frame(frame)

    def show_cv_frame(self, frame: np.ndarray):
        """Display a raw OpenCV frame (BGR numpy array)."""
        self._render_frame(frame)

    def _render_frame(self, frame: np.ndarray):
        """Convert and display an OpenCV frame on the canvas."""
        h, w = frame.shape[:2]
        self._source_w = w
        self._source_h = h

        # Fit to canvas while preserving aspect ratio
        scale = min(self.canvas_width / w, self.canvas_height / h)
        self._render_scale = scale
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        x_off = (self.canvas_width - new_w) // 2
        y_off = (self.canvas_height - new_h) // 2
        self._render_offset_x = x_off
        self._render_offset_y = y_off
        self.canvas.create_image(x_off, y_off, anchor="nw", image=self._photo)

    def canvas_to_source(self, cx, cy):
        """Map canvas pixel coords to source frame coords."""
        sx = (cx - self._render_offset_x) / self._render_scale
        sy = (cy - self._render_offset_y) / self._render_scale
        sx = max(0, min(sx, self._source_w))
        sy = max(0, min(sy, self._source_h))
        return int(sx), int(sy)

    def _show_placeholder(self, text: str):
        """Show a placeholder message on the canvas."""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas_width // 2, self.canvas_height // 2,
            text=text, fill=C["text3"], font=F["sub"],
        )

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def fps(self):
        return self._fps


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class SmartCropperGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Havoc Rundown Shorts Creator")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1100x900")
        self.root.minsize(950, 800)

        # State
        self._processing = False
        self._paused = False
        self._seek_to_frame = None   # set by user during pause to jump
        self._playing = False
        self._play_after_id = None
        self._current_play_frame = 0
        self._reselect_requested = False
        self._current_frame = None  # latest frame for reselection
        self._pending_roi = None    # (x, y, w, h) in source coords from drag
        self._drag_start = None     # canvas coords of drag start
        self._drag_rect_id = None   # canvas rectangle id

        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.youtube_url = tk.StringVar()
        self.smoothing = tk.IntVar(value=15)
        self.confidence = tk.DoubleVar(value=0.15)
        self.start_time = tk.DoubleVar(value=0.0)
        self.duration = tk.DoubleVar(value=0.0)
        self.zoom = tk.DoubleVar(value=1.0)
        self.training_mode = tk.BooleanVar(value=False)
        self.progress_pct = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="Ready")
        self._downloading = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # =====================================================================
    # UI CONSTRUCTION
    # =====================================================================

    def _build_ui(self):
        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=10)

        self._build_title(main)
        self._build_file_section(main)
        self._build_previews(main)
        self._build_playback_bar(main)
        self._build_settings_and_actions(main)
        self._build_progress_bar(main)
        self._build_log(main)

    def _build_title(self, parent):
        f = tk.Frame(parent, bg=C["bg"])
        f.pack(fill="x", pady=(0, 10))
        tk.Label(f, text="🤖  Havoc Rundown Shorts Creator", font=F["title"],
                 bg=C["bg"], fg=C["text"]).pack(side="left")
        tk.Label(f, text="Landscape → Portrait with AI tracking",
                 font=F["sub"], bg=C["bg"], fg=C["text2"]).pack(side="left", padx=(12, 0))

    def _build_file_section(self, parent):
        card = tk.Frame(parent, bg=C["card"], highlightbackground=C["border"],
                        highlightthickness=1)
        card.pack(fill="x", pady=(0, 8))
        inner = tk.Frame(card, bg=C["card"])
        inner.pack(fill="x", padx=12, pady=10)

        # YouTube URL row
        yt_row = tk.Frame(inner, bg=C["card"])
        yt_row.pack(fill="x", pady=(0, 6))
        tk.Label(yt_row, text="▶  YouTube", font=F["lbl_b"], bg=C["card"],
                 fg=C["accent"], anchor="w").pack(side="left")
        yt_entry = tk.Entry(yt_row, textvariable=self.youtube_url, bg=C["input"],
                            fg=C["text"], insertbackground=C["accent"], relief="flat",
                            font=F["input"], highlightthickness=1,
                            highlightbackground=C["border"], highlightcolor=C["accent"])
        yt_entry.pack(side="left", fill="x", expand=True, padx=(8, 6))
        yt_entry.insert(0, "")
        yt_entry.bind("<FocusIn>", lambda e: None)
        self.yt_download_btn = tk.Button(
            yt_row, text="⬇  Download", command=self._download_youtube,
            bg="#c0392b", fg=C["text"], activebackground="#e74c3c",
            relief="flat", font=F["small"], cursor="hand2",
            padx=12, pady=3,
        )
        self.yt_download_btn.pack(side="right")

        # Divider
        div = tk.Frame(inner, bg=C["border"], height=1)
        div.pack(fill="x", pady=(0, 6))

        # Input row
        row1 = tk.Frame(inner, bg=C["card"])
        row1.pack(fill="x", pady=(0, 4))
        tk.Label(row1, text="Input", font=F["lbl_b"], bg=C["card"],
                 fg=C["text2"], width=7, anchor="w").pack(side="left")
        e1 = tk.Entry(row1, textvariable=self.input_path, bg=C["input"],
                      fg=C["text"], insertbackground=C["accent"], relief="flat",
                      font=F["input"], highlightthickness=1,
                      highlightbackground=C["border"], highlightcolor=C["accent"])
        e1.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(row1, text="Browse", command=self._browse_input,
                  bg=C["btn"], fg=C["text"], activebackground=C["btn_hover"],
                  relief="flat", font=F["small"], cursor="hand2",
                  padx=12, pady=3).pack(side="right")

        # Output row
        row2 = tk.Frame(inner, bg=C["card"])
        row2.pack(fill="x")
        tk.Label(row2, text="Output", font=F["lbl_b"], bg=C["card"],
                 fg=C["text2"], width=7, anchor="w").pack(side="left")
        e2 = tk.Entry(row2, textvariable=self.output_path, bg=C["input"],
                      fg=C["text"], insertbackground=C["accent"], relief="flat",
                      font=F["input"], highlightthickness=1,
                      highlightbackground=C["border"], highlightcolor=C["accent"])
        e2.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(row2, text="Browse", command=self._browse_output,
                  bg=C["btn"], fg=C["text"], activebackground=C["btn_hover"],
                  relief="flat", font=F["small"], cursor="hand2",
                  padx=12, pady=3).pack(side="right")

    def _build_previews(self, parent):
        """Side-by-side video previews: input (landscape) + output (portrait)."""
        preview_frame = tk.Frame(parent, bg=C["bg"])
        preview_frame.pack(fill="x", pady=(0, 4))
        preview_frame.columnconfigure(0, weight=3)
        preview_frame.columnconfigure(1, weight=2)

        self.input_preview = VideoPreview(preview_frame, "📹  INPUT (Landscape)",
                                          width=540, height=304)
        self.input_preview.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # Bind mouse events for inline tracker repositioning
        self.input_preview.canvas.bind("<ButtonPress-1>", self._on_preview_mousedown)
        self.input_preview.canvas.bind("<B1-Motion>", self._on_preview_mousemove)
        self.input_preview.canvas.bind("<ButtonRelease-1>", self._on_preview_mouseup)

        self.output_preview = VideoPreview(preview_frame, "📱  OUTPUT (Portrait)",
                                           width=171, height=304)
        self.output_preview.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # Initial placeholders
        self.input_preview._show_placeholder("Select an input video →  Browse")
        self.output_preview._show_placeholder("Output preview\nafter processing")

    def _build_playback_bar(self, parent):
        """Seek slider + play/pause for comparing input vs output."""
        bar = tk.Frame(parent, bg=C["bg"])
        bar.pack(fill="x", pady=(0, 8))

        self.play_btn = tk.Button(
            bar, text="▶", command=self._toggle_playback,
            bg=C["btn"], fg=C["text"], activebackground=C["btn_hover"],
            relief="flat", font=(FONT, 12), cursor="hand2", width=3, pady=1,
        )
        self.play_btn.pack(side="left", padx=(0, 8))

        self.seek_var = tk.IntVar(value=0)
        self.seek_slider = tk.Scale(
            bar, from_=0, to=100, orient="horizontal", variable=self.seek_var,
            command=self._on_seek, showvalue=False,
            bg=C["bg"], fg=C["text"], troughcolor=C["input"],
            activebackground=C["accent"], highlightthickness=0,
            sliderrelief="flat", length=200,
        )
        self.seek_slider.pack(side="left", fill="x", expand=True)

        self.time_label = tk.Label(bar, text="0:00 / 0:00", font=F["small"],
                                   bg=C["bg"], fg=C["text3"])
        self.time_label.pack(side="right", padx=(8, 0))

    def _build_settings_and_actions(self, parent):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", pady=(0, 8))

        # Settings
        settings = tk.Frame(row, bg=C["card"], highlightbackground=C["border"],
                            highlightthickness=1)
        settings.pack(side="left", fill="x", expand=True, padx=(0, 8))
        si = tk.Frame(settings, bg=C["card"])
        si.pack(fill="x", padx=10, pady=8)

        pairs = [
            ("Smooth", self.smoothing, 4), ("Conf", self.confidence, 5),
            ("Start(s)", self.start_time, 6), ("Dur(s)", self.duration, 6),
            ("Zoom", self.zoom, 4),
        ]
        for i, (lbl, var, w) in enumerate(pairs):
            tk.Label(si, text=lbl, font=F["small"], bg=C["card"],
                     fg=C["text2"]).grid(row=0, column=i * 2, padx=(8 if i else 0, 2))
            tk.Entry(si, textvariable=var, width=w, bg=C["input"], fg=C["text"],
                     insertbackground=C["accent"], relief="flat", font=F["small"],
                     highlightthickness=1, highlightbackground=C["border"],
                     highlightcolor=C["accent"]).grid(row=0, column=i * 2 + 1, padx=(0, 6))

        tk.Label(si, text="(dur 0 = full, zoom 1 = no zoom)", font=(FONT, 8), bg=C["card"],
                 fg=C["text3"]).grid(row=0, column=len(pairs) * 2, padx=(0, 4))

        # Training mode toggle
        train_row = tk.Frame(settings, bg=C["card"])
        train_row.pack(fill="x", padx=10, pady=(0, 8))
        self.training_checkbox = tk.Checkbutton(
            train_row, text="🏋  Training Mode Only (collect labels, skip video output)",
            variable=self.training_mode, font=F["small"],
            bg=C["card"], fg=C["yellow"], activebackground=C["card"],
            activeforeground=C["yellow"], selectcolor=C["input"],
            highlightthickness=0, cursor="hand2",
        )
        self.training_checkbox.pack(side="left")

        # Action buttons
        btns = tk.Frame(row, bg=C["bg"])
        btns.pack(side="right")

        self.start_btn = tk.Button(
            btns, text="▶  Process", command=self._start_processing,
            bg=C["accent"], fg=C["text"], activebackground=C["accent2"],
            relief="flat", font=F["btn"], cursor="hand2", padx=16, pady=6,
        )
        self.start_btn.pack(side="left", padx=(0, 6))

        self.pause_btn = tk.Button(
            btns, text="⏸  Pause", command=self._toggle_pause,
            bg=C["yellow"], fg="#1a1a1a", activebackground="#e67e22",
            relief="flat", font=F["btn"], cursor="hand2", padx=12, pady=6,
            state="disabled",
        )
        self.pause_btn.pack(side="left", padx=(0, 6))

        self.stop_btn = tk.Button(
            btns, text="■  Save & Stop", command=self._stop_processing,
            bg=C["red"], fg=C["text"], activebackground="#c0392b",
            relief="flat", font=F["btn"], cursor="hand2", padx=12, pady=6,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=(0, 6))

        self.reselect_btn = tk.Button(
            btns, text="⟳  Reselect", command=self._request_reselect,
            bg=C["blue"], fg=C["text"], activebackground="#2980b9",
            relief="flat", font=F["btn"], cursor="hand2", padx=12, pady=6,
            state="disabled",
        )
        self.reselect_btn.pack(side="left")

    def _build_progress_bar(self, parent):
        pf = tk.Frame(parent, bg=C["bg"])
        pf.pack(fill="x", pady=(0, 6))

        # Status
        sf = tk.Frame(pf, bg=C["bg"])
        sf.pack(fill="x")
        self.status_label = tk.Label(sf, textvariable=self.status_text, font=F["stat"],
                                     bg=C["bg"], fg=C["text"])
        self.status_label.pack(side="left")
        self.fps_label = tk.Label(sf, text="", font=F["small"], bg=C["bg"], fg=C["text2"])
        self.fps_label.pack(side="right")

        # Bar
        self.prog_canvas = tk.Canvas(pf, height=6, bg=C["input"], highlightthickness=0, bd=0)
        self.prog_canvas.pack(fill="x", pady=(4, 0))
        self.prog_canvas.bind("<Configure>", self._draw_progress)

    def _build_log(self, parent):
        lf = tk.Frame(parent, bg=C["bg"])
        lf.pack(fill="both", expand=True)

        tk.Label(lf, text="📋  LOG", font=F["lbl_b"], bg=C["bg"],
                 fg=C["accent"]).pack(anchor="w", pady=(0, 3))

        container = tk.Frame(lf, bg=C["border"])
        container.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            container, bg=C["log_bg"], fg=C["text2"], font=F["log"],
            relief="flat", wrap="word", padx=8, pady=6,
            insertbackground=C["accent"], selectbackground=C["accent"],
            state="disabled", height=6,
        )
        sb = ttk.Scrollbar(container, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        for tag, color in [("info", C["text"]), ("progress", C["accent"]),
                           ("success", C["green"]), ("warning", C["yellow"]),
                           ("error", C["red"]), ("reacquire", C["blue"])]:
            self.log_text.tag_configure(tag, foreground=color)

    # =====================================================================
    # PROGRESS BAR
    # =====================================================================

    def _draw_progress(self, _=None):
        self.prog_canvas.delete("all")
        w = self.prog_canvas.winfo_width()
        h = self.prog_canvas.winfo_height()
        fill_w = int(w * self.progress_pct.get() / 100)
        if fill_w > 0:
            self.prog_canvas.create_rectangle(0, 0, fill_w, h,
                                              fill=C["accent"], outline="")

    # =====================================================================
    # FILE BROWSING
    # =====================================================================

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                       ("All files", "*.*")],
        )
        if path:
            self._load_input_video(path)

    def _load_input_video(self, path: str):
        """Shared helper — load a video file into the input preview and set paths."""
        self.input_path.set(path)
        p = Path(path)
        self.output_path.set(str(p.parent / f"{p.stem}_portrait{p.suffix}"))

        self.input_preview.load_video(path)
        self._update_seek_range(self.input_preview.total_frames,
                                self.input_preview.fps)
        self._log(f"📂 Loaded: {p.name}", "info")

        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = frames / fps if fps > 0 else 0
        cap.release()
        self._log(f"   {w}×{h} @ {fps:.1f}fps — {dur:.1f}s ({frames} frames)", "info")

    # =====================================================================
    # YOUTUBE DOWNLOAD
    # =====================================================================

    def _is_youtube_url(self, url: str) -> bool:
        """Check if a string looks like a YouTube URL."""
        patterns = [
            r'(https?://)?(www\.)?youtube\.com/watch\?v=',
            r'(https?://)?(www\.)?youtube\.com/shorts/',
            r'(https?://)?youtu\.be/',
            r'(https?://)?(www\.)?youtube\.com/live/',
        ]
        return any(re.search(p, url) for p in patterns)

    def _download_youtube(self):
        """Download a YouTube video via yt-dlp."""
        url = self.youtube_url.get().strip()
        if not url:
            self._log("❌ Paste a YouTube URL first.", "error")
            return
        if not self._is_youtube_url(url):
            self._log("❌ That doesn't look like a YouTube URL.", "error")
            return
        if self._downloading:
            self._log("⚠️  Download already in progress.", "warning")
            return

        self._downloading = True
        self.yt_download_btn.config(state="disabled", text="⬇  Downloading...")
        self._log(f"⬇  Downloading: {url}", "info")
        self._update_progress(0, status="Downloading from YouTube...")

        threading.Thread(target=self._yt_download_thread, args=(url,), daemon=True).start()

    def _yt_download_thread(self, url: str):
        """Background thread that runs yt-dlp to download the video."""
        try:
            import yt_dlp

            # Create downloads folder next to the script
            dl_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "downloads"
            dl_dir.mkdir(exist_ok=True)

            # Progress hook
            def progress_hook(d):
                if d['status'] == 'downloading':
                    total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        pct = downloaded / total * 100
                        speed = d.get('speed', 0) or 0
                        speed_mb = speed / 1024 / 1024
                        self._update_progress(pct, status=f"Downloading... {pct:.0f}% ({speed_mb:.1f} MB/s)")
                elif d['status'] == 'finished':
                    self._update_progress(100, status="Download complete — merging...")
                    self._log("✅ Download complete, merging streams...", "success")

            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(dl_dir / '%(title)s.%(ext)s'),
                'progress_hooks': [progress_hook],
                'merge_output_format': 'mp4',
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                # yt-dlp may change extension after merge
                if not os.path.exists(filename):
                    base, _ = os.path.splitext(filename)
                    filename = base + '.mp4'

            if os.path.exists(filename):
                self._log(f"🎉 Saved: {os.path.basename(filename)}", "success")
                self._update_progress(100, status="Ready to process")
                # Auto-load the downloaded video as input
                self.root.after(0, lambda f=filename: self._load_input_video(f))
            else:
                self._log("❌ Download finished but file not found.", "error")
                self._update_progress(0, status="Download failed")

        except ImportError:
            self._log("❌ yt-dlp not installed. Run: pip install yt-dlp", "error")
            self._update_progress(0, status="Missing yt-dlp")
        except Exception as e:
            self._log(f"❌ Download failed: {e}", "error")
            self._update_progress(0, status="Download failed")
        finally:
            self._downloading = False
            self.root.after(0, lambda: self.yt_download_btn.config(
                state="normal", text="⬇  Download"))

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save Output As", defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All", "*.*")],
        )
        if path:
            self.output_path.set(path)

    # =====================================================================
    # PLAYBACK CONTROLS
    # =====================================================================

    def _update_seek_range(self, total_frames: int, fps: float):
        self.seek_slider.config(to=max(total_frames - 1, 1))
        self._playback_fps = fps
        self._playback_total = total_frames

    def _on_seek(self, val):
        """User dragged the seek slider."""
        # Allow seeking when paused (to jump to a new point)
        if self._processing and not self._paused:
            return
        frame_idx = int(val)
        self.input_preview.show_frame(frame_idx)

        # If paused during processing, record the seek target
        if self._processing and self._paused:
            self._seek_to_frame = frame_idx

        # If output is loaded and not processing, seek it too
        if not self._processing and self.output_preview._cap is not None:
            out_total = self.output_preview.total_frames
            in_total = max(self.input_preview.total_frames, 1)
            out_idx = int(frame_idx * out_total / in_total)
            out_idx = min(out_idx, out_total - 1)
            self.output_preview.show_frame(out_idx)

        self._update_time_label(frame_idx)

    def _update_time_label(self, frame_idx):
        fps = getattr(self, '_playback_fps', 30)
        total = getattr(self, '_playback_total', 0)
        cur_sec = frame_idx / fps if fps > 0 else 0
        tot_sec = total / fps if fps > 0 else 0
        self.time_label.config(
            text=f"{int(cur_sec // 60)}:{int(cur_sec % 60):02d} / "
                 f"{int(tot_sec // 60)}:{int(tot_sec % 60):02d}"
        )

    def _toggle_playback(self):
        if self._processing:
            return
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self.input_preview._cap is None:
            return
        self._playing = True
        self.play_btn.config(text="⏸")
        self._current_play_frame = self.seek_var.get()
        self._play_next_frame()

    def _stop_playback(self):
        self._playing = False
        self.play_btn.config(text="▶")
        if self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _play_next_frame(self):
        if not self._playing:
            return
        total = getattr(self, '_playback_total', 0)
        if self._current_play_frame >= total - 1:
            self._stop_playback()
            return

        self.input_preview.show_frame(self._current_play_frame)
        if self.output_preview._cap is not None:
            out_total = self.output_preview.total_frames
            out_idx = int(self._current_play_frame * out_total / max(total, 1))
            self.output_preview.show_frame(min(out_idx, out_total - 1))

        self.seek_var.set(self._current_play_frame)
        self._update_time_label(self._current_play_frame)

        self._current_play_frame += 1
        fps = getattr(self, '_playback_fps', 30)
        delay = max(1, int(1000 / fps))
        self._play_after_id = self.root.after(delay, self._play_next_frame)

    # =====================================================================
    # LOGGING
    # =====================================================================

    def _log(self, msg: str, tag: str = "info"):
        def _do():
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg + "\n", tag)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, _do)

    def _update_progress(self, pct, fps=0, status=""):
        def _do():
            self.progress_pct.set(pct)
            self._draw_progress()
            if status:
                self.status_text.set(status)
            if fps > 0:
                self.fps_label.config(text=f"{fps:.1f} FPS")
        self.root.after(0, _do)

    def _show_live_output_frame(self, frame):
        """Show a cropped frame in the output preview during processing."""
        self.root.after(0, lambda f=frame.copy(): self.output_preview.show_cv_frame(f))

    # =====================================================================
    # INLINE TRACKER REPOSITIONING (drag on input preview)
    # =====================================================================

    def _on_preview_mousedown(self, event):
        """Start drawing a selection rectangle."""
        if not self._processing:
            return
        self._drag_start = (event.x, event.y)
        if self._drag_rect_id:
            self.input_preview.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def _on_preview_mousemove(self, event):
        """Update the selection rectangle as user drags."""
        if not self._processing or self._drag_start is None:
            return
        if self._drag_rect_id:
            self.input_preview.canvas.delete(self._drag_rect_id)
        x0, y0 = self._drag_start
        self._drag_rect_id = self.input_preview.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#00ff00", width=2, dash=(4, 2),
        )

    def _on_preview_mouseup(self, event):
        """Finalize the selection — map to source coords and set pending ROI."""
        if not self._processing or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None
        if self._drag_rect_id:
            self.input_preview.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        # Ignore tiny accidental clicks
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return

        # Map canvas coords to source frame coords
        sx0, sy0 = self.input_preview.canvas_to_source(min(x0, x1), min(y0, y1))
        sx1, sy1 = self.input_preview.canvas_to_source(max(x0, x1), max(y0, y1))
        w = sx1 - sx0
        h = sy1 - sy0
        if w > 10 and h > 10:
            self._pending_roi = (sx0, sy0, w, h)
            self._log(f"\u27f3 New target drawn at ({sx0},{sy0}) {w}x{h} — syncing...", "info")

    # =====================================================================
    # PROCESSING
    # =====================================================================

    def _start_processing(self):
        inp = self.input_path.get().strip()
        out = self.output_path.get().strip()
        if not inp:
            self._log("❌ Select an input video first.", "error")
            return
        if not os.path.exists(inp):
            self._log(f"❌ File not found: {inp}", "error")
            return
        if not out:
            self._log("❌ Specify an output path.", "error")
            return

        # Clear log
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

        self._stop_playback()
        self._processing = True
        self._paused = False
        self._seek_to_frame = None
        self._reselect_requested = False
        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.stop_btn.config(state="normal")
        self.reselect_btn.config(state="normal")
        self.progress_pct.set(0)
        self._draw_progress()
        self.status_text.set("Initializing...")

        threading.Thread(target=self._run_cropper, daemon=True).start()

    def _toggle_pause(self):
        """Pause or resume processing. While paused the user can seek."""
        if not self._processing:
            return
        if self._paused:
            # Resume
            self._paused = False
            self.pause_btn.config(text="⏸  Pause", bg=C["yellow"])
            self._log("▶  Resumed.", "success")
            self._update_progress(status="Processing...")
        else:
            # Pause
            self._paused = True
            self.pause_btn.config(text="▶  Resume", bg=C["green"])
            self._log("⏸  Paused — scrub the slider to jump, then press Resume.", "warning")
            self._update_progress(status="Paused — seek and resume")

    def _stop_processing(self):
        """Stop processing and save whatever has been written so far."""
        self._processing = False
        self._paused = False  # unblock the pause loop so thread exits
        self._log("⏹  Stopping — saving progress...", "warning")

    def _request_reselect(self):
        """Signal the processing thread to pause and let the user reselect."""
        if self._processing:
            self._reselect_requested = True
            self._log("⟳ Reselect requested — pausing after current frame...", "info")
            self.reselect_btn.config(state="disabled")

    def _run_cropper(self):
        try:
            from smart_cropper import SmartCropper

            inp = self.input_path.get().strip()
            out = self.output_path.get().strip()

            cropper = SmartCropper(
                input_path=inp, output_path=out,
                model_name="yolo26n.pt",
                smoothing_window=self.smoothing.get(),
                confidence=self.confidence.get(),
                auto_select=False,
                start_time=self.start_time.get(),
                duration=self.duration.get(),
                zoom=self.zoom.get(),
                training_only=self.training_mode.get(),
            )

            mode_label = "TRAINING ONLY" if cropper.training_only else "normal"
            self._log(f"📐 {cropper.src_width}×{cropper.src_height} → "
                      f"{cropper.out_w}×{cropper.out_h} | "
                      f"{cropper._max_frames} frames | "
                      f"zoom {cropper.zoom:.1f}x | {mode_label}", "info")

            self._run_loop(cropper)

        except SystemExit as e:
            self._log(f"⚠️  {e}", "warning")
        except Exception as e:
            self._log(f"❌ {e}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
        finally:
            self._processing = False
            self._paused = False
            self.root.after(0, lambda: self.start_btn.config(state="normal"))
            self.root.after(0, lambda: self.pause_btn.config(state="disabled", text="⏸  Pause", bg=C["yellow"]))
            self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
            self.root.after(0, lambda: self.reselect_btn.config(state="disabled"))

    def _run_loop(self, cropper):
        t0 = time.time()
        training_only = cropper.training_only

        base, ext = os.path.splitext(cropper.output_path)
        silent = f"{base}_silent{ext}"

        # Skip writer in training-only mode
        writer = None
        if not training_only:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(silent, fourcc, cropper.fps,
                                     (cropper.out_w, cropper.out_h))

        fi = 0
        lost = 0
        sel = False
        src_frame_idx = cropper._start_frame

        self._update_progress(0, status="Select your target...")
        if training_only:
            self._log("🏋  TRAINING MODE — select both bots, no video output", "warning")
        self._log("👆 Select target in the popup window...", "info")

        while self._processing:
            if self._paused:
                time.sleep(0.05)
                continue

            if self._seek_to_frame is not None:
                target = self._seek_to_frame
                self._seek_to_frame = None
                abs_target = cropper._start_frame + target
                cropper.cap.set(cv2.CAP_PROP_POS_FRAMES, abs_target)
                src_frame_idx = abs_target
                cropper._prev_center = None
                cropper._drift_cooldown = 10
                self._log(f"⏩ Jumped to frame {target} — continuing from there.", "info")

            ret, frame = cropper.cap.read()
            if not ret:
                break
            frames_into_clip = int(cropper.cap.get(cv2.CAP_PROP_POS_FRAMES)) - cropper._start_frame
            if cropper.duration > 0 and frames_into_clip > cropper._max_frames:
                break

            if not sel:
                results = cropper.model.track(
                    frame, tracker="bytetrack.yaml", persist=True,
                    verbose=False, conf=cropper.confidence)
                mode, tid = cropper._select_target_interactive(frame, results)
                cropper._tracking_mode = mode
                cropper._target_track_id = tid
                sel = True
                self._log(f"✅ Mode: {mode.upper()}", "success")
                if cropper._secondary_tracker is not None:
                    self._log("✅ Secondary bot tracker active", "success")
                self._update_progress(0, status="Training..." if training_only else "Processing...")

            if cropper._tracking_mode == cropper.MODE_YOLO:
                results = cropper.model.track(
                    frame, tracker="bytetrack.yaml", persist=True,
                    verbose=False, conf=cropper.confidence)
                center = cropper._get_target_center_yolo(results)
            else:
                center = cropper._get_target_center_manual(frame)

            # --- Check for inline ROI drag ---
            if self._pending_roi is not None:
                new_roi = self._pending_roi
                self._pending_roi = None
                cropper._init_csrt_tracker(frame, new_roi)
                cropper._save_template(frame, new_roi)
                cropper._tracking_mode = cropper.MODE_MANUAL
                cropper._prev_center = None
                cropper._frames_tracked_ok = 0
                cropper._drift_cooldown = 20
                lost = 0
                self._log("\u2705 Tracker re-synced to new position.", "success")
                continue

            # --- Check for reselect button request ---
            if self._reselect_requested:
                self._reselect_requested = False
                self._update_progress(pct=0, status="Reselecting target...")
                self._log("\u27f3 Draw a new box around the target...", "info")
                try:
                    _, _ = cropper._manual_roi_select(frame)
                    cropper._prev_center = None
                    cropper._frames_tracked_ok = 0
                    cropper._drift_cooldown = 20
                    lost = 0
                    self._log("\u2705 Target reselected! Resuming...", "success")
                    self._update_progress(pct=0, status="Training..." if training_only else "Processing...")
                    self.root.after(0, lambda: self.reselect_btn.config(state="normal"))
                except SystemExit:
                    self._log("\u26a0\ufe0f Reselect cancelled, continuing with current target.", "warning")
                    self.root.after(0, lambda: self.reselect_btn.config(state="normal"))
                continue

            if center is not None:
                sx = cropper.smoother_x.update(center[0])
                sy = cropper.smoother_y.update(center[1])
                lost = 0
            else:
                sx = cropper.smoother_x.hold()
                sy = cropper.smoother_y.hold()
                lost += 1
                if sx is None:
                    sx = cropper.src_width / 2.0
                if sy is None:
                    sy = cropper.src_height / 2.0

            # Update secondary tracker + auto-label (for YOLO mode;
            # manual mode handles this internally via _get_target_center_manual)
            if cropper._tracking_mode == cropper.MODE_YOLO:
                sec_bbox = cropper._update_secondary_tracker(frame)
                if fi % cropper._label_save_interval == 0 and cropper._template_bbox is not None:
                    cropper._auto_label_save(frame, cropper._template_bbox, sec_bbox)

            # Crop and write (skip in training-only mode)
            cropped = None
            if not training_only:
                cropped = cropper._crop_frame(frame, sx, sy)
                writer.write(cropped)

            # Draw tracking boxes on input frame for preview
            display_frame = frame.copy()
            # Primary bot
            if cropper._template_bbox is not None:
                bx, by, bw, bh = [int(v) for v in cropper._template_bbox]
                color = (0, 255, 0) if lost == 0 else (0, 0, 255)
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 3)
                label = "BOT 1 (TRACKING)" if lost == 0 else "BOT 1 (LOST)"
                cv2.putText(display_frame, label, (bx, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            # Secondary bot
            if cropper._secondary_bbox is not None:
                bx2, by2, bw2, bh2 = [int(v) for v in cropper._secondary_bbox]
                cv2.rectangle(display_frame, (bx2, by2), (bx2 + bw2, by2 + bh2),
                              (255, 165, 0), 2)  # orange
                cv2.putText(display_frame, "BOT 2", (bx2, by2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)

            # Live preview
            if cropped is not None:
                self._show_live_output_frame(cropped)
            self.root.after(0, lambda f=display_frame:
                            self.input_preview.show_cv_frame(f))

            fi += 1
            total_expected = cropper._max_frames
            if fi % 20 == 0 or fi == total_expected:
                e = time.time() - t0
                pct = min(fi / max(total_expected, 1) * 100, 100)
                fps = fi / max(e, 0.001)
                st = "TRACKING" if lost == 0 else f"LOST ({lost}f)"
                mode_str = "Training" if training_only else "Processing"
                self._update_progress(pct, fps, f"{mode_str} — {st}")

            if fi % 100 == 0:
                e = time.time() - t0
                pct = min(fi / max(total_expected, 1) * 100, 100)
                fps = fi / max(e, 0.001)
                st = "TRACKING" if lost == 0 else f"LOST ({lost}f)"
                tag = "progress" if lost == 0 else "warning"
                lbl_info = f" | {cropper._label_count} labels" if cropper.auto_label else ""
                self._log(f"  {fi}/{total_expected} ({pct:.0f}%) | "
                          f"{fps:.1f} FPS | {st}{lbl_info}", tag)

        if writer is not None:
            writer.release()
        cropper.cap.release()

        # --- Results ---
        if fi == 0:
            self._log("⚠️  No frames were processed.", "warning")
            self._update_progress(0, status="Nothing to save")
            if os.path.exists(silent):
                os.remove(silent)
            return

        e = time.time() - t0
        stopped_early = not self._processing

        if training_only:
            # Training mode: just report label stats
            self._log(f"\n🏋  Training run complete: {fi} frames in {e:.1f}s "
                      f"({fi / max(e, 0.001):.1f} FPS)", "success")
            self._log(f"📊  {cropper._label_count} labeled frames saved to "
                      f"{cropper._label_dataset_dir}", "success")
            self._update_progress(100, status="Training complete!")
            # Clean up silent file if it was created
            if os.path.exists(silent):
                os.remove(silent)
        else:
            if stopped_early:
                dur_secs = fi / cropper.fps if cropper.fps > 0 else 0
                self._log(f"\n⏹  Stopped at {fi} frames ({dur_secs:.1f}s) in {e:.1f}s", "warning")
            else:
                self._log(f"\n✅ {fi} frames in {e:.1f}s ({fi / max(e, 0.001):.1f} FPS)", "success")

            self._update_progress(95, status="Muxing audio...")
            self._log("🔊 Muxing audio...", "info")
            cropper._mux_audio(cropper.input_path, silent, cropper.output_path)
            self._log(f"🎉 Saved: {cropper.output_path}", "success")
            self._update_progress(100, status="Complete!" if not stopped_early else "Saved (partial)")

            if cropper._label_count > 0:
                self._log(f"📊  Also saved {cropper._label_count} training labels", "info")

            out_path = cropper.output_path
            self.root.after(100, lambda: self._load_output_preview(out_path))

    def _load_output_preview(self, path):
        """Load the finished output video into the preview panel."""
        if os.path.exists(path):
            self.output_preview.load_video(path)
            self._log("📱 Output loaded — use ▶ to compare side-by-side.", "success")

    # =====================================================================
    # LIFECYCLE
    # =====================================================================

    def _on_close(self):
        self._processing = False
        self._paused = False
        self._stop_playback()
        self.input_preview.release()
        self.output_preview.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SmartCropperGUI()
    app.run()
