# Havoc Rundown Shorts Creator

Convert landscape (1920×1080) videos to portrait (1080×1920) with intelligent object tracking.

The cropper uses **YOLO26** to detect and track objects, lets you **click to select** which object to follow, and smoothly pans a 9:16 crop window to keep it centered. Audio is preserved via ffmpeg.

## Requirements

- **Python 3.10+**
- **ffmpeg** on your system PATH (for audio muxing)

## Installation

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — video I/O and frame manipulation
- `ultralytics` — YOLOv8 object detection and tracking
- `requests` — used by the test script to download sample videos

The YOLO26 nano model (`yolo26n.pt`) will be auto-downloaded on first run.

## Usage

### Basic (Interactive Mode)

```bash
python smart_cropper.py --input my_video.mp4 --output portrait.mp4
```

A window will pop up showing the first frame with all detected objects highlighted.
**Click on the object** you want to track, or press **M** to draw a manual box around any object.

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | *(required)* | Input landscape video path |
| `--output`, `-o` | `output_portrait.mp4` | Output portrait video path |
| `--model`, `-m` | `yolo26n.pt` | YOLO model weights |
| `--smoothing`, `-s` | `15` | Moving-average window size (frames) |
| `--confidence`, `-c` | `0.3` | Minimum detection confidence |
| `--auto-select` | off | Skip interactive selection, auto-pick largest object |

### Examples

```bash
# Track a specific object with higher confidence threshold
python smart_cropper.py -i gameplay.mp4 -o gameplay_portrait.mp4 -c 0.5

# Wider smoothing for slower, more cinematic panning
python smart_cropper.py -i drone_shot.mp4 -o drone_portrait.mp4 -s 30

# Auto-select mode (no popup window)
python smart_cropper.py -i input.mp4 -o output.mp4 --auto-select
```

## How It Works

1. **Detect** — YOLO26n scans the first frame and displays all found objects.
2. **Select** — You click on the object to track.
3. **Track** — ByteTrack maintains the object's identity across frames.
4. **Smooth** — A Simple Moving Average (SMA) over the last 15 frames stabilizes the X-coordinate, preventing jitter.
5. **Crop** — A 9:16 vertical slice (607×1080 for 1080p source) is extracted, centered on the smoothed position.
6. **Resize** — The crop is scaled to 1080×1920.
7. **Audio** — ffmpeg copies the original audio track into the final output.

### When the object is lost

If the tracked object leaves the frame or is briefly occluded, the crop window **holds its last known position** until the object reappears.

## Running Tests

```bash
python test_cropper.py
```

This will:
1. Download a free sample 1080p video (~10s)
2. Run the cropper in auto-select mode
3. Validate the output (resolution, frame count)
4. Print a pass/fail report

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio in output | Install ffmpeg: https://ffmpeg.org/download.html and ensure it's on your PATH |
| `No objects detected` | Lower the `--confidence` threshold (e.g., `-c 0.1`) |
| Jittery panning | Increase `--smoothing` (e.g., `-s 30`) |
| Slow processing | Ensure you're using `yolo26n.pt` (nano). A GPU with CUDA will significantly improve speed. |
