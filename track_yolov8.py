# track_yolov8.py
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/train/yolo-custom/weights/best.pt")

# Run ByteTrack tracking
results = model.track(
    source=1,
    show=True,
    tracker="bytetrack.yaml"
)