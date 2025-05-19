from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolo-custom",
    project="runs/train"
)

# Optional: Evaluate on val set
metrics = model.val()