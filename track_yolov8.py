from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/train/yolo-custom/weights/best.pt")

# Run tracking on a video
results = model.track(
    source="Traffic IP Camera video.mp4",
    show=True,
    tracker="bytetrack.yaml",
    persist=True
)

# Count unique track IDs per class
unique_ids_per_class = {}

for frame in results:
    if frame.boxes is not None and frame.boxes.id is not None:
        class_ids = frame.boxes.cls.cpu().numpy().astype(int)
        track_ids = frame.boxes.id.cpu().numpy().astype(int)

        for cls_id, tid in zip(class_ids, track_ids):
            if cls_id not in unique_ids_per_class:
                unique_ids_per_class[cls_id] = set()
            unique_ids_per_class[cls_id].add(tid)

# Print total count per class
print("🔢 Object Counts by Class:")
for cls_id, ids in unique_ids_per_class.items():
    print(f"Class {cls_id}: {len(ids)} objects tracked")
