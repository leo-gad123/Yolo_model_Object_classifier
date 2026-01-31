from ultralytics import YOLO
import csv
from collections import defaultdict

# Load model
model = YOLO("yolov8s.pt")

# Run detection on video
results = model("video.mp4", stream=True)

# Dictionary to store counts
class_counts = defaultdict(int)

# Process frames
for r in results:
    if r.boxes is not None:
        classes = r.boxes.cls.tolist()
        for cls in classes:
            class_name = model.names[int(cls)]
            class_counts[class_name] += 1

# Save to CSV
with open("detections_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Object Name", "Total Count"])
    for obj, count in class_counts.items():
        writer.writerow([obj, count])

print("✅ CSV saved as detections_summary.csv")