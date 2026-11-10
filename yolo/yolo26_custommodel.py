from pathlib import Path

import cv2
from ultralytics import YOLO

# Resolve the local custom model relative to this repo instead of using a machine-
# specific absolute path.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    BASE_DIR / "takingstock_flags_nov9_yolo26n" / "weights" / "best.pt",
    BASE_DIR / "yolo" / "runs" / "takingstock_flags_nov9_yolo26n" / "weights" / "best.pt",
    BASE_DIR / "yolo26n.pt",
]
MODEL_PATH = "next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])"

# YOLO26 custom models are standard YOLO checkpoints, not YOLOE text-prompt models.
model = YOLO(str(MODEL_PATH))

# Use a local image path that exists in the repo.
image_path = BASE_DIR / "computer_vision" / "1.00_17472115_YOLO_debug.jpg"
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

results = model.predict(str(image_path), conf=0.25, verbose=False)[0]

img = cv2.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
    x1, y1, x2, y2 = map(int, box)
    class_id = int(cls)
    confidence = float(score)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{model.names[class_id]}: {confidence:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("detected_objects.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
