import cv2
from ultralytics import YOLOE

USE_TEXT_PROMPT = True  # Set to False to use a prompt-free model
TEXT_PROMPT = "hairstyle"

# Load the correct YOLOE checkpoint for the selected prompting mode.
# Prompt-free models end with '-pf.pt' and do not support text class prompts.
# Text-prompt models (without '-pf') support model.set_classes([...]).
model_path = "yoloe-v8l-seg.pt" if USE_TEXT_PROMPT else "yoloe-v8l-seg-pf.pt"
model = YOLOE(model_path)

# 2. Run inference on your target image
image_path = "1.00_17472115_YOLO_debug.jpg"

if USE_TEXT_PROMPT:
    subject = input(f"Enter the object to detect (default: {TEXT_PROMPT}): ").strip()
    if not subject:
        subject = TEXT_PROMPT

    # Text-prompt YOLOE models accept user-defined subjects as class names.
    model.set_classes([subject])

results = model.predict(image_path, conf=0.25, verbose=False)[0]

# 3. Read the image with OpenCV to overlay detections
img = cv2.imread(image_path)

# 4. Extract boxes, class labels, and confidence metrics
for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
    # Convert box coordinates to absolute pixel integers
    x1, y1, x2, y2 = map(int, box)
    class_id = int(cls)
    confidence = float(score)
    
    # Draw the boundary rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Label the object with its identified class string
    label = f"{model.names[class_id]}: {confidence:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save or display the mapped detections
cv2.imshow("detected_objects.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
