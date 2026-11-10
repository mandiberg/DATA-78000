import os
import cv2
import json
from pathlib import Path

# ---- paths ----
# DATASET_ROOT = "/Volumes/OWC52/segment_images_OWC4/test_output/sort/move_these"
# DATASET_ROOT = "/Users/michaelmandiberg/Documents/yolo/sorted_images_balls_yoga/misc_balls_detect_round2"
DATASET_ROOT = "/Volumes/SD128/DATA7800_YOLO"

IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
CLASSES_FILE = os.path.join(DATASET_ROOT, "classes.txt")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "images_detections")
classes_dict = {}
SAVE_TO_CLASS_FOLDERS = True  # Set to True to save images in class-named subfolders
BOX_COLOR = (0, 255, 0)  # Green color for bounding boxes
BOX_COLOR = (0, 0, 255)  # Blue color for bounding boxes

# Alternative: Single folder with both images and JSON labels
SINGLE_FOLDER_MODE = False  # Set to True for single folder with JSON labels
SINGLE_FOLDER_PATH = Path("/Users/michaelmandiberg/Documents/projects-active/facemap_production/labeling_round2/osama200")

TRAIN_VAL_MODE = False  # Set to True if dataset has train/val subfolders
if TRAIN_VAL_MODE and not SINGLE_FOLDER_MODE:
    IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "train")
    LABELS_DIR = os.path.join(DATASET_ROOT, "labels", "train")
    CLASSES_FILE = os.path.join(DATASET_ROOT, "data.yaml")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- load class names ----
if not SINGLE_FOLDER_MODE and not TRAIN_VAL_MODE:
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: {CLASSES_FILE} not found. Will use class IDs as labels.")
        classes = []
elif TRAIN_VAL_MODE:
    # this creates a list with placeholders for none existing class IDs
    # this maps list index to class_id to class_name
    classes_raw = []  # Will be loaded from data.yaml in this mode
    classes = []
    # load class names from data.yaml
    import yaml
    with open(CLASSES_FILE, 'r') as f:
        data = yaml.safe_load(f)
        classes_raw = data.get('names', [])
    print(f"Raw classes from data.yaml: {classes_raw}")
    for train_id, class_id_name in enumerate(classes_raw):
        print(f"train_id {train_id} class_id_name: {class_id_name}", type(class_id_name), classes_raw[class_id_name])
        class_id, class_name = classes_raw[class_id_name].split('_')
        classes_dict[train_id] = class_name
        classes.append(class_name)
    print(f"Loaded classes from data.yaml: {classes_raw} and mapping: {classes_dict}")
    max_class_id = max(int(cid) for cid in classes_dict.keys())
    print(f"Max class ID: {max_class_id}")
else:
    classes = []  # For JSON format, labels are in the JSON itself

print(f"Loaded classes: {classes}")

# ---- helper: YOLO -> pixel bbox ----
def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2


def json_to_pixel_bbox(points, img_w, img_h):
    """Convert JSON rectangle format (pixel coords) to pixel bbox"""
    x1, y1 = points[0]
    x2, y2 = points[1]
    return int(x1), int(y1), int(x2), int(y2)


def process_yolo_format():
    """Process images and labels in YOLO format"""
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(image_files)} images: {image_files}")

    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)
        print(f"\nProcessing image: {img_name}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image: {img_name}")
            continue

        h, w = image.shape[:2]

        # Find corresponding label file: label filename contains image filename
        label_file = None
        for f in os.listdir(LABELS_DIR):
            # file_name_no_ext = 
            if os.path.splitext(img_name)[0] in f:
                label_file = os.path.join(LABELS_DIR, f)
                break

        if label_file is None:
            print(f"No label file found for image: {img_name}")
            out_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(out_path, image)
            continue

        print(f"Using label file: {os.path.basename(label_file)}")

        with open(label_file, "r") as f:
            lines = f.readlines()
            print(f"Label file contains {len(lines)} lines")
            label_set = set()

            for i, line in enumerate(lines):
                parts = line.strip().split()
                print(f"Line {i}: {parts}")

                if len(parts) != 5:
                    print(f"Skipping line {i} (not 5 elements)")
                    continue

                class_id, xc, yc, bw, bh = parts
                try:
                    class_id = int(class_id)
                    xc, yc, bw, bh = map(float, [xc, yc, bw, bh])
                except Exception as e:
                    print(f"Skipping line {i} due to parse error: {e}")
                    continue

                x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
                label = classes[class_id] if class_id < len(classes) else str(class_id)

                # draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (BOX_COLOR), 1)
                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                label_set.add(label)
                print(f"Drew box for class '{label}' at ({x1},{y1},{x2},{y2})")
        # Save output image
        if SAVE_TO_CLASS_FOLDERS and label_set:
            class_string = "_".join(sorted(label_set))
            class_folder = os.path.join(OUTPUT_DIR, class_string)
            os.makedirs(class_folder, exist_ok=True)
            out_path = os.path.join(class_folder, img_name)
            cv2.imwrite(out_path, image)
            print(f"Saved annotated image to {out_path}")
        else:
            out_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(out_path, image)
            print(f"Saved annotated image to {out_path}")


def process_json_format():
    """Process images and labels in JSON format (same folder)"""
    image_files = [f for f in os.listdir(SINGLE_FOLDER_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(image_files)} images: {image_files}")

    for img_name in image_files:
        img_path = os.path.join(SINGLE_FOLDER_PATH, img_name)
        print(f"\nProcessing image: {img_name}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image: {img_name}")
            continue

        h, w = image.shape[:2]

        # Find corresponding JSON label file
        json_path = Path(img_path).with_suffix('.json')
        
        if not json_path.exists():
            print(f"No JSON label file found for image: {img_name}")
            out_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(out_path, image)
            continue

        print(f"Using label file: {json_path.name}")

        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        shapes = label_data.get("shapes", [])
        print(f"JSON file contains {len(shapes)} shapes")

        for i, shape in enumerate(shapes):
            if shape.get("shape_type") != "rectangle":
                print(f"Skipping shape {i} (not a rectangle)")
                continue

            try:
                points = shape["points"]
                label = shape["label"]
                
                x1, y1, x2, y2 = json_to_pixel_bbox(points, w, h)
            except Exception as e:
                print(f"Skipping shape {i} due to parse error: {e}")
                continue

            # draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            print(f"Drew box for class '{label}' at ({x1},{y1},{x2},{y2})")

        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, image)
        print(f"Saved annotated image to {out_path}")


# ---- main execution ----
if SINGLE_FOLDER_MODE:
    process_json_format()
else:
    process_yolo_format()

print("\nDone. All images processed.")
