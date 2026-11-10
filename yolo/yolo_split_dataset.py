import shutil
from pathlib import Path
import random
import os
from collections import Counter
import yaml

SKIP_LIST = []
SKIP_SET = {int(class_id) for class_id in SKIP_LIST}


def split_dataset(source_images, source_labels, output_dir, class_id_to_YOLOid=None, train_ratio=0.8, removed_class_counts=None):
    """Split dataset into train/val"""
    if class_id_to_YOLOid is None:
        class_id_to_YOLOid = {}
    if removed_class_counts is None:
        removed_class_counts = Counter()

    normalized_class_id_to_YOLOid = {
        int(source_class_id): int(yolo_id)
        for source_class_id, yolo_id in class_id_to_YOLOid.items()
    }
    print(f"Normalized class_id_to_YOLOid mapping: {normalized_class_id_to_YOLOid}")
    def convert_label_file(input_path, output_path, class_id_to_YOLOid):
        print(f"mapping is {class_id_to_YOLOid}")
        """Read label file, convert class IDs to YOLO IDs, and write to output"""
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line_number, line in enumerate(lines, start=1):
            # print(f"Processing line {line_number} in {input_path}: {line.strip()}")
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    old_class_id = int(parts[0])
                    # print(f"Original class_id '{old_class_id}' in {input_path}:{line_number}")
                    if old_class_id in SKIP_SET:
                        continue

                    if old_class_id not in class_id_to_YOLOid:
                        removed_class_counts[old_class_id] += 1
                        print(
                            f"Removed unmapped label in {input_path}:{line_number} "
                            f"(class_id={old_class_id})"
                        )
                        continue

                    yolo_id = class_id_to_YOLOid[old_class_id]
                    if yolo_id in SKIP_SET:
                        continue

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    left = max(0.0, min(1.0, x_center - width / 2))
                    right = max(0.0, min(1.0, x_center + width / 2))
                    top = max(0.0, min(1.0, y_center - height / 2))
                    bottom = max(0.0, min(1.0, y_center + height / 2))

                    clipped_width = right - left
                    clipped_height = bottom - top
                    if clipped_width <= 0 or clipped_height <= 0:
                        continue

                    parts[0] = str(yolo_id)
                    parts[1] = f"{(left + right) / 2:.6f}"
                    parts[2] = f"{(top + bottom) / 2:.6f}"
                    parts[3] = f"{clipped_width:.6f}"
                    parts[4] = f"{clipped_height:.6f}"
                    converted_lines.append(' '.join(parts) + '\n')
                except Exception as e:
                    print(f"Error converting line '{line.strip()}' in {input_path}: {e}")
        # print(f"Converted {len(converted_lines)} lines for {input_path}")
        if not converted_lines:
            print(f" ❌❌❌ Skipping empty label (no valid annotations after conversion): {input_path}")
            return
        with open(output_path, 'w') as f:
            f.writelines(converted_lines)
    
    # Get all images
    images = list(Path(source_images).glob('*.jpg')) + \
             list(Path(source_images).glob('*.png')) + \
             list(Path(source_images).glob('*.jpeg'))
    
    # Shuffle
    random.seed(42)  # For reproducibility
    random.shuffle(images)
    
    # Split
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Total images: {len(images)}")
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Create directories
    for split in ['train', 'val']:
        (Path(output_dir) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copy train files
    for img in train_images:
        shutil.copy(img, Path(output_dir) / 'images' / 'train' / img.name)
        label = Path(source_labels) / f"{img.stem}.txt"
        if label.exists():
            output_label = Path(output_dir) / 'labels' / 'train' / label.name
            convert_label_file(label, output_label, normalized_class_id_to_YOLOid)
        else:
            print(f" ❌❌❌ Missing label file for image: {img}, expected at: {label}")
            removed_class_counts['missing_label'] += 1
    
    # Copy val files
    for img in val_images:
        shutil.copy(img, Path(output_dir) / 'images' / 'val' / img.name)
        label = Path(source_labels) / f"{img.stem}.txt"
        if label.exists():
            output_label = Path(output_dir) / 'labels' / 'val' / label.name
            convert_label_file(label, output_label, normalized_class_id_to_YOLOid)
        else:
            print(f" ❌❌❌ Missing label file for image: {img}, expected at: {label}")
            removed_class_counts['missing_label'] += 1

    return removed_class_counts

# Usage - adjust paths to your export
SORTED_IMAGES_FOLDER = os.path.join(os.path.expanduser("~"), "Documents/YOLO_Training_Data/sorted_images")
YOLO_READY_DATASET_FOLDER = os.path.join(os.path.expanduser("~"), "/Users/michaelmandiberg/Documents/GitHub/DATA-78000/yolo_dataset")
YAML_FILE_PATH = os.path.join(SORTED_IMAGES_FOLDER, 'data.yaml')
# get all folders in SORTED_IMAGES_FOLDER
folders = [f.path for f in os.scandir(SORTED_IMAGES_FOLDER) if f.is_dir()]

# Collect class information
class_names = {}
class_id = 0
class_name_to_YOLOid = {}
class_id_to_YOLOid = {}

# load existing class names from data.yaml if it exists
if os.path.exists(YAML_FILE_PATH):
    with open(YAML_FILE_PATH, 'r', encoding='utf-8') as f:
        existing_yaml = yaml.safe_load(f)
        if 'names' in existing_yaml:
            for idx, name in existing_yaml['names'].items():
                class_names[int(idx)] = name
                # create 1:1 mapping, as hack to deal with existing class IDs that are already YOLO IDs
                class_name_to_YOLOid[name] = int(idx)
                class_id_to_YOLOid[int(idx)] = int(idx)
                print(f"Loaded existing class from YAML: {idx} -> {name}")

else:
    print(f"No existing YAML found at {YAML_FILE_PATH}, building class mapping from folder names.")
    # build class_names dict first
    for folder in folders:
        class_name = os.path.basename(folder)
        if class_name[0].isdigit():
            class_names[class_id] = class_name
            class_id += 1

    # create reverse mapping
    class_name_to_YOLOid = {v: k for k, v in class_names.items()}
    for class_name, yolo_id in class_name_to_YOLOid.items():
        class_id = class_name.split('_')[0]
        # print(f"Class '{class_id}' -> YOLO ID: {yolo_id}")
        class_id_to_YOLOid[class_id] = yolo_id
print(f"\nClass ID to YOLO ID mapping: {class_id_to_YOLOid}")

removed_class_counts = Counter()

for folder in folders:
    class_name = os.path.basename(folder)
    this_folder = os.path.join(SORTED_IMAGES_FOLDER,folder)
    print(f"Processing folder: {this_folder}")
    this_folder_images = os.path.join(this_folder, 'images')
    
    # Check if images folder exists and has files
    if not os.path.exists(this_folder_images):
        print(f"Images folder doesn't exist: {this_folder_images}, skipping.")
        continue
    
    files = os.listdir(this_folder_images)
    if len(files) == 0:
        print(f"No images found in {this_folder_images}, skipping.")
        continue

    # this is for handling some kind of legacy edge case
    if not class_name[0].isdigit():
        print(f"Class name '{class_name}' does not start with a digit, it is type of {type(class_name)}, doing some other path to splitting.")
    #     # Store class information
    #     class_names[class_id] = class_name
    #     class_id += 1
    # else:
    #     print(f"need to move '{class_name}' directly as it won't be added to class_id_to_YOLOid.")
        split_dataset(
            source_images= this_folder_images,
            source_labels=os.path.join(this_folder, 'labels'),
            output_dir=YOLO_READY_DATASET_FOLDER,
            class_id_to_YOLOid=class_id_to_YOLOid,
            train_ratio=0.8,
            removed_class_counts=removed_class_counts
        )

# Create YOLO data.yaml file
yaml_data = {
    'path': YOLO_READY_DATASET_FOLDER,
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(class_names),
    'names': class_names
}

yaml_path = os.path.join(YOLO_READY_DATASET_FOLDER, 'data.yaml')
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

# write a YOLO classes.txt file for reference
classes_txt_path = os.path.join(YOLO_READY_DATASET_FOLDER, 'classes.txt')
with open(classes_txt_path, 'w', encoding='utf-8') as f:
    for class_id in sorted(class_names.keys()):
        f.write(f"{class_names[class_id]}\n")

print(f"\nCreated YOLO config file: {yaml_path}")
print(f"Total classes: {len(class_names)}")
print(f"Classes: {class_names}")


for class_name, yolo_id in class_name_to_YOLOid.items():
    this_folder = os.path.join(SORTED_IMAGES_FOLDER,class_name)
    print(f"Processing folder: {this_folder}")
    this_folder_images = os.path.join(this_folder, 'images')

    split_dataset(
        source_images= this_folder_images,
        source_labels=os.path.join(this_folder, 'labels'),
        output_dir=YOLO_READY_DATASET_FOLDER,
        class_id_to_YOLOid=class_id_to_YOLOid,
        train_ratio=0.8,
        removed_class_counts=removed_class_counts
    )

total_removed_labels = sum(removed_class_counts.values())
print(f"\nTotal unmapped labels removed: {total_removed_labels}")
if total_removed_labels:
    print("Removed label counts by class_id:")
    for class_id in sorted(removed_class_counts, key=lambda class_key: int(class_key) if str(class_key).isdigit() else str(class_key)):
        print(f"  {class_id}: {removed_class_counts[class_id]}")
