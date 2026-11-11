import os

'''
This script walks through a hashed directory structure
skips any non-directory files
and makes each image_path available for processing
'''


PATH = "/Volumes/OWC4/DATA7800_YOLO/images_123rf"

for folder in os.listdir(PATH):
    folder_path = os.path.join(PATH, folder)
    if not os.path.isdir(folder_path):
        continue
    print(f"Processing folder: {folder}")
    print(f"Processing folder: {folder_path}")
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        print(f"Processing subfolder: {subfolder}")
        print(f"Processing subfolder: {subfolder_path}")
        for file in os.listdir(subfolder_path):
            if file.lower().endswith(".jpg"):
                image_path = os.path.join(subfolder_path, file)
                print(f"Processing image: {image_path}")
                # here you can imread the image_path and do stuff with the image