from ultralytics import YOLO

# 1. Load your custom trained weights
model = YOLO("/Users/michaelmandiberg/Documents/GitHub/taking-stock-yolo/runs/takingstock_c45_h200_4x_yolo26x/weights/best.pt")

# 2. Rerun evaluation on the Validation Set (Default behavior)
print("--- Evaluating Validation Set ---")
val_results = model.val(data="/Users/michaelmandiberg/Documents/GitHub/taking-stock-yolo/yolo_dataset/data.yaml", split="val", device='mps')

# 3. Rerun evaluation on the Training Set 
print("--- Evaluating Training Set ---")
train_results = model.val(data="/Users/michaelmandiberg/Documents/GitHub/taking-stock-yolo/yolo_dataset/data.yaml", split="train", device='mps')
