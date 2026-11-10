from ultralytics import YOLO
import os
import torch
from ultralytics.utils.tal import TaskAlignedAssigner

# --- Workaround for PyTorch MPS non-adjacent advanced indexing bug ---
# pd_scores[ind[0], :, ind[1]] mixes advanced indices on dim 0 & 2 with a
# basic slice on dim 1. On MPS (PyTorch 2.9.1) this produces inconsistent
# output shapes, causing a shape mismatch RuntimeError in tal.py:195.
# Fix: permute pd_scores so the two advanced indices become adjacent.
def _mps_fixed_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    na = pd_bboxes.shape[-2]
    mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
    overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
    bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

    # PyTorch 2.9.1 MPS has two bugs in the original implementation:
    #   1. Non-adjacent advanced indexing (pd_scores[ind0, :, ind1]) gives wrong shapes.
    #   2. Boolean indexing on .expand()-ed (stride-0) tensors miscounts True elements.
    # Fix: use nonzero() to convert mask_gt to explicit integer indices, then use
    # all-adjacent integer indexing throughout — no boolean indexing, no expanded tensors.
    mask_idx = mask_gt.nonzero(as_tuple=False)  # [N, 3]: columns = (batch, obj, anchor)
    if mask_idx.numel() > 0:
        b_idx   = mask_idx[:, 0]  # [N]
        obj_idx = mask_idx[:, 1]  # [N]
        anc_idx = mask_idx[:, 2]  # [N]

        # Class label for each (batch, obj) pair → used to index the score dim
        # gt_labels is float32 in Ultralytics; must cast to long before using as index
        cls_idx = gt_labels.squeeze(-1)[b_idx, obj_idx].long()  # [N]

        # pd_scores: [bs, na, nc] — simultaneous integer indexing on all 3 dims → [N]
        bbox_scores[b_idx, obj_idx, anc_idx] = pd_scores[b_idx, anc_idx, cls_idx]

        # pd_bboxes: [bs, na, 4], gt_bboxes: [bs, n_max_boxes, 4] → [N, 4] each
        pd_boxes = pd_bboxes[b_idx, anc_idx]   # [N, 4]
        gt_boxes = gt_bboxes[b_idx, obj_idx]   # [N, 4]
        overlaps[b_idx, obj_idx, anc_idx] = self.iou_calculation(gt_boxes, pd_boxes)  # [N]

    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps

TaskAlignedAssigner.get_box_metrics = _mps_fixed_get_box_metrics
# --- End workaround ---

# Load pretrained model
model = YOLO('yolo26n.pt')  # Options: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26x-objv1-150.pt
# model2 = YOLO('yolov8x.pt')  # Options: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26x-objv1-150.pt
# model2 = YOLO('yolo26x-objv1-150.pt')  # Options: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26x-objv1-150.pt
# DATA_FOLDER = "/Users/michael.mandiberg/Documents/takingstock_production/labeled_images_nov19"
    
# Source - https://stackoverflow.com/a
# Posted by the_artemi8
# Retrieved 2025-12-08, License - CC BY-SA 4.0

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

model.add_callback("on_train_start", freeze_layer)
# model2.add_callback("on_train_start", freeze_layer)
# model.train(data="./dataset.yaml")

# if MBP use:
    # batch=16,        # Fixed small batch for low-memory debugging on 32GB RAM


# Train
results = model.train(
    data="/Users/michaelmandiberg/Documents/GitHub/DATA-78000/yolo_dataset/data.yaml",  # absolute path
    epochs=40,
    imgsz=640,
    batch=-1,       # Reduce if you get memory errors
    name='takingstock_flags_nov9_yolo26n',  # Experiment name
    patience=20,    # Early stopping
    device='mps',       # mps for Mac with M1/M2/M3 chips, else 'cuda' or 'cpu'
    workers=8,        # Lower host RAM pressure while debugging MPS crashes
    project='/Users/michaelmandiberg/Documents/GitHub/DATA-78000/yolo/runs',  # Save to project directory
    exist_ok=True,  # Overwrite existing experiment with same name
    augment=True,
    cache=False    # Avoid large RAM spikes during debugging
)

# results2 = model2.train(
#     data="/Users/michaelmandiberg/Documents/GitHub/DATA-78000/yolo_dataset2/data.yaml",  # absolute path
#     epochs=100,
#     imgsz=640,
#     batch=16,       # Reduce if you get memory errors
#     name='takingstock_glassescards_v0_yolov8x',  # Experiment name
#     patience=20,    # Early stopping
#     device='mps',       # mps for Mac with M1/M2/M3 chips, else 'cuda' or 'cpu'
#     workers=8,        # ✅ CPU workers, not GPU cores
#     project='/Users/michael.mandiberg/Documents/GitHub/taking-stock-yolo/runs',  # Save to project directory
#     exist_ok=True,  # Overwrite existing experiment with same name
#     augment=True,
#     cache='ram'   # ⭐ Enable RAM caching
# )

# Source - https://stackoverflow.com/a
# Posted by the_artemi8
# Retrieved 2025-12-08, License - CC BY-SA 4.0

