import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


COCO_TARGET_ALIASES = {
    "sofa": {"couch", "sofa"},
    "couch": {"couch", "sofa"},
    "chair": {"chair"},
    "bed": {"bed"},
    "table": {"dining table", "table"},
    "desk": {"dining table", "table", "desk"},
}


class TorchvisionTargetDetector:
    """
    Real RGB object detector based on torchvision Faster R-CNN.

    It only consumes RGB images.
    It does not use simulator object coordinates, semantic scene graph,
    target pose, or shortest-path oracle.
    """

    def __init__(self, score_threshold=0.45, device=None):
        self.score_threshold = score_threshold

        # Use CPU for robustness on macOS. Torchvision detection ops may be unstable on MPS.
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"[Detector] Loading Faster R-CNN on device: {self.device}")

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.categories = weights.meta["categories"]

        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, rgb):
        rgb = np.asarray(rgb)

        if rgb.ndim != 3:
            raise ValueError(f"Expected HxWxC RGB image, got shape {rgb.shape}")

        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # HWC uint8 -> CHW float32 in [0, 1]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    def detect(self, rgb, target_name: str):
        target_name = target_name.lower()
        allowed_labels = COCO_TARGET_ALIASES.get(target_name, {target_name})

        rgb_arr = np.asarray(rgb)
        h, w = rgb_arr.shape[:2]

        image_tensor = self._to_tensor(rgb_arr)

        with torch.no_grad():
            outputs = self.model([image_tensor])[0]

        best = None
        best_score = -1.0

        boxes = outputs["boxes"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()

        for box, label_id, score in zip(boxes, labels, scores):
            if score < self.score_threshold:
                continue

            label = self.categories[int(label_id)].lower()

            if label not in allowed_labels:
                continue

            x1, y1, x2, y2 = box.tolist()

            if float(score) > best_score:
                best_score = float(score)
                best = {
                    "label": label,
                    "target": target_name,
                    "score": float(score),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center_x": float(((x1 + x2) / 2.0) / w),
                    "center_y": float(((y1 + y2) / 2.0) / h),
                }

        return best

    def estimate_distance(self, depth, detection):
        if detection is None:
            return float("inf")

        depth = np.asarray(depth)

        x1, y1, x2, y2 = detection["bbox"]

        h, w = depth.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return float("inf")

        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 20.0)]

        if valid.size == 0:
            return float("inf")

        return float(np.median(valid))


# Keep this alias so existing run_demo.py can still import DetrTargetDetector.
DetrTargetDetector = TorchvisionTargetDetector


class DebugTargetDetector:
    """
    Old debug detector. Only for pipeline testing, not for final demo.
    """

    def __init__(self):
        self.steps = 0

    def detect(self, rgb, target_name: str):
        self.steps += 1

        if self.steps < 4:
            return None

        return {
            "label": target_name,
            "target": target_name,
            "center_x": 0.52,
            "center_y": 0.50,
            "bbox": [260, 180, 380, 320],
            "score": 0.90,
        }

    def estimate_distance(self, depth, detection):
        return 2.0
