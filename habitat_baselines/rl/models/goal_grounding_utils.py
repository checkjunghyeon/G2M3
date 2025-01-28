import numpy as np
import torch


def _build_goal_image(depth, results, device):
    B, H, W = depth.shape
    goal_image = torch.zeros(B, 8, H, W, device=device)

    for i, res in enumerate(results):
        boxes = np.round(res['boxes']).astype(int)  # Batch of bounding boxes [N, 4]
        labels = res['labels']  # Batch of labels [N, 1]
        scores = res['scores']  # Batch of scores [N, 1]

        N = len(boxes)
        for j in range(N):
            b = boxes[j]
            label = labels[j]
            score = round(scores[j], 4)

            goal_image[i, label, b[1]:b[3], b[0]:b[2]] = score
    return goal_image  # [B, C, H, W]


# ablation 1: Faster R-CNN
def build_goal_image_fasterrcnn(observations, detector, device):
    depth = (observations['depth'] * 10).squeeze(-1).cpu().numpy()
    depth[depth == 0] = np.NaN

    results = detector.predict(observations['rgb'].squeeze(-1))
    goal_image = _build_goal_image(depth, results, device)

    return goal_image


# ours: Grounding-DINO
def build_goal_image_groudingdino(observations, detector, device, box_thresh=0.41, text_thresh=0.25):
    depth = (observations['depth'] * 10).squeeze(-1).cpu().numpy()
    depth[depth == 0] = np.NaN

    results = detector.run(observations['rgb'], box_thresh=box_thresh, text_thresh=text_thresh, save=False)
    goal_image = _build_goal_image(depth, results, device)

    return goal_image
