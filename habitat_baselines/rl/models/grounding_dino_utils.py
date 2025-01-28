import os
import cv2
import torch
import pickle
import numpy as np
import supervision as sv
from typing import List

from PIL import Image
from torchvision.ops import box_convert
import matplotlib.pyplot as plt

import groundingdino.datasets.transforms as T
from habitat_baselines.rl.models.GroundingDINO.groundingdino.util.inference import load_model, predict_batch, annotate

PREFIX = os.path.dirname(os.path.realpath(__file__)) + "/GroundingDINO/"
GOAL_LIST = ['red', 'green', 'blue', 'yellow', 'white', 'pink', 'black', 'cyan']


def imshow(img, phrases, scores, option):
    if option == "filtered_":
        phrases = [GOAL_LIST[label] for label in phrases]
        scores = [str(round(score, 2)).split(".")[-1] for score in scores]
    else:
        scores = [str(round(score.item(), 2)).split(".")[-1] for score in scores]
    caption = "_".join(phrases)
    score_caption = "_".join(scores)

    plt.imshow(img)
    plt.axis("off")
    plt.savefig('/home/ailab/MCFMO_DINO/output_dir/' + option + caption + score_caption)


class GroundingDINO:
    def __init__(self, config_path=None, weight_path=None, cpu_only=True):
        if config_path is None:
            config_path = os.path.join(PREFIX, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        if weight_path is None:
            weight_path = os.path.join(PREFIX, "weights", "groundingdino_swint_ogc.pth")
        self.cpu_only = cpu_only
        self.model = load_model(config_path, weight_path, cpu_only)
        # self.knn = pickle.load(open('data/object_detection_models/knn_colors_aug.pkl', 'rb'))

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # image_pillow = Image.fromarray(cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB))
        image_pillow = Image.fromarray(image_bgr.astype(np.uint8))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.max(dim=1)[0].numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)

    def filter_pred(self, pred, img, threshold_knn=0.6):
        pred_boxes = pred.xyxy.tolist()
        pred_scores = pred.confidence.tolist()
        pred_labels = pred.class_id.tolist()

        # print("=================== <ORIGIN RESULT> ===================")
        # print(f"== [Boxes]: {pred.xyxy}")
        # print(f"== [Logits]: {pred.confidence}")
        # print(f"== [Phrases]: {pred.class_id}")
        # print("================== ORIGIN RESULT END ==================")

        res = {'boxes': [], 'scores': [], 'labels': []}
        # colors = []

        for idx, score in enumerate(pred_scores):
            box = pred_boxes[idx]
            res['boxes'].append(box)
            res['scores'].append(pred_scores[idx])
            res['labels'].append(pred_labels[idx])
        #     center_x = int((box[2] + box[0]) / 2)
        #     center_y = int((box[3] + box[1]) / 2)
        #     colors.append(img[center_y][center_x])
        #
        # if colors:
        #     boxes, scores, labels = [], [], []
        #     labels_tmp = self.knn.predict_proba(colors)
        #
        #     for idx, lab in enumerate(labels_tmp):
        #         idx_true = np.argmax(lab)
        #         if lab[idx_true] >= threshold_knn:
        #             boxes.append(res['boxes'][idx])
        #             scores.append(res['scores'][idx] * lab[idx_true])
        #             labels.append(idx_true)
        #
        #     res['boxes'] = boxes
        #     res['scores'] = scores
        #     res['labels'] = labels
        #
            # # Why do almost labels show up as 6(black)? => I will not use filtering.
            # print("=================== <KNN RESULT> ===================")
            # print(f"== [Boxes]: {res['boxes']}")
            # print(f"== [Logits]: {res['scores']}")
            # print(f"== [Phrases]: {res['labels']}")
            # print("================== KNN RESULT END ==================")
        return res

    def run(self, obs, box_thresh, text_thresh, save=False):
        classes = ["red cylinder", "green cylinder", "blue cylinder", "yellow cylinder", "white cylinder",
                   "pink cylinder", "black cylinder", "cyan cylinder"]
        caption = ". ".join(classes)

        images = []
        for image in obs:
            # * load_image_batch(): Tensor Input must be (C, H, W) <-> Numpy Input must be (H, W, C)
            processed_image = GroundingDINO.preprocess_image(image_bgr=image.cpu().numpy())  # .to("cuda")
            images.append(processed_image)

        images = torch.stack(images, dim=0)
        boxes, logits, phrases = predict_batch(
            model=self.model,
            images=images,
            caption=caption,
            box_threshold=box_thresh,
            text_threshold=text_thresh
        )

        source_h, source_w, _ = images[0].shape
        results = []
        for i, box in enumerate(boxes):
            detection = GroundingDINO.post_process_result(
                source_h=source_h,
                source_w=source_w,
                boxes=box,
                logits=logits[i]
            )

            class_id = GroundingDINO.phrases2classes(phrases=phrases[i], classes=classes)
            detection.class_id = class_id
            results.append(detection)

        res = [self.filter_pred(r, obs[i].cpu().numpy()) for i, r in enumerate(results)]

        if save:
            for i, image in enumerate(obs):
                if len(res[i]['boxes']) > 0:
                    # annotated_frame = (
                    #     annotate(image_source=image.cpu().numpy().astype(np.uint8), boxes=boxes[i],
                    #              logits=logits[i].max(dim=1)[0], phrases=phrases[i]))
                    # imshow(annotated_frame, phrases[i], logits[i].max(dim=1)[0], "original_")

                    filtered_annotated_frame = (
                        annotate(image_source=image.cpu().numpy().astype(np.uint8), boxes=torch.Tensor(res[i]['boxes']),
                                 logits=torch.Tensor(res[i]['scores']), phrases=res[i]['labels']))
                    imshow(filtered_annotated_frame, res[i]['labels'], res[i]['scores'], "filtered_")

        return res
