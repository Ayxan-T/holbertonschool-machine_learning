#!/usr/bin/env python3
""" Module: 1-yolo """

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


class Yolo:
    """ Yolo Darknet Keras model """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t  # IOU threshold
        self.anchors = anchors  # (num_outputs, anchor_boxes, 2)

    def process_outputs(self, outputs, image_size):
        """Process model output.

        Args:
            outputs (list): a list of numpy.ndarrays,
                each of shape (grid_height, grid_width, anchor_boxes,
                4 + 1 + classes)
            image_size (numpy.ndarray): [image_height, image_width]

        Returns:
            boxes (list): list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)
            box_confidences (list): list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs (list): list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # anchors for this output scale
            anchors = np.array(self.anchors[i], dtype=float)
            anchors = anchors.reshape(1, 1, anchor_boxes, 2)

            # cell size in the original image
            cell_w = image_size[1] / grid_w
            cell_h = image_size[0] / grid_h

            # grid indices for this output
            y_index, x_index = np.meshgrid(
                np.arange(grid_h),
                np.arange(grid_w),
                indexing='ij'
            )
            x_index = x_index[:, :, None]
            y_index = y_index[:, :, None]

            # t_x, t_y, t_w, t_h
            txty_sigmoid = sigmoid(output[:, :, :, :2])
            twth_exp = np.exp(output[:, :, :, 2:4]) * anchors

            # centers in the original image
            center_x = (txty_sigmoid[:, :, :, 0] + x_index) * cell_w
            center_y = (txty_sigmoid[:, :, :, 1] + y_index) * cell_h

            # widths and heights in the original image
            width = twth_exp[:, :, :, 0] * cell_w
            height = twth_exp[:, :, :, 1] * cell_h

            # boundary boxes: (x1, y1, x2, y2)
            x1 = center_x - (width / 2)
            y1 = center_y - (height / 2)
            x2 = center_x + (width / 2)
            y2 = center_y + (height / 2)

            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))
            box_confidences.append(sigmoid(output[:, :, :, 4:5]))
            box_class_probs.append(sigmoid(output[:, :, :, 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes using confidence threshold and non-max suppression."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i, grid in enumerate(boxes):
            # Compute class-specific score for each anchor prediction
            box_scores_per_class = box_confidences[i] * box_class_probs[i]
            box_scores_per_class = box_scores_per_class.reshape(
                grid.shape[0], grid.shape[1], grid.shape[2], -1
            )

            # Keep the strongest class score for each box
            best_scores = np.max(box_scores_per_class, axis=-1)
            best_classes = np.argmax(box_scores_per_class, axis=-1)

            # Apply class threshold
            mask = best_scores > self.class_t
            boxs = grid[mask]
            boxs_scores = best_scores[mask]
            boxs_classes = best_classes[mask]

            # Sort boxes by descending score
            order = np.argsort(boxs_scores)[::-1]
            boxs = boxs[order]
            boxs_scores = boxs_scores[order]
            boxs_classes = boxs_classes[order]

            # Apply non-max suppression
            keep = []
            delete = set()

            for a in range(len(boxs)):
                if a in delete:
                    continue

                keep.append(a)

                for b in range(a + 1, len(boxs)):
                    if b in delete:
                        continue

                    box_a = boxs[a]
                    box_b = boxs[b]

                    inter_x1 = max(box_a[0], box_b[0])
                    inter_y1 = max(box_a[1], box_b[1])
                    inter_x2 = min(box_a[2], box_b[2])
                    inter_y2 = min(box_a[3], box_b[3])

                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h

                    a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                    b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                    union_area = a_area + b_area - inter_area

                    if union_area <= 0:
                        continue

                    iou = inter_area / union_area

                    if iou > self.nms_t:
                        delete.add(b)

            keep = np.array(keep, dtype=int)
            filtered_boxes.append(boxs[keep])
            box_classes.append(boxs_classes[keep])
            box_scores.append(boxs_scores[keep])

        return (
            np.array(filtered_boxes, dtype=float),
            np.array(box_scores, dtype=float),
            np.array(box_classes, dtype=int)
        )
