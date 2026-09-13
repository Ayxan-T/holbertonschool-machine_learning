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
