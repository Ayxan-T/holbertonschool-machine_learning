#!/usr/bin/env python3
""" Module: 0-yolo """

from tensorflow.keras.models import load_model


class Yolo:
    """ Yolo Darknet Keras model """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
