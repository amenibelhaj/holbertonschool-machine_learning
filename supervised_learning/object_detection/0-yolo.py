#!/usr/bin/env python3
"""Defines the Yolo class for performing object detection using YOLOv3."""
import tensorflow.keras as K


class Yolo:
    """Uses the YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize a Yolo instance.

        Args:
            model_path (str): path to where a Darknet Keras model is stored
            classes_path (str): path to the list of class names used for
                the Darknet model, listed in order of index
            class_t (float): box score threshold for the initial
                filtering step
            nms_t (float): IOU threshold for non-max suppression
            anchors (numpy.ndarray): shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
