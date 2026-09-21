#!/usr/bin/env python3
"""Defines the Yolo class for performing object detection using YOLOv3."""
import numpy as np
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

    @staticmethod
    def sigmoid(x):
        """Apply the sigmoid function element-wise."""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process the raw outputs of the Darknet model.

        Args:
            outputs (list of numpy.ndarray): predictions for a single
                image, each of shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size (numpy.ndarray): [image_height, image_width]

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            cx = np.tile(np.arange(grid_w).reshape(1, grid_w, 1),
                         (grid_h, 1, anchor_boxes))
            cy = np.tile(np.arange(grid_h).reshape(grid_h, 1, 1),
                         (1, grid_w, anchor_boxes))

            bx = (self.sigmoid(output[..., 0]) + cx) / grid_w
            by = (self.sigmoid(output[..., 1]) + cy) / grid_h
            bw = self.anchors[i, :, 0] * np.exp(output[..., 2]) / input_w
            bh = self.anchors[i, :, 1] * np.exp(output[..., 3]) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes whose best class score is below the threshold.

        Args:
            boxes (list of numpy.ndarray): processed boundary boxes, each
                of shape (grid_height, grid_width, anchor_boxes, 4)
            box_confidences (list of numpy.ndarray): box confidences,
                each of shape (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs (list of numpy.ndarray): class probabilities,
                each of shape
                (grid_height, grid_width, anchor_boxes, classes)

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                filtered_boxes: numpy.ndarray of shape (?, 4)
                box_classes: numpy.ndarray of shape (?,)
                box_scores: numpy.ndarray of shape (?)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, probs in zip(boxes, box_confidences,
                                    box_class_probs):
            scores = conf * probs
            classes = np.argmax(scores, axis=-1)
            best_scores = np.max(scores, axis=-1)
            mask = best_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(best_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
