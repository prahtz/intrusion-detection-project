from typing import Callable, Iterable, List, Union

import cv2
import numpy as np
import subtraction_utils
from numpy.typing import NDArray


class BackgroundSubtraction:
    def __init__(
        self,
        background_samples: List[NDArray],
        threshold: float = 20.0,
        alpha=0.1,
        beta=0.01,
        opening_k_shape=(5, 5),
        closing_k_shape=(13, 13),
        handle_light_changes: bool = True,
        zncc_threshold: float = 0.95,
        area_filter_fn: Callable = None,
        c_optimized: bool = True,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.opening_k_shape = opening_k_shape
        self.closing_k_shape = closing_k_shape
        self.handle_light_changes = handle_light_changes
        self.zncc_threshold = zncc_threshold
        self.area_filter_fn = area_filter_fn
        self.c_optimized = c_optimized
        self.bkg = self.compute_blind_background(background_samples)

        self.open_k = cv2.getStructuringElement(cv2.MORPH_RECT, self.opening_k_shape)
        self.closing_k = cv2.getStructuringElement(cv2.MORPH_RECT, self.closing_k_shape)

    def compute_blind_background(
        self,
        background_samples: List[NDArray],
    ):
        bkg = np.stack(background_samples, axis=0)
        return np.float32(np.median(bkg, axis=0))

    def binarize_frame(self, frame: NDArray):
        if self.c_optimized:
            return subtraction_utils.binarize_frame(self.bkg, frame, self.threshold)
        binary = np.abs(self.bkg - frame)
        binary[binary <= self.threshold] = 0
        binary[binary > self.threshold] = 255
        return binary

    def apply_morphological_operations(self, binary: NDArray):
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.open_k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.closing_k)
        return binary

    def get_bounding_boxes(self, binary: NDArray):
        contours, _ = cv2.findContours(np.uint8(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(contour) for contour in contours]
        new_bboxes = []
        for i in range(len(bboxes)):
            contained = False
            for j in range(len(bboxes)):
                if i != j:
                    x1, y1, w1, h1 = bboxes[j]
                    x2, y2, w2, h2 = bboxes[i]
                    if x1 < x2 < x2 + w2 < x1 + w1 and y1 < y2 < y2 + h2 < y1 + h1:
                        contained = True
                        break
            if not contained:
                new_bboxes.append(bboxes[i])
        return new_bboxes

    def remove_light_changes(self, binary: NDArray, bboxes: List[List[Union[int, float]]], frame: NDArray):
        new_binary = binary.copy()
        new_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            bkg_patch = self.bkg[y : y + h, x : x + w].copy()
            frame_patch = frame[y : y + h, x : x + w].copy()
            bkg_patch[binary[y : y + h, x : x + w] == 0] = 0
            frame_patch[binary[y : y + h, x : x + w] == 0] = 0
            zncc = cv2.matchTemplate(bkg_patch, np.float32(frame_patch), cv2.TM_CCOEFF_NORMED)
            if zncc > self.zncc_threshold:
                new_binary[y : y + h, x : x + w] = 0
            else:
                new_bboxes.append(bbox)
        return new_binary, new_bboxes

    def update_background(self, binary: NDArray, frame: NDArray):
        if self.c_optimized:
            self.bkg = subtraction_utils.update_background(binary, frame, self.bkg, self.alpha, self.beta)
        else:
            f_mask = binary == 255
            b_mask = binary == 0
            x = np.zeros(frame.shape)
            y = np.zeros(self.bkg.shape)
            x[b_mask] = frame[b_mask]
            y[b_mask] = self.bkg[b_mask]
            new_bkg = x * self.alpha + (1 - self.alpha) * y
            new_bkg[f_mask] = self.bkg[f_mask]
            self.bkg = np.float32(new_bkg)

            x = np.zeros(self.bkg.shape)
            y = np.zeros(self.bkg.shape)
            x[f_mask] = frame[f_mask]
            y[f_mask] = self.bkg[f_mask]
            new_bkg = x * self.beta + (1 - self.beta) * y
            self.bkg[f_mask] = new_bkg[f_mask]

    def step(self, frame: NDArray):
        binary = self.binarize_frame(frame)
        binary = self.apply_morphological_operations(binary)

        bboxes = self.get_bounding_boxes(binary)
        if self.handle_light_changes:
            binary, bboxes = self.remove_light_changes(binary, bboxes, frame)
        if self.area_filter_fn is not None:
            bboxes = [bbox for bbox in bboxes if self.area_filter_fn(bbox[2] * bbox[3])]

        self.update_background(binary, frame)
        return bboxes
