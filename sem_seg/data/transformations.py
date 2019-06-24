from abc import ABCMeta, abstractmethod
from typing import Tuple, List

import numpy as np
from PIL import Image


class ImageTransformation(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, input_image: Image.Image) -> Image.Image:
        raise NotImplementedError


class Resize(ImageTransformation):

    def __init__(self, target_size: Tuple[int, int]) -> None:
        super().__init__()

        self.target_size = target_size

    def __call__(self, input_image: Image.Image) -> Image.Image:
        return input_image.resize(size=self.target_size)


def split_label_image(label_image: np.ndarray, classes: List[int]) -> np.ndarray:
    assert label_image.ndim == 2, 'Input image must be single channel'

    # SET VALUES
    mask_list: List[np.ndarray] = []
    for label in classes:
        mask = np.equal(label_image, label)
        mask_list.append(mask)

    # PREPARE OUTPUT
    split_label_images: np.ndarray = np.stack(mask_list, axis=-1)
    split_label_images = split_label_images.astype(float)

    return split_label_images


def merge_label_images(label_image: np.ndarray, labels: List[int]) -> np.ndarray:
    result: np.ndarray = np.zeros(shape=(label_image.shape[0], label_image.shape[1]), dtype=np.uint8)
    for index, label in enumerate(labels):
        mask = label_image[:, :, index] == 1.0
        result[mask] = label

    return result
