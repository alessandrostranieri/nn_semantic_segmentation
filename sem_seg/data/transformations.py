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


class Crop(ImageTransformation):

    def __init__(self, target_size: Tuple[int, int]) -> None:
        super().__init__()

        self.new_width = target_size[0]
        self.new_height = target_size[1]

    def __call__(self, input_image: Image.Image) -> Image.Image:

        old_width, old_height = input_image.size
        left = (old_width - self.new_width) / 2
        top = (old_height - self.new_height) / 2
        right = (old_width + self.new_width) / 2
        bottom = (old_height + self.new_height) / 2

        return input_image.crop((left, top, right, bottom))


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
