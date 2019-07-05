from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Union

import numpy as np
from PIL import Image
from numpy.random import RandomState


class ImageTransformation(metaclass=ABCMeta):
    @abstractmethod
    def _apply(self, input_images: List[Image.Image]) -> List[Image.Image]:
        raise NotImplementedError

    def __call__(self, input_images: Union[Image.Image, List[Image.Image]]) -> List[Image.Image]:
        if not isinstance(input_images, List):
            input_images = [input_images]
        return self._apply(input_images)


class Resize(ImageTransformation):

    def __init__(self, target_size: Tuple[int, int]) -> None:
        super().__init__()

        self.target_size = target_size

    def _apply(self, input_images: List[Image.Image]) -> List[Image.Image]:
        return [input_image.resize(size=self.target_size) for input_image in input_images]


class Crop(ImageTransformation):

    def __init__(self, target_size: Tuple[int, int]) -> None:
        super().__init__()

        self.new_width = target_size[0]
        self.new_height = target_size[1]

    def _apply(self, input_images: List[Image.Image]) -> List[Image.Image]:
        old_width, old_height = input_images[0].size
        left = (old_width - self.new_width) / 2
        top = (old_height - self.new_height) / 2
        right = (old_width + self.new_width) / 2
        bottom = (old_height + self.new_height) / 2

        return [input_image.crop((left, top, right, bottom)) for input_image in input_images]


class RandomCrop(ImageTransformation):

    def __init__(self, target_size: Tuple[int, int], random_seed: int = 42):
        super().__init__()

        self.random_state: RandomState = RandomState(random_seed)
        self.target_width = target_size[0]
        self.target_height = target_size[1]

    def _apply(self, input_images: List[Image.Image]) -> List[Image.Image]:
        old_width, old_height = input_images[0].size
        width_range = (0, old_width - self.target_width)
        height_range = (0, old_height - self.target_height)
        left: int = self.random_state.randint(*width_range)
        right: int = left + self.target_width
        top: int = self.random_state.randint(*height_range)
        bottom: int = top + self.target_height

        return [input_image.crop((left, top, right, bottom)) for input_image in input_images]


class Fit(ImageTransformation):

    def __init__(self, target_size: Tuple[int, int]):
        super().__init__()

        self.target_size = target_size

    def _apply(self, input_images: List[Image.Image]) -> List[Image.Image]:
        result = [pad_and_resize(input_image, self.target_size) for input_image in input_images]
        return result


def pad_and_resize(input_image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    longest_size: int = max(*input_image.size)
    shortest_size: int = min(*input_image.size)
    square_size: Tuple[int, int] = (longest_size, longest_size)
    canvas: Image.Image = Image.new(mode=input_image.mode, size=square_size)
    top_left: Tuple[int, int] = (0, (longest_size - shortest_size) // 2)
    canvas.paste(input_image, box=top_left)
    resized = canvas.resize(target_size, resample=Image.NEAREST)
    return resized


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


# noinspection PyTypeChecker
def from_pil_to_np(input_image: Image.Image) -> np.ndarray:
    return np.asarray(input_image)
