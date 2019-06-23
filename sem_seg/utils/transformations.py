from typing import Tuple

from PIL import Image


def resize(input_image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    return input_image.resize(target_size)

