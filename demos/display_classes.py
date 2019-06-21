import pathlib as pl
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sem_seg.utils.labels import generate_semantic_rgb
from sem_seg.utils.paths import KITTI_LABEL_IMAGES_DIR, KITTI_RAW_IMAGES_DIR

if __name__ == '__main__':
    """
    In this demo we see how to display an image and the classes.
    We do this to understand how to write some utility methods.

    Inputs:
    * The original image
    * The label image

    Output:
    Display of the image with the overlaid classes.

    """

    # LOAD ORIGINAL AND MASK IMAGE
    file_name: str = '000000_10.png'
    original_image_path: pl.Path = KITTI_RAW_IMAGES_DIR / file_name
    assert original_image_path.exists()
    label_image_path: pl.Path = KITTI_LABEL_IMAGES_DIR / file_name
    assert label_image_path.exists()

    original_image: np.ndarray = np.array(Image.open(original_image_path))
    label_image: np.ndarray = np.array(Image.open(label_image_path))

    labels: List[int] = [4, 5, 6, 7, 8, 9]
    semantic_rgb: np.ndarray = generate_semantic_rgb(label_image, labels=labels)

    # PLOT IMAGES
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(original_image)
    ax2.imshow(semantic_rgb)

    plt.show()
