import pathlib as pl

import matplotlib.pyplot as plt
from PIL import Image

from sem_seg.utils.labels import generate_semantic_rgb
from sem_seg.utils.paths import KITTI_LABEL_IMAGE, KITTI_RAW_IMAGE

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
    original_image_path: pl.Path = KITTI_RAW_IMAGE / file_name
    assert original_image_path.exists()
    label_image_path: pl.Path = KITTI_LABEL_IMAGE / file_name
    assert label_image_path.exists()

    original_image: Image = Image.open(original_image_path)
    label_image: Image = Image.open(label_image_path)
    semantic_rgb: Image = generate_semantic_rgb(label_image)

    # PLOT IMAGES
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(original_image)
    ax2.imshow(semantic_rgb)

    plt.show()
