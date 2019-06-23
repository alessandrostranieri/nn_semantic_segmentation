from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sem_seg.data.generator import DataGenerator
from sem_seg.utils.labels import generate_semantic_rgb, merge_label_images, resize_and_crop
from sem_seg.utils.paths import KITTI_BASE_DIR
from sem_seg.utils.transformations import resize

if __name__ == '__main__':

    labels = [0, 6, 7, 8, 9]

    # CREATE GENERATOR
    training_generator: DataGenerator = DataGenerator(KITTI_BASE_DIR,
                                                      'train.txt',
                                                      batch_size=4,
                                                      target_size=(256, 256),
                                                      active_labels=labels)

    # CHECK THAT GENERATOR LEN WORKS
    generator_len: int = len(training_generator)
    assert type(generator_len) == int
    print(f'Number of training batches: {generator_len}')

    # GET FIRST BATCH AND CHECK THAT DIMENSIONS MATCH
    i, m = training_generator[0]
    expected_shape_x: Tuple[int, int, int, int] = (4, 256, 256, 3)
    assert i.shape == expected_shape_x, f"Image batch in the wrong shape: {i.shape} instead of {expected_shape_x}"
    expected_shape_y: Tuple[int, int, int, int] = (4, 256, 256, 5)
    assert m.shape == expected_shape_y, f"Mask batch in the wrong shape: {m.shape} instead of {expected_shape_y}"

    # CHECK THE LOOPING THROUGH WORKS
    for batch in training_generator:
        i, m = batch
        # DO SOMETHING SIMPLE
        assert i.shape == expected_shape_x, f"Image batch in the wrong shape: {i.shape} instead of {expected_shape_x}"
        assert m.shape == expected_shape_y, f"Mask batch in the wrong shape: {m.shape} instead of {expected_shape_y}"

    # DO THE SAME FOR VALIADATION GENERATOR
    validation_generator: DataGenerator = DataGenerator(KITTI_BASE_DIR,
                                                        'val.txt',
                                                        batch_size=4,
                                                        target_size=(256, 256),
                                                        active_labels=labels)
    print(f'Number of validation batches: {len(validation_generator)}')
    for batch in validation_generator:
        i, m = batch
        # DO SOMETHING SIMPLE
        assert i.shape == expected_shape_x, f"Image batch in the wrong shape: {i.shape} instead of {expected_shape_x}"
        assert m.shape == expected_shape_y, f"Mask batch in the wrong shape: {m.shape} instead of {expected_shape_y}"

    # GENERATOR ORIGINAL IMAGES
    val_original_image, val_original_labels = validation_generator.get_batch(0)[0]
    val_original_image_array: np.ndarray = np.array(val_original_image)
    val_original_labels: np.ndarray = np.array(val_original_labels)
    val_original_label_rgb: np.ndarray = generate_semantic_rgb(val_original_labels)

    # GENERATOR PRE-PROCESSED IMAGES
    val_image_batch, val_labels_batch = validation_generator[0]

    # GET SINGLE IMAGES FROM BATCH
    val_image = val_image_batch[0]
    val_image = val_image.astype(np.uint8)
    val_labels = val_labels_batch[0]
    val_labels = val_labels.astype(np.uint8)

    # COLORIZE LABELS
    val_labels = merge_label_images(val_labels, labels)
    val_labels_rgb: np.ndarray = generate_semantic_rgb(val_labels)

    # IMAGES RECOMPOSED FROM INPUT
    original_size: Tuple[int, int] = val_original_image.size
    val_recomposed_image = resize(Image.fromarray(val_image), original_size)
    val_recomposed_image_array = np.array(val_recomposed_image)

    val_recomposed_labels = resize(Image.fromarray(val_labels), original_size)
    val_recomposed_labels_array = np.array(val_recomposed_labels)

    val_recomposed_labels_rgb = generate_semantic_rgb(val_recomposed_labels_array)

    # VISUALIZED DATA
    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3)

    ax11.set_title('Original Camera')
    ax11.imshow(val_original_image_array)

    ax21.set_title('Original Semantic')
    ax21.imshow(val_original_label_rgb)

    ax12.set_title('Input Camera')
    ax12.imshow(val_image)
    ax22.set_title('Input Semantic')
    ax22.imshow(val_labels_rgb)
    ax13.set_title('Recomposed Camera')
    ax13.imshow(val_recomposed_image_array)
    ax23.set_title('Recomposed Semantic')
    ax23.imshow(val_recomposed_labels_rgb)

    plt.show()