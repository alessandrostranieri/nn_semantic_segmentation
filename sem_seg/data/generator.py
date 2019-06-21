import pathlib as pl
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.utils import Sequence
from keras_preprocessing.image import img_to_array

from sem_seg.utils.labels import split_label_image, generate_semantic_rgb, merge_label_images
from sem_seg.utils.paths import KITTI_BASE_DIR, SETS_DIR, IMAGE_DIR, LABEL_DIR


class DataGenerator(Sequence):

    def __init__(self,
                 image_dir: pl.Path,
                 image_name_file_name: str,
                 target_size: Tuple[int, int],
                 batch_size: int = 32,
                 active_labels: List[int] = None) -> None:
        """
        Constructor
        :param image_dir: Main data-set directory
        :param batch_size:
        """
        super().__init__()

        self.image_dir = image_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.classes = [1] if not active_labels else active_labels

        # GET LIST OF FILES
        self.list_names: List[str] = []
        image_names_file: pl.Path = self.image_dir / SETS_DIR / image_name_file_name
        with image_names_file.open() as f:
            self.list_names = f.read().splitlines()

    def __len__(self) -> int:
        """
        Effectively the number of batches
        :return: Number of batches
        """
        return int(np.ceil(len(self.list_names) / float(self.batch_size)))

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        batch_names: List[str] = self.list_names[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (3,)))
        batch_masks: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (len(self.classes),)))
        for index, batch_name in enumerate(batch_names):
            image: Image.Image = Image.open(self.image_dir / IMAGE_DIR / batch_name)
            resized_image = image.resize(size=self.target_size)
            resized_image_array: np.ndarray = img_to_array(resized_image)
            batch_images[index] = resized_image_array

            mask: Image.Image = Image.open(self.image_dir / LABEL_DIR / batch_name)
            resized_mask = mask.resize(self.target_size)
            resized_mask_array: np.ndarray = img_to_array(resized_mask)
            prepared_mask: np.ndarray = split_label_image(resized_mask_array, self.classes)
            batch_masks[index] = prepared_mask

        return batch_images, batch_masks


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

    # GET ONE IMAGE-LABELS PAIR
    val_images, val_labels = validation_generator[0]
    val_image: np.ndarray = val_images[0]
    val_image = val_image.astype(int)
    val_labels: np.ndarray = val_labels[0]
    # CONVERT LABELS TO RGB
    val_labels = merge_label_images(val_labels, labels)
    val_labels_rgb: np.ndarray = generate_semantic_rgb(val_labels, labels=labels)

    # VISUALIZED DATA
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(val_image)
    ax2.imshow(val_labels_rgb)

    plt.show()
