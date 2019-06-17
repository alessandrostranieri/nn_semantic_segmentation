import pathlib as pl
from typing import List, Tuple

import numpy as np
from PIL import Image
from keras.utils import Sequence
from keras_preprocessing.image import load_img, img_to_array

from sem_seg.utils.paths import KITTI_BASE_DIR


class DataGenerator(Sequence):

    def __init__(self,
                 image_dir: pl.Path,
                 image_name_file_name: str,
                 target_size: Tuple[int, int],
                 batch_size: int = 32) -> None:
        """
        Constructor
        :param image_dir: Main data-set directory
        :param batch_size:
        """
        super().__init__()

        self.image_dir = image_dir
        self.target_size = target_size
        self.batch_size = batch_size

        # GET LIST OF FILES
        self.list_names: List[str] = []
        image_names_file: pl.Path = self.image_dir / 'ImageSets' / image_name_file_name
        with image_names_file.open() as f:
            self.list_names = f.read().splitlines()

    def __len__(self) -> int:
        """
        Effectively the number of batches
        :return: Number of batches
        """
        return np.ceil(len(self.list_names) / float(self.batch_size))

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        batch_names: List[str] = self.list_names[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (3,)))
        batch_masks: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (3, )))
        for index, batch_name in enumerate(batch_names):
            image: Image.Image = Image.open(self.image_dir / 'JPEGImages' / batch_name)
            resized_image = image.resize(size=self.target_size)
            batch_images[index] = img_to_array(resized_image)

            mask: Image.Image = Image.open(self.image_dir / 'SegmentationClass' / batch_name)
            resized_mask = mask.resize(self.target_size)
            batch_masks[index] = img_to_array(resized_mask)

        return batch_images, batch_masks


if __name__ == '__main__':

    # CREATE GENERATOR
    # TODO Add configuration dictionary
    generator: DataGenerator = DataGenerator(KITTI_BASE_DIR, 'train.txt', batch_size=4, target_size=(256, 256))

    # GET FIRST BATCH AND CHECK THAT DIMENSIONS MATCH
    i, m = generator[0]
    expected_shape: Tuple[int, int, int, int] = (4, 256, 256, 3)
    assert i.shape == expected_shape, f"Image batch in the wrong shape: {i.shape}"
    assert m.shape == expected_shape, f"Mask batch in the wrong shape: {m.shape}"
