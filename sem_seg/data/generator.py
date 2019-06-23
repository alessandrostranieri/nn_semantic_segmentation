import pathlib as pl
from typing import List, Tuple

import numpy as np
from PIL import Image
from keras.utils import Sequence

from sem_seg.data.data_source import DataSource
from sem_seg.utils.labels import split_label_image, pad_and_resize
from sem_seg.utils.paths import SETS_DIR, IMAGE_DIR, LABEL_DIR
from sem_seg.utils.transformations import resize


class DataGenerator(Sequence):

    def __init__(self,
                 data_sources: List[DataSource],
                 phase: str,
                 target_size: Tuple[int, int],
                 batch_size: int = 32,
                 active_labels: List[int] = None) -> None:
        """
        Constructor
        :param image_dir: Main data-set directory
        :param batch_size:
        """
        super().__init__()

        self.data_sources = data_sources
        self.target_size = target_size
        self.batch_size = batch_size
        self.classes = [1] if not active_labels else active_labels

        self.file_paths: List[Tuple[str, str]] = []
        for index, source in enumerate(self.data_sources):
            if phase == 'train':
                file_names = source.get_train_data()
            elif phase == 'val':
                file_names = source.get_val_data()
            else:
                raise ValueError

            self.file_paths.extend(file_names)

    def __len__(self) -> int:
        """
        Effectively the number of batches
        :return: Number of batches
        """
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        batch_images: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (3,)))
        batch_masks: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (len(self.classes),)))
        pure_batch = self.get_batch(index)

        for idx, instance in enumerate(pure_batch):
            image, mask = instance

            transformed_image: Image.Image = resize(image, target_size=self.target_size)
            image_array: np.ndarray = np.array(transformed_image)
            batch_images[idx] = image_array

            transformed_mask: Image.Image = resize(mask, target_size=self.target_size)
            mask_array: np.ndarray = np.array(transformed_mask)
            prepared_mask: np.ndarray = split_label_image(mask_array, self.classes)
            batch_masks[idx] = prepared_mask

        return batch_images, batch_masks

    def get_batch(self, index: int) -> List[Tuple[Image.Image, Image.Image]]:
        batch_names: List[Tuple[str, str]] = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]

        output_batch: List[Tuple[Image.Image, Image.Image]] = []
        for batch_image, batch_mask in batch_names:
            image: Image.Image = Image.open(batch_image)
            mask: Image.Image = Image.open(batch_mask)
            output_batch.append((image, mask))

        return output_batch
