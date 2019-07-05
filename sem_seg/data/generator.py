from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from keras.utils import Sequence
from sklearn.utils import shuffle

from sem_seg.data.data_source import DataSource
from sem_seg.data.transformations import split_label_image, ImageTransformation, from_pil_to_np


class DataGenerator(Sequence):

    def __init__(self,
                 data_sources: List[DataSource],
                 phase: str,
                 transformation: ImageTransformation,
                 target_size: Tuple[int, int],
                 batch_size: int = 32,
                 active_labels: List[int] = None,
                 random_seed: int = 42) -> None:
        """
        Constructor
        :param batch_size:
        """
        super().__init__()

        self.data_sources = data_sources
        self.phase = phase
        self.target_size = target_size[0], target_size[1]
        self.batch_size = batch_size
        self.classes = [1] if not active_labels else active_labels
        self.transformation: ImageTransformation = transformation

        self.file_paths: List[Tuple[str, str, int]] = []
        for index, source in enumerate(self.data_sources):
            if self.phase == 'train':
                file_names = source.get_train_data()
            elif self.phase == 'val':
                file_names = source.get_val_data()
            else:
                raise ValueError

            # ADD DATA SOURCE INDEX
            file_names_with_index: List[Tuple[str, str, int]] = [(fn[0], fn[1], index) for fn in file_names]

            self.file_paths.extend(file_names_with_index)

        self.file_paths = shuffle(self.file_paths, random_state=random_seed)

        # CREATE ONE HOT ENCODING FOR DATA SOURCES
        self.data_source_encoder = np.eye(len(self.data_sources))

    def __len__(self) -> int:
        """
        Effectively the number of batches
        :return: Number of batches
        """
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, index) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # SIMPLE 4-DIMENSIONAL MATRIX, FIRST DIMENSION IS BATCHES
        batch_images: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (3,)))
        # THIS IS A DICTIONARY, ALL MATRICES ARE ZEROED, EXCEPT FOR THE ACTIVE TARGETS
        batch_masks: Dict[str, np.ndarray] = {}
        for data_source in self.data_sources:
            batch_size: Tuple[int, int, int, int] = ((self.batch_size,) + self.target_size + (len(self.classes),))
            batch_masks[data_source.get_name()] = np.zeros(batch_size)
        # THIS IS A DICTIONARY OF 1-D VECTORS
        batch_sample_weights: Dict[str, np.ndarray] = {}
        for data_source in self.data_sources:
            batch_sample_weights[data_source.get_name()] = np.zeros(shape=self.batch_size)

        # LOOP THROUGH ORIGINAL DATA
        original_batch = self.get_batch(index)
        for batch_index, instance in enumerate(original_batch):
            image, mask, ds_name = instance

            # TRANSFORM IMAGE AND MASK
            transformed = self.transformation([image, mask])
            transformed_image: Image.Image = transformed[0]
            transformed_mask: Image.Image = transformed[1]

            # STORE IMAGE
            image_array: np.ndarray = from_pil_to_np(transformed_image) / 255
            batch_images[batch_index] = image_array

            # STORE MASK
            mask_array: np.ndarray = from_pil_to_np(transformed_mask)
            prepared_mask: np.ndarray = split_label_image(mask_array, self.classes)
            batch_masks[ds_name][batch_index] = prepared_mask

            # STORE SAMPLE WEIGHTS
            batch_sample_weights[ds_name][batch_index] = 1.0

        return batch_images, batch_masks, batch_sample_weights

    def get_batch(self, index: int) -> List[Tuple[Image.Image, Image.Image, str]]:
        batch_names: List[Tuple[str, str, int]] = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]

        output_batch: List[Tuple[Image.Image, Image.Image, str]] = []
        for batch_image, batch_mask, batch_ds in batch_names:
            image: Image.Image = Image.open(batch_image)
            mask: Image.Image = Image.open(batch_mask)
            ds_name: str = self.data_sources[batch_ds].get_name()
            output_batch.append((image, mask, ds_name))

        return output_batch

    def summary(self) -> None:
        print(f'Generator: Phase <{self.phase}> - '
              f'Number of data sources: <{len(self.data_sources)}> - '
              f'Number of samples <{len(self.file_paths)}>')
