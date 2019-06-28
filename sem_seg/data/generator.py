from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from keras.utils import Sequence
from sklearn.utils import shuffle

from sem_seg.data.data_source import DataSource
from sem_seg.data.transformations import split_label_image, ImageTransformation


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
        self.target_size = target_size[0], target_size[1]
        self.batch_size = batch_size
        self.classes = [1] if not active_labels else active_labels
        self.transformation: ImageTransformation = transformation

        self.file_paths: List[Tuple[str, str, int]] = []
        for index, source in enumerate(self.data_sources):
            if phase == 'train':
                file_names = source.get_train_data()
            elif phase == 'val':
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
        batch_images: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (3,)))
        batch_masks: np.ndarray = np.zeros(((self.batch_size,) + self.target_size + (len(self.classes),)))
        batch_ds: np.ndarray = np.zeros((self.batch_size, len(self.data_sources)))
        batch_ds = np.squeeze(batch_ds)
        pure_batch = self.get_batch(index)

        for idx, instance in enumerate(pure_batch):
            image, mask, ds_idx = instance

            transformed = self.transformation([image, mask])
            transformed_image: Image.Image = transformed[0]
            transformed_mask: Image.Image = transformed[1]

            image_array: np.ndarray = np.array(transformed_image) / 255
            batch_images[idx] = image_array

            mask_array: np.ndarray = np.array(transformed_mask)
            prepared_mask: np.ndarray = split_label_image(mask_array, self.classes)
            batch_masks[idx] = prepared_mask

            one_hot_ds: np.ndarray = self.data_source_encoder[ds_idx]
            one_hot_ds = one_hot_ds.squeeze()
            batch_ds[idx] = one_hot_ds

        targets: Dict[str, np.ndarray] = dict()
        sample_weights: Dict[str, np.ndarray] = dict()
        for idx, ds in enumerate(self.data_sources):
            targets[ds.get_name()] = batch_masks
            if batch_ds.ndim == 1:
                sample_weights[ds.get_name()] = batch_ds
            else:
                sample_weights[ds.get_name()] = batch_ds[:, idx]

        return batch_images, targets, sample_weights

    def get_batch(self, index: int) -> List[Tuple[Image.Image, Image.Image, int]]:
        batch_names: List[Tuple[str, str, int]] = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]

        output_batch: List[Tuple[Image.Image, Image.Image, int]] = []
        for batch_image, batch_mask, batch_ds in batch_names:
            image: Image.Image = Image.open(batch_image)
            mask: Image.Image = Image.open(batch_mask)
            output_batch.append((image, mask, batch_ds))

        return output_batch
