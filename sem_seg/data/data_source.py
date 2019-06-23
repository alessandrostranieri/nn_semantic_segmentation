import pathlib as pl
from typing import List, Tuple

from sklearn.model_selection import train_test_split


class DataSource:

    def get_train_data(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def get_val_data(self) -> List[Tuple[str, str]]:
        raise NotImplementedError


class KittiDataSource(DataSource):

    def __init__(self, base_dir: pl.Path, test_ratio: float = 0.2, random_seed: int = 42) -> None:
        super().__init__()

        # READ ALL FILES
        original_image_dir: pl.Path = base_dir / 'data_semantics' / 'training' / 'image_2' / 'all'
        image_names: List[str] = [image_name.name for image_name in original_image_dir.glob('*.png')]

        # GET ALL LABEL IMAGES AND CHECK INTEGRITY

        original_label_dir: pl.Path = base_dir / 'data_semantics' / 'training' / 'semantic' / 'all'
        label_names: List[str] = [label_name.name for label_name in original_label_dir.glob('*.png')]
        assert image_names == label_names, "Image and Label file names are different."

        # PERFORM TRAIN/VAL SPLIT - CHECK AGAIN
        x_train, x_val, y_train, y_val = train_test_split(image_names, label_names,
                                                          test_size=test_ratio, random_state=random_seed)
        assert x_train == y_train, "Training Image and Label sets are different."
        assert x_val == y_val, "Validation Image and Label sets are different."

        self._train_data: List[Tuple[str, str]] = []
        for image_name in x_train:
            self._train_data.append((str(original_image_dir / image_name), str(original_label_dir / image_name)))

        self._val_data: List[Tuple[str, str]] = []
        for image_name in x_val:
            self._val_data.append((str(original_image_dir / image_name), str(original_label_dir / image_name)))

    def get_train_data(self) -> List[Tuple[str, str]]:
        return self._train_data

    def get_val_data(self) -> List[Tuple[str, str]]:
        return self._val_data



