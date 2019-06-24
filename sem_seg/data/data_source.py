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


class CityscapesDataSource(DataSource):
    def __init__(self, base_dir: pl.Path) -> None:
        super().__init__()

        # BASE DIRECTORIES
        camera_images_base_dir: pl.Path = base_dir / 'leftImg8bit_trainvaltest' / 'leftImg8bit'
        label_images_base_dir: pl.Path = base_dir / 'gtFine_trainvaltest' / 'gtFine'

        # TRAIN IMAGES
        camera_images_train_dir: pl.Path = camera_images_base_dir / 'train'
        camera_images_train_names: List[str] = [str(name) for name in
                                                camera_images_train_dir.glob("**/*leftImg8bit.png")]

        # TRAIN LABEL IMAGES
        label_images_train_dir: pl.Path = label_images_base_dir / 'train'
        label_images_train_names: List[str] = [str(name) for name in label_images_train_dir.glob("**/*_labelIds.png")]

        # TRAIN DATA
        self._train_data: List[Tuple[str, str]] = [(i, l) for i, l in
                                                   zip(camera_images_train_names, label_images_train_names)]

        # VAL IMAGES
        camera_images_val_dir: pl.Path = camera_images_base_dir / 'val'
        camera_images_val_names: List[str] = [str(name) for name in
                                              camera_images_val_dir.glob("**/*leftImg8bit.png")]

        # VAL LABEL IMAGES
        label_images_val_dir: pl.Path = label_images_base_dir / 'val'
        label_images_val_names: List[str] = [str(name) for name in label_images_val_dir.glob("**/*_labelIds.png")]

        # VAL DATA
        self._val_data: List[Tuple[str, str]] = [(i, l) for i, l in
                                                 zip(camera_images_val_names, label_images_val_names)]

    def get_train_data(self) -> List[Tuple[str, str]]:
        return self._train_data

    def get_val_data(self) -> List[Tuple[str, str]]:
        return self._val_data




