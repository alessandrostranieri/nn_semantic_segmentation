import pathlib as pl
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split


class DataSource:

    def __init__(self, limit: Optional[int] = None) -> None:
        super().__init__()
        self.limit = limit
        self._train_data: List[Tuple[str, str]] = []
        self._val_data: List[Tuple[str, str]] = []
        self._read_data()
        self._slice_data()

    def _read_data(self) -> None:
        raise NotImplementedError

    def _slice_data(self) -> None:
        if self.limit:
            self.limit = min(self.limit, len(self._train_data))
            self._train_data = self._train_data[:self.limit]
            self._val_data = self._train_data[:self.limit]

    def get_train_data(self) -> List[Tuple[str, str]]:
        return self._train_data

    def get_val_data(self) -> List[Tuple[str, str]]:
        return self._val_data


class KittiDataSource(DataSource):

    def __init__(self, base_dir: pl.Path, test_ratio: float = 0.2, random_seed: int = 42,
                 limit: Optional[int] = None) -> None:
        self.base_dir = base_dir
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        super().__init__(limit=limit)

    def _read_data(self) -> None:
        # READ ALL FILES
        original_image_dir: pl.Path = self.base_dir / 'data_semantics' / 'training' / 'image_2' / 'all'
        image_names: List[str] = [image_name.name for image_name in original_image_dir.glob('*.png')]

        # GET ALL LABEL IMAGES AND CHECK INTEGRITY

        original_label_dir: pl.Path = self.base_dir / 'data_semantics' / 'training' / 'semantic' / 'all'
        label_names: List[str] = [label_name.name for label_name in original_label_dir.glob('*.png')]
        assert image_names == label_names, "Image and Label file names are different."

        # PERFORM TRAIN/VAL SPLIT - CHECK AGAIN
        x_train, x_val, y_train, y_val = train_test_split(image_names, label_names,
                                                          test_size=self.test_ratio, random_state=self.random_seed)
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
    def __init__(self, base_dir: pl.Path, limit: Optional[int] = None) -> None:
        self.base_dir = base_dir

        super().__init__(limit=limit)

    def _read_data(self) -> None:
        # BASE DIRECTORIES
        camera_images_base_dir: pl.Path = self.base_dir / 'leftImg8bit_trainvaltest' / 'leftImg8bit'
        label_images_base_dir: pl.Path = self.base_dir / 'gtFine_trainvaltest' / 'gtFine'

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
