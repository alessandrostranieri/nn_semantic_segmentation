import pathlib as pl
from typing import Optional, List, Any, Dict, Tuple

import yaml
from keras.optimizers import Optimizer, Adam, SGD

from sem_seg.data.data_source import DataSource, KittiDataSource, CityscapesDataSource
from sem_seg.data.transformations import ImageTransformation, Crop, RandomCrop, Resize, Fit
from sem_seg.utils.paths import MODELS_DIR, KITTI_BASE_DIR, CITYSCAPES_BASE_DIR


class Configuration:
    DEFAULT_BATCH_SIZE: int = 1
    DEFAULT_NUM_EPOCHS: int = 1
    DEFAULT_INPUT_SIZE: int = 64
    DEFAULT_RANDOM_SEED: int = 0
    DEFAULT_PATIENCE: int = 10
    DEFAULT_SAVE_DIR: pl.Path = MODELS_DIR / 'demo'
    DEFAULT_TRANSFORMATION: str = 'random_crop'

    KNOWN_OPTIMIZERS: List[str] = ['adam', 'sgd']

    def __init__(self, config_file: pl.Path) -> None:
        super().__init__()

        with config_file.open(mode='r') as f:
            config_data = yaml.safe_load(f)

            self.batch_size: int = config_data.get('batch_size', Configuration.DEFAULT_BATCH_SIZE)
            self.num_epochs: int = config_data.get('num_epochs', Configuration.DEFAULT_NUM_EPOCHS)
            self.input_size: Tuple[int, int] = (config_data.get('input_size', Configuration.DEFAULT_INPUT_SIZE),) * 2
            self.random_seed: int = config_data.get('random_seed', Configuration.DEFAULT_RANDOM_SEED)
            self.patience: int = config_data.get('patience', Configuration.DEFAULT_PATIENCE)
            self.limit: Optional[int] = config_data.get('limit', None)
            self.save_dir: pl.Path = config_data.get('save_dir', Configuration.DEFAULT_SAVE_DIR)
            self.save_over: bool = config_data.get('save_over', False)

            self.transformation: ImageTransformation = self._read_transformation(config_data)

            self.optimizer: Optimizer = self._read_optimizer(config_data)

            self.datasets: List[DataSource] = self._read_datasets(config_data)

    def _read_optimizer(self, config_data) -> Optimizer:
        optimizer_config: Dict[str, Any] = config_data['optimizer']
        optimizer_name = list(optimizer_config.keys())[0]
        assert optimizer_name in Configuration.KNOWN_OPTIMIZERS, f'Unknown optimizer: {optimizer_name}'

        if optimizer_name == 'adam':
            return Adam.from_config(optimizer_config[optimizer_name])
        elif optimizer_name == 'sgd':
            return SGD.from_config(optimizer_config[optimizer_name])

    def _read_datasets(self, config_data) -> List[DataSource]:
        dataset_names: List[str] = config_data['dataset']

        datasets: List[DataSource] = []
        if 'kitti' in dataset_names:
            datasets.append(KittiDataSource(KITTI_BASE_DIR, limit=self.limit))
        if 'cityscapes' in dataset_names:
            datasets.append(CityscapesDataSource(CITYSCAPES_BASE_DIR, limit=self.limit))

        return datasets

    def _read_transformation(self, config_data: Dict[str, Any]) -> ImageTransformation:
        transformation_name = config_data.get('transformation', Configuration.DEFAULT_TRANSFORMATION)
        assert transformation_name in ['crop', 'random_crop', 'resize', 'fit']
        if transformation_name == 'crop':
            return Crop(self.input_size)
        elif transformation_name == 'random_crop':
            return RandomCrop(self.input_size, self.random_seed)
        elif transformation_name == 'resize':
            return Resize(self.input_size)
        elif transformation_name == 'fit':
            return Fit(self.input_size)