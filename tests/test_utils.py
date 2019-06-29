import pathlib as pl

from keras.optimizers import Adam

from sem_seg.data.data_source import KittiDataSource, CityscapesDataSource
from sem_seg.data.transformations import Crop
from sem_seg.utils.configuration import Configuration


def test_configuration():
    configuration: Configuration = Configuration(pl.Path('../templates/template.yml'))

    assert configuration.batch_size == 4
    assert configuration.num_epochs == 10
    assert configuration.input_size == (256, 256)
    assert configuration.random_seed == 42
    assert configuration.patience == 15

    assert configuration.limit is None

    assert configuration.save_dir == 'test'
    assert configuration.save_over

    optimizer = configuration.optimizer
    assert isinstance(optimizer, Adam)

    assert len(configuration.datasets) == 2
    assert isinstance(configuration.datasets[0], KittiDataSource)
    assert isinstance(configuration.datasets[1], CityscapesDataSource)

    assert isinstance(configuration.transformation, Crop)