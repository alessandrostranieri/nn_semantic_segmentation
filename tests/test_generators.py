from re import search
from typing import List

from sem_seg.data.data_source import KittiDataSource, CityscapesDataSource, DataSource
from sem_seg.data.generator import DataGenerator
from sem_seg.data.transformations import Resize, split_label_image, merge_label_images
from sem_seg.utils.labels import CityscapesLabels
from sem_seg.utils.paths import KITTI_BASE_DIR, CITYSCAPES_BASE_DIR
import pathlib as pl
import numpy as np


def test_kitti_data_source():
    kitti_data_source: KittiDataSource = KittiDataSource(KITTI_BASE_DIR)

    train_data = kitti_data_source.get_train_data()
    assert len(train_data) == 160
    image_name: str = pl.Path(train_data[0][0]).name
    label_name: str = pl.Path(train_data[0][1]).name
    assert image_name == label_name
    assert image_name == '000079_10.png'

    val_data = kitti_data_source.get_val_data()
    assert len(val_data) == 40
    image_name: str = pl.Path(val_data[0][0]).name
    label_name: str = pl.Path(val_data[0][1]).name
    assert image_name == label_name
    assert image_name == '000095_10.png'


def test_kitti_data_source_with_limig():
    kitti_data_source: KittiDataSource = KittiDataSource(KITTI_BASE_DIR, limit=4)

    train_data = kitti_data_source.get_train_data()
    assert len(train_data) == 4
    val_data = kitti_data_source.get_val_data()
    assert len(val_data) == 4


def test_cityscapes_data_source():
    cityscapes_data_source: CityscapesDataSource = CityscapesDataSource(CITYSCAPES_BASE_DIR)

    train_data = cityscapes_data_source.get_train_data()
    assert len(train_data) == 2975
    assert pl.Path(train_data[0][0]).name == 'aachen_000000_000019_leftImg8bit.png'
    for train_instance in train_data:
        image_name = search(r"\w+_\d+_\d+", pl.Path(train_instance[0]).name)[0]
        segmentation_name = search(r"\w+_\d+_\d+", pl.Path(train_instance[1]).name)[0]
        assert image_name == segmentation_name

    val_data = cityscapes_data_source.get_val_data()
    assert len(val_data) == 500
    assert pl.Path(val_data[0][0]).name == 'frankfurt_000000_000294_leftImg8bit.png'
    for val_instance in val_data:
        image_name = search(r"\w+_\d+_\d+", pl.Path(val_instance[0]).name)[0]
        segmentation_name = search(r"\w+_\d+_\d+", pl.Path(val_instance[1]).name)[0]
        assert image_name == segmentation_name


def test_cityscapes_data_source_with_limit():
    cityscapes_data_source: CityscapesDataSource = CityscapesDataSource(CITYSCAPES_BASE_DIR, limit=4)

    train_data = cityscapes_data_source.get_train_data()
    assert len(train_data) == 4
    val_data = cityscapes_data_source.get_val_data()
    assert len(val_data) == 4


def test_generator_combine_data_sources():
    kitti_data_source: KittiDataSource = KittiDataSource(KITTI_BASE_DIR)
    cityscapes_data_source: CityscapesDataSource = CityscapesDataSource(CITYSCAPES_BASE_DIR)
    data_sources: List[DataSource] = [kitti_data_source, cityscapes_data_source]

    train_data_generator: DataGenerator = DataGenerator(data_sources=data_sources,
                                                        phase='train',
                                                        batch_size=4,
                                                        target_size=(256, 256),
                                                        active_labels=[0, 1],
                                                        random_seed=42)

    assert len(train_data_generator) == 784

    val_data_generator: DataGenerator = DataGenerator(data_sources=data_sources,
                                                      phase='val',
                                                      batch_size=4,
                                                      target_size=(256, 256),
                                                      active_labels=[0, 1],
                                                      random_seed=42)

    assert len(val_data_generator) == 135


def test_label_integrity():
    """
    Check that the resized version of the original labels can be reconstructed from the matrix which will be the
    neural network's input
    """

    kitti_data_source: KittiDataSource = KittiDataSource(KITTI_BASE_DIR)
    train_data_generator: DataGenerator = DataGenerator(data_sources=[kitti_data_source],
                                                        phase='train',
                                                        batch_size=4,
                                                        target_size=(256, 256),
                                                        active_labels=[0, 1],
                                                        random_seed=42)

    original_image, original_labels = train_data_generator.get_batch(0)[0]
    resized_original = Resize((256, 256))(original_labels)
    resized_original_array = np.array(resized_original)

    resized_split = split_label_image(resized_original_array, CityscapesLabels.ALL)
    resized_merged = merge_label_images(resized_split, CityscapesLabels.ALL)

    assert (resized_original_array == resized_merged).all()