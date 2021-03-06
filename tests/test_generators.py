from re import search
from typing import List

from sem_seg.data.data_source import KittiDataSource, CityscapesDataSource, DataSource
from sem_seg.data.generator import DataGenerator
from sem_seg.data.transformations import Resize, split_label_image, merge_label_images, Crop, from_pil_to_np
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
                                                        transformation=Crop((256, 256)),
                                                        target_size=(256, 256),
                                                        active_labels=[0, 1],
                                                        random_seed=42)

    assert len(train_data_generator) == 784

    val_data_generator: DataGenerator = DataGenerator(data_sources=data_sources,
                                                      phase='val',
                                                      batch_size=4,
                                                      transformation=Crop((256, 256)),
                                                      target_size=(256, 256),
                                                      active_labels=[0, 1],
                                                      random_seed=42)

    assert len(val_data_generator) == 135


def test_generator_combine_data_sources_returns_sample_weight():
    kitti_data_source: KittiDataSource = KittiDataSource(KITTI_BASE_DIR)
    cityscapes_data_source: CityscapesDataSource = CityscapesDataSource(CITYSCAPES_BASE_DIR)
    data_sources: List[DataSource] = [kitti_data_source, cityscapes_data_source]

    data_generator: DataGenerator = DataGenerator(data_sources=data_sources,
                                                  phase='train',
                                                  batch_size=4,
                                                  transformation=Crop((256, 256)),
                                                  target_size=(256, 256),
                                                  active_labels=[0, 1],
                                                  random_seed=42)

    original_batch = data_generator.get_batch(0)
    original_batch_tuple = original_batch[0]
    original_batch_ds_name = original_batch_tuple[2]
    assert original_batch_ds_name == 'cityscapes'

    batch_images, batch_masks, batch_sample_weights = data_generator[0]

    assert isinstance(batch_sample_weights, dict)
    assert np.equal(batch_sample_weights['kitti'], np.asarray([0, 0, 0, 0])).all()
    assert np.equal(batch_sample_weights['cityscapes'], np.asarray([1, 1, 1, 1])).all()


def test_label_integrity():
    """
    Check that the resized version of the original labels can be reconstructed from the matrix which will be the
    neural network's input
    """

    kitti_data_source: KittiDataSource = KittiDataSource(KITTI_BASE_DIR)
    train_data_generator: DataGenerator = DataGenerator(data_sources=[kitti_data_source],
                                                        phase='train',
                                                        batch_size=4,
                                                        transformation=Crop((256, 256)),
                                                        target_size=(256, 256),
                                                        active_labels=[0, 1],
                                                        random_seed=42)

    original_image, original_labels, _ = train_data_generator.get_batch(0)[0]
    resized_original = Resize((256, 256))(original_labels)[0]
    resized_original_array = from_pil_to_np(resized_original)

    resized_split = split_label_image(resized_original_array, CityscapesLabels.ALL)
    resized_merged = merge_label_images(resized_split, CityscapesLabels.ALL)

    assert (resized_original_array == resized_merged).all()


def test_argmax_on_split_images():
    kitti_data_source: KittiDataSource = KittiDataSource(KITTI_BASE_DIR)
    train_data_generator: DataGenerator = DataGenerator(data_sources=[kitti_data_source],
                                                        phase='train',
                                                        batch_size=4,
                                                        transformation=Crop((256, 256)),
                                                        target_size=(256, 256),
                                                        active_labels=CityscapesLabels.ALL,
                                                        random_seed=42)

    original_image, original_labels, _ = train_data_generator.get_batch(0)[0]
    resized_original_labels = Crop((256, 256))(original_labels)[0]
    resized_original_labels_np = from_pil_to_np(resized_original_labels)

    input_image, input_labels, _ = train_data_generator[0]
    input_labels = input_labels['kitti']
    input_labels = input_labels[0]
    input_labels_merged = np.argmax(input_labels, axis=-1)

    assert (input_labels_merged == resized_original_labels_np).all()
