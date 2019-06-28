from typing import Tuple

import numpy as np
from PIL import Image

from sem_seg.data.data_source import KittiDataSource
from sem_seg.data.transformations import split_label_image, merge_label_images, RandomCrop
from sem_seg.utils.labels import generate_semantic_rgb, pad_and_resize, \
    resize_and_crop
from sem_seg.utils.paths import KITTI_BASE_DIR


def test_split_label_image():
    ms: int = 12

    # CREATE INPUT MASK
    input_mask: np.ndarray = np.zeros(shape=(ms, ms))
    # INSERT 2 LABELLED SHAPES IN THE MIDDLE
    input_mask[6:8, 6:8] = 1.0
    input_mask[1:3, 1:3] = 2.0

    expected_mask: np.ndarray = np.zeros(shape=(ms, ms, 3))
    expected_mask[:, :, 0] = 1.0
    expected_mask[6:8, 6:8, 0] = 0.0
    expected_mask[1:3, 1:3, 0] = 0.0
    expected_mask[6:8, 6:8, 1] = 1.0
    expected_mask[1:3, 1:3, 2] = 1.0

    actual: np.ndarray = split_label_image(input_mask, [0, 1, 2])

    assert (actual == expected_mask).all()


def test_merge_label_images():
    ms: int = 12

    # CREATE INPUT MASK
    input_mask: np.ndarray = np.zeros(shape=(ms, ms, 3))
    input_mask[:, :, 0] = 1.0
    input_mask[6:8, 6:8, 0] = 0.0
    input_mask[1:3, 1:3, 0] = 0.0
    input_mask[6:8, 6:8, 1] = 1.0
    input_mask[1:3, 1:3, 2] = 1.0

    expected_mask: np.ndarray = np.zeros(shape=(ms, ms), dtype=int)
    # INSERT 2 LABELLED SHAPES IN THE MIDDLE
    expected_mask[6:8, 6:8] = 1
    expected_mask[1:3, 1:3] = 2

    actual: np.ndarray = merge_label_images(input_mask, [0, 1, 2])

    assert (actual == expected_mask).all()


def test_generate_semantic_rgb():
    ms: int = 12

    # CREATE DUMMY LABEL IMAGE
    input_image: np.ndarray = np.zeros(shape=(ms, ms))  # TYPE MIGHT NOT BE SPECIFIED
    # INSERT 2 LABELLED SHAPES IN THE MIDDLE
    input_image[6:8, 6:8] = 7  # ROAD
    input_image[1:3, 1:3] = 24  # PERSON

    # CREATE EXPECTED OUTPUT
    expected_image: np.ndarray = np.zeros(shape=(ms, ms, 3), dtype=int)
    expected_image[6:8, 6:8] = (128, 64, 128)  # ROAD
    expected_image[1:3, 1:3] = (220, 20, 60)  # PERSON

    actual: np.ndarray = generate_semantic_rgb(input_image)

    assert (actual == expected_image).all()


def test_image_pad_resize():
    # WHITE WIDE IMAGE
    dummy_image: Image.Image = Image.new(mode='RGB', size=(200, 100), color=(1, 1, 1))

    # EXPECTED
    expected: np.ndarray = np.zeros((12, 12, 3), dtype=np.uint8)
    expected[3:9, :, :] = (1, 1, 1)

    actual: Image.Image = pad_and_resize(dummy_image, (12, 12))
    actual_array: np.ndarray = np.array(actual)

    assert (expected == actual_array).all()


def test_resize_and_crop():
    # DUMMY IMAGE - SQUARE IMAGE WITH WHITE BANDS
    dummy_image: np.ndarray = np.zeros((12, 12, 3), dtype=np.uint8)
    dummy_image[3:9, :, :] = (1, 1, 1)

    # EXPECTED - JUST THE WHITE IMAGE
    expected: np.ndarray = np.ones(shape=(200, 100, 3), dtype=np.uint8)

    actual: Image.Image = resize_and_crop(dummy_image, (200, 100))
    actual_array: np.ndarray = np.array(actual).swapaxes(0, 1)

    assert (expected == actual_array).all()


def test_repeated_random_crop_returns_different_images():
    # OPEN AN IMAGE
    kitti_data_source: KittiDataSource = KittiDataSource(KITTI_BASE_DIR)
    train_data: Tuple[str, str] = kitti_data_source.get_train_data()
    test_camera_image_path: str = train_data[0][0]
    test_camera_image: Image.Image = Image.open(test_camera_image_path)

    # CROP IT TWICE
    random_crop: RandomCrop = RandomCrop(target_size=(16, 16))
    cropped_1: Image.Image = random_crop(test_camera_image)
    cropped_2: Image.Image = random_crop(test_camera_image)

    cropped_1_np: np.ndarray = np.array(cropped_1)
    cropped_2_np: np.ndarray = np.array(cropped_2)

    assert not np.array_equal(cropped_1_np, cropped_2_np)
