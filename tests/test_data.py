import numpy as np

from sem_seg.utils.labels import split_label_image, merge_label_images, generate_semantic_rgb


def test_split_label_image():
    ms: int = 12

    # CREATE INPUT MASK
    input_mask: np.ndarray = np.zeros(shape=(ms, ms, 1))
    # INSERT 2 LABELLED SHAPES IN THE MIDDLE
    input_mask[6:8, 6:8, :] = 1.0
    input_mask[1:3, 1:3, :] = 2.0
    # print(input_mask)

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

    actual: np.ndarray = generate_semantic_rgb(input_image, labels=[0, 7, 24])

    assert (actual == expected_image).all()
