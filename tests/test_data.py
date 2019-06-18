import numpy as np

from sem_seg.utils.labels import split_label_image


def test_time_in_words():

    ms: int = 12

    # CREATE INPUT MASK
    input_mask: np.ndarray = np.zeros(shape=(ms, ms, 1))
    # INSERT 2 LABELLED SHAPES IN THE MIDDLE
    input_mask[6:8, 6:8, :] = 1.0
    input_mask[1:3, 1:3, :] = 2.0
    # print(input_mask)

    expected_mask: np.ndarray = np.zeros(shape=(ms, ms, 2))
    expected_mask[6:8, 6:8, 0] = 1.0
    expected_mask[1:3, 1:3, 1] = 1.0

    actual: np.ndarray = split_label_image(input_mask, [1, 2])

    assert (actual == expected_mask).all()
