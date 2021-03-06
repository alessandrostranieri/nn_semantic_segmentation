from typing import Tuple, List

import numpy as np
from PIL import Image, ImageOps


class Label:
    """Represents a segmentation label."""

    def __init__(self, name: str, label_id: int, color: Tuple[int, int, int]) -> None:
        super().__init__()

        self.name = name
        self.label_id = label_id
        self.color = color


class CityscapesLabels:
    ALL = list(range(0, 34))


# @formatter:off
# noinspection PyPep8
segmentation_labels: List[Label] = [
    Label('unlabeled'            ,  0 , (  0,  0,  0) ),
    Label('ego vehicle'          ,  1 , (  0,  0,  0) ),
    Label('rectification border' ,  2 , (  0,  0,  0) ),
    Label('out of roi'           ,  3 , (  0,  0,  0) ),
    Label('static'               ,  4 , (  0,  0,  0) ),
    Label('dynamic'              ,  5 , (111, 74,  0) ),
    Label('ground'               ,  6 , ( 81,  0, 81) ),
    Label('road'                 ,  7 , (128, 64,128) ),
    Label('sidewalk'             ,  8 , (244, 35,232) ),
    Label('parking'              ,  9 , (250,170,160) ),
    Label('rail track'           , 10 , (230,150,140) ),
    Label('building'             , 11 , ( 70, 70, 70) ),
    Label('wall'                 , 12 , (102,102,156) ),
    Label('fence'                , 13 , (190,153,153) ),
    Label('guard rail'           , 14 , (180,165,180) ),
    Label('bridge'               , 15 , (150,100,100) ),
    Label('tunnel'               , 16 , (150,120, 90) ),
    Label('pole'                 , 17 , (153,153,153) ),
    Label('polegroup'            , 18 , (153,153,153) ),
    Label('traffic light'        , 19 , (250,170, 30) ),
    Label('traffic sign'         , 20 , (220,220,  0) ),
    Label('vegetation'           , 21 , (107,142, 35) ),
    Label('terrain'              , 22 , (152,251,152) ),
    Label('sky'                  , 23 , ( 70,130,180) ),
    Label('person'               , 24 , (220, 20, 60) ),
    Label('rider'                , 25 , (255,  0,  0) ),
    Label('car'                  , 26 , (  0,  0,142) ),
    Label('truck'                , 27 , (  0,  0, 70) ),
    Label('bus'                  , 28 , (  0, 60,100) ),
    Label('caravan'              , 29 , (  0,  0, 90) ),
    Label('trailer'              , 30 , (  0,  0,110) ),
    Label('train'                , 31 , (  0, 80,100) ),
    Label('motorcycle'           , 32 , (  0,  0,230) ),
    Label('bicycle'              , 33 , (119, 11, 32) ),
    Label('license plate'        , -1 , (  0,  0,142) ),
]
# @formatter:on


def generate_semantic_rgb(label_image: np.ndarray) -> np.ndarray:
    """
    Generate a color coded label image from an original label image
    :param label_image: 1-channel label image
    :return: 3-channels color coded
    """
    # WORKING LABEL IMAGE MUST BE PURE 2D
    input_shape = label_image.shape
    assert len(input_shape) == 2 or input_shape[2] == 1, 'Input image must be single channel'
    if label_image.ndim == 3:
        label_image = np.squeeze(label_image, -1)

    # CREATE OUTPUT IMAGE
    label_rgb_image: np.ndarray = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)

    for label in segmentation_labels:
        label_id: int = label.label_id
        label_color: Tuple[int, int, int] = label.color

        # GET MASK
        mask = (label_image == label_id)

        label_rgb_image[mask] = label_color

    return label_rgb_image


def resize_and_crop(input_image: np.ndarray, target_size: Tuple[int, int]) -> Image.Image:
    assert input_image.dtype == np.uint8, "Input array type must be np.uint8"

    mode: str = 'RGB' if input_image.ndim == 3 else 'L'
    input_pil_image: Image.Image = Image.fromarray(input_image, mode)
    fitted = ImageOps.fit(input_pil_image, target_size)

    return fitted
