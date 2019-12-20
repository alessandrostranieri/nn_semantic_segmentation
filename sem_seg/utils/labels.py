from enum import Enum, unique
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image, ImageOps


class LabelColor:
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
segmentation_labels: List[LabelColor] = [
    LabelColor('unlabeled'            ,  0 , (  0,  0,  0) ),
    LabelColor('ego vehicle'          ,  1 , (  0,  0,  0) ),
    LabelColor('rectification border' ,  2 , (  0,  0,  0) ),
    LabelColor('out of roi'           ,  3 , (  0,  0,  0) ),
    LabelColor('static'               ,  4 , (  0,  0,  0) ),
    LabelColor('dynamic'              ,  5 , (111, 74,  0) ),
    LabelColor('ground'               ,  6 , ( 81,  0, 81) ),
    LabelColor('road'                 ,  7 , (128, 64,128) ),
    LabelColor('sidewalk'             ,  8 , (244, 35,232) ),
    LabelColor('parking'              ,  9 , (250,170,160) ),
    LabelColor('rail track'           , 10 , (230,150,140) ),
    LabelColor('building'             , 11 , ( 70, 70, 70) ),
    LabelColor('wall'                 , 12 , (102,102,156) ),
    LabelColor('fence'                , 13 , (190,153,153) ),
    LabelColor('guard rail'           , 14 , (180,165,180) ),
    LabelColor('bridge'               , 15 , (150,100,100) ),
    LabelColor('tunnel'               , 16 , (150,120, 90) ),
    LabelColor('pole'                 , 17 , (153,153,153) ),
    LabelColor('polegroup'            , 18 , (153,153,153) ),
    LabelColor('traffic light'        , 19 , (250,170, 30) ),
    LabelColor('traffic sign'         , 20 , (220,220,  0) ),
    LabelColor('vegetation'           , 21 , (107,142, 35) ),
    LabelColor('terrain'              , 22 , (152,251,152) ),
    LabelColor('sky'                  , 23 , ( 70,130,180) ),
    LabelColor('person'               , 24 , (220, 20, 60) ),
    LabelColor('rider'                , 25 , (255,  0,  0) ),
    LabelColor('car'                  , 26 , (  0,  0,142) ),
    LabelColor('truck'                , 27 , (  0,  0, 70) ),
    LabelColor('bus'                  , 28 , (  0, 60,100) ),
    LabelColor('caravan'              , 29 , (  0,  0, 90) ),
    LabelColor('trailer'              , 30 , (  0,  0,110) ),
    LabelColor('train'                , 31 , (  0, 80,100) ),
    LabelColor('motorcycle'           , 32 , (  0,  0,230) ),
    LabelColor('bicycle'              , 33 , (119, 11, 32) ),
    LabelColor('license plate'        , -1 , (  0,  0,142) ),
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


@unique
class Label(Enum):

    CAR = 1
    PERSON = 2
    ROAD = 3
    SKY = 4
    TREE = 5


class Ade20Label:

    def __init__(self, id: int, names: List[str]) -> None:
        super().__init__()

        self.id = id
        self.names = names
