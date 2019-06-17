import pathlib as pl

DATA_DIR: pl.Path = pl.Path.home() / 'University' / 'NN' / 'Project' / 'data'

SETS_DIR: str = 'ImageSets'
IMAGE_DIR: str = 'JPEGImages'
LABEL_DIR: str = 'SegmentationClass'

KITTI_BASE_DIR: pl.Path = DATA_DIR / 'kitti'
KITTI_TRAINING: pl.Path = KITTI_BASE_DIR / 'data_semantics' / 'training'
KITTI_RAW_IMAGES_DIR: pl.Path = KITTI_TRAINING / 'image_2' / 'all'
KITTI_LABEL_IMAGES_DIR: pl.Path = KITTI_TRAINING / 'semantic' / 'all'

