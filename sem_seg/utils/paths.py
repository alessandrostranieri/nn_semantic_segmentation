import pathlib as pl

DATA_DIR: pl.Path = pl.Path.home() / 'University' / 'NN' / 'Project' / 'data'

MODELS_DIR: pl.Path = DATA_DIR / 'models'

SETS_DIR: str = 'ImageSets'
IMAGE_DIR: str = 'JPEGImages'
LABEL_DIR: str = 'SegmentationClass'

KITTI_BASE_DIR: pl.Path = DATA_DIR / 'kitti'
KITTI_TRAINING: pl.Path = KITTI_BASE_DIR / 'data_semantics' / 'training'
KITTI_RAW_IMAGES_DIR: pl.Path = KITTI_TRAINING / 'image_2' / 'all'
KITTI_LABEL_IMAGES_DIR: pl.Path = KITTI_TRAINING / 'semantic' / 'all'

CITYSCAPES_BASE_DIR: pl.Path = DATA_DIR / 'cityscapes'

ADE20K_BASE_DIR: pl.Path = DATA_DIR / 'ade20k'
ADE20K_INDEX_FILE: pl.Path = ADE20K_BASE_DIR / 'ADE20K_2016_07_26' / 'index_ade20k.mat'