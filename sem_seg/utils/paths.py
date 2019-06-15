import pathlib as pl

DATA_DIR: pl.Path = pl.Path.home() / 'University' / 'NN' / 'Project' / 'data'

KITTI_BASE: pl.Path = DATA_DIR / 'kitti'
KITTI_TRAINING: pl.Path = KITTI_BASE / 'data_semantics' / 'training'
KITTI_RAW_IMAGE: pl.Path = KITTI_TRAINING / 'image_2' / 'all'
KITTI_LABEL_IMAGE: pl.Path = KITTI_TRAINING / 'semantic' / 'all'
