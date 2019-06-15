import pathlib as pl

DATA_DIR: pl.Path = pl.Path.home() / 'University' / 'NN' / 'Project' / 'data'

KITTI: pl.Path = DATA_DIR / 'kitti' / 'data_semantics' / 'training'
KITTI_RAW_IMAGE: pl.Path = KITTI / 'image_2' / 'all'
KITTI_LABEL_IMAGE: pl.Path = KITTI / 'semantic' / 'all'
