from sem_seg.data.data_source import KittiDataSource
from sem_seg.utils.paths import KITTI_BASE_DIR
import pathlib as pl


def test_kitti_data_source_train():
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
