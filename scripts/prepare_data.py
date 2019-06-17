import pathlib as pl
from shutil import copyfile
from typing import List, Dict

from sklearn.model_selection import train_test_split

from sem_seg.utils.paths import KITTI_RAW_IMAGES_DIR, KITTI_LABEL_IMAGES_DIR, SETS_DIR, KITTI_BASE_DIR, IMAGE_DIR, \
    LABEL_DIR

if __name__ == '__main__':
    """
    This script is used to arrange different datasets into the same layout. Given a dataset type, the script will:
    
    1. Move image and label files to respective folders
    2. Create train and validation data text files 
    """
    # STORE ALL IMAGE FILE NAMES
    original_image_dir: pl.Path = KITTI_RAW_IMAGES_DIR
    image_names: List[str] = [image_name.name for image_name in original_image_dir.glob('*.png')]

    # GET ALL LABEL IMAGES AND CHECK INTEGRITY

    original_label_dir: pl.Path = KITTI_LABEL_IMAGES_DIR
    label_names: List[str] = [label_name.name for label_name in original_label_dir.glob('*.png')]
    assert image_names == label_names, "Image and Label file names are different."

    # PERFORM TRAIN/VAL SPLIT - CHECK AGAIN
    x_train, x_val, y_train, y_val = train_test_split(image_names, label_names, test_size=0.2, random_state=42)
    assert x_train == y_train, "Training Image and Label sets are different."
    assert x_train == y_train, "Validation Image and Label sets are different."

    # CREATE DESTINATION DIRECTORIES
    training_directories: Dict[str, pl.Path] = {
        'sets': KITTI_BASE_DIR / SETS_DIR,
        'images': KITTI_BASE_DIR / IMAGE_DIR,
        'labels': KITTI_BASE_DIR / LABEL_DIR
    }
    for name, directory in training_directories.items():
        if directory.exists():
            print(f'Directory {directory} exists')
            assert len(list(directory.glob('*'))) == 0, f'Directory {directory} not empty'
        else:
            directory.mkdir()
            if directory.exists():
                print(f'Directory {name}:{directory} created')

        if name == 'sets':
            # CREATE TRAIN AND VALIDATION FILES
            training_file: str = str(training_directories[name] / 'train.txt')
            with open(training_file, mode='w') as f:
                for line in x_train:
                    f.write(line + '\n')
            validation_file: str = str(training_directories[name] / 'val.txt')
            with open(validation_file, mode='w') as f:
                for line in x_val:
                    f.write(line + '\n')

        elif name == 'images':
            for image_name in image_names:
                copyfile(src=str(KITTI_RAW_IMAGES_DIR / image_name),
                         dst=str(training_directories[name] / image_name))

        elif name == 'labels':
            for label_name in label_names:
                copyfile(src=str(KITTI_LABEL_IMAGES_DIR / label_name),
                         dst=str(training_directories[name] / label_name))
