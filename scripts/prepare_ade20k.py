from typing import Dict, List, Optional, Any

import numpy as np
from scipy.io import loadmat

from sem_seg.data.transformations import from_pil_to_np
from sem_seg.utils.paths import ADE20K_INDEX_FILE, ADE20K_BASE_DIR
import pandas as pd
import pathlib as pl
from PIL import Image

"""
This script should:

1. Select the images to be used for train and validation based on labels
2. Create label images and store them in folder
3. Create train and val files for data sources
"""


def contains_cars(image_index: int, presence_matrix: np.ndarray) -> bool:
    image_vector: np.ndarray = presence_matrix[:, image_index]
    car_index: int = 400
    count = image_vector[car_index]
    return count > 0


if __name__ == '__main__':

    ade20k_index: np.ndarray = loadmat(str(ADE20K_INDEX_FILE))['index']

    print(f'Info from ade20k index file:')

    file_names: List[str] = [file_name[0] for file_name in ade20k_index['filename'][0][0][0]]
    print(f'{len(file_names)} file names')

    folder_names: List[str] = [folder_name[0] for folder_name in ade20k_index['folder'][0][0][0]]
    print(f'{len(folder_names)} folder names')

    object_names: List[str] = [object_name[0] for object_name in ade20k_index['objectnames'][0][0][0]]
    print(f'{len(object_names)} object names')

    object_presence: np.ndarray = ade20k_index['objectPresence'][0][0]
    print(f'Object-presence matrix size: {object_presence.shape}')

    ade20k_labels_df: pd.DataFrame = pd.DataFrame({'objectnames': object_names})
    ade20k_labels_df.to_csv(ADE20K_BASE_DIR / 'ade20_labels.csv')

    semantic_folder: pl.Path = ADE20K_BASE_DIR / 'semantic'
    semantic_folder.mkdir(exist_ok=True)

    train_semantic: pl.Path = semantic_folder / 'training'
    train_semantic.mkdir(exist_ok=True)

    validation_semantic: pl.Path = semantic_folder / 'validation'
    validation_semantic.mkdir(exist_ok=True)

    # LOOP THROUGH IMAGE
    output_sample_counter = 0
    for image_index, (folder, image_file) in enumerate(zip(folder_names, file_names)):

        if contains_cars(image_index, object_presence):
            output_sample_counter += 1
            # OPEN SEGMENTATION IMAGE NAME
            segmentation_file: str = image_file.replace('.jpg', '_seg.png')
            segmentation_image_path: pl.Path = ADE20K_BASE_DIR / folder / segmentation_file
            segmentation_image: Image.Image = Image.open(segmentation_image_path)
            segmentation_matrix: np.ndarray = from_pil_to_np(segmentation_image)
            # CREATE A ONE LAYER MATRIX WHICH IS THE SUM OF THE FIRST TWO LAYERS
            new_segmentation_matrix: np.ndarray = (segmentation_matrix[:, :, 0] / 10) * 256 + \
                                                  segmentation_matrix[:, :, 1]
            # TRANSFORM THE CAR LABEL INTO THE OTHER SYSTEM SYSTEM
            transformed_segmentation_matrix: np.ndarray = np.where(new_segmentation_matrix == 401, 255, 0).astype(np.uint8)
            # CREATE RGB AND SEGMENTATION IMAGE PATHS
            output_rgb_image_path: pl.Path = semantic_folder / image_file
            output_segmentation_image_path: pl.Path = semantic_folder / segmentation_file
            # CREATE IMAGES TO FOLDER
            output_segmentation: Image = Image.fromarray(transformed_segmentation_matrix)
            print(output_segmentation.getpixel((200, 700)))
            print(output_segmentation.mode)
            output_segmentation.save(output_segmentation_image_path)
            output_rgb: Optional[Any] = Image.open(ADE20K_BASE_DIR / folder / image_file)
            output_rgb.save(output_rgb_image_path)

            if output_sample_counter == 1:
                break

    print(f"Number of images stored: {output_sample_counter}")
