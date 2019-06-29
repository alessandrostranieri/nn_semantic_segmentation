from typing import List

import matplotlib.pyplot as plt
import numpy as np

from sem_seg.data.data_source import DataSource, CityscapesDataSource
from sem_seg.data.generator import DataGenerator
from sem_seg.data.transformations import merge_label_images, RandomCrop
from sem_seg.utils.labels import generate_semantic_rgb, CityscapesLabels
from sem_seg.utils.paths import CITYSCAPES_BASE_DIR

if __name__ == '__main__':
    """
    This demo allows to visually inspect what the generator is feeding to the model.
    """

    labels = CityscapesLabels.ALL

    # CREATE GENERATOR
    data_sources: List[DataSource] = [CityscapesDataSource(CITYSCAPES_BASE_DIR)]
    generator: DataGenerator = DataGenerator(data_sources=data_sources,
                                             phase='train',
                                             transformation=RandomCrop((256, 256)),
                                             batch_size=4,
                                             target_size=(256, 256),
                                             active_labels=labels)

    # GENERATOR ORIGINAL IMAGES
    original_image, original_labels, _ = generator.get_batch(0)[0]
    original_image_np: np.ndarray = np.array(original_image)
    original_labels_np: np.ndarray = np.array(original_labels)
    original_labels_rgb: np.ndarray = generate_semantic_rgb(original_labels_np)

    # GENERATOR PRE-PROCESSED IMAGES
    image_batch, labels_batch, _ = generator[0]

    # GET SINGLE IMAGES FROM BATCH
    input_image = image_batch[0] * 255
    input_image = input_image.astype(np.uint8)
    input_labels = labels_batch['cityscapes'][0]
    input_labels = input_labels.astype(np.uint8)

    # COLORIZE LABELS
    merged_labels = merge_label_images(input_labels, labels)
    input_labels_rgb: np.ndarray = generate_semantic_rgb(merged_labels)

    # VISUALIZED DATA
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)

    ax11.set_title('Original Camera')
    ax11.imshow(original_image_np)

    ax21.set_title('Original Semantic')
    ax21.imshow(original_labels_rgb)

    ax12.set_title('Input Camera')
    ax12.imshow(input_image)
    ax22.set_title('Input Semantic')
    ax22.imshow(input_labels_rgb)

    plt.show()
