import pathlib as pl
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.engine.saving import load_model

from sem_seg.data.data_source import KittiDataSource, DataSource
from sem_seg.data.generator import DataGenerator
from sem_seg.data.transformations import Crop
from sem_seg.utils.labels import generate_semantic_rgb, CityscapesLabels
from sem_seg.utils.paths import MODELS_DIR, KITTI_BASE_DIR

if __name__ == '__main__':

    image_shape: Tuple[int, int] = (256, 256)
    image_array_shape: Tuple[int, int, int] = image_shape + (3,)

    # LOAD MODEL
    model_dir: pl.Path = MODELS_DIR / 'kitti_02'
    model: Model = load_model(str(model_dir / 'model.h5'))

    # GET A VALIDATION BATCH
    data_sources: List[DataSource] = [KittiDataSource(KITTI_BASE_DIR)]
    validation_generator: DataGenerator = DataGenerator(data_sources=data_sources,
                                                        phase='val',
                                                        target_size=image_shape,
                                                        transformation=Crop(image_shape),
                                                        batch_size=4,
                                                        active_labels=CityscapesLabels.ALL)
    input_images, input_labels, _ = validation_generator[0]
    original_batch = validation_generator.get_batch(0)
    original_image, original_labels, _ = original_batch[0]
    original_size: Tuple[int, int] = original_image.size

    # GENERATE A PREDICTION
    predicted: np.ndarray = model.predict(input_images)[0]
    # CONVERT TO LABEL IMAGES
    predicted_labels: np.ndarray = np.argmax(predicted.squeeze(), -1)

    # VISUALIZE ORIGINAL, TARGET AND PREDICTED
    original_labels_array = np.array(original_labels)
    original_labels_rgb = generate_semantic_rgb(original_labels_array)
    predicted_rgb = generate_semantic_rgb(predicted_labels)

    # PLOT IMAGES
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # ORIGINAL IMAGE
    ax1.set_title('Original')
    ax1.imshow(original_image)
    # FULL SEGMENTATION
    ax2.set_title('Semantic: Original')
    ax2.imshow(original_labels_rgb)

    # PREDICTED SEGMENTATION
    ax3.set_title('Semantic: Predicted')
    predicted_rgb_resized = np.array(predicted_rgb)
    ax3.imshow(predicted_rgb_resized)

    plt.show()
