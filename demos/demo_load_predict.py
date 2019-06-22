import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from sem_seg.data.generator import DataGenerator
from sem_seg.models.deeplabv3plus import Deeplabv3
from sem_seg.utils.labels import generate_semantic_rgb, resize_and_crop, merge_label_images
from sem_seg.utils.paths import MODELS_DIR, KITTI_BASE_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':

    # PARAMETERS
    labels = [0,  # UNLABELLED
              7,  # ROAD
              21,  # VEGETATION
              24,  # PERSON
              26]  # CAR

    image_shape: Tuple[int, int] = (256, 256)
    image_array_shape: Tuple[int, int, int] = image_shape + (3,)

    # LOAD THE MODEL
    model = Deeplabv3(weights=None, input_shape=image_array_shape, num_classes=len(labels), OS=8)
    model.load_weights(str(MODELS_DIR / 'model_weights.h5'))

    # GET A VALIDATION BATCH
    validation_generator: DataGenerator = DataGenerator(KITTI_BASE_DIR,
                                                        'val.txt',
                                                        target_size=image_shape,
                                                        batch_size=1,
                                                        active_labels=labels)
    input_images, input_labels = validation_generator[0]
    original_batch = validation_generator.get_batch(0)
    original_image, original_labels = original_batch[0]

    # GENERATE A PREDICTION
    predicted: np.ndarray = model.predict(input_images)
    # CONVERT TO LABEL IMAGES
    predicted_labels: np.ndarray = np.argmax(predicted.squeeze(), -1)

    # WE MAP VALUES TO THE RIGHT LABELS
    replacements = {0: 0, 1: 7, 2: 21, 3: 24, 5: 26}
    new_predicted: np.ndarray = np.zeros_like(predicted_labels, dtype=np.uint8)
    for k, v in replacements.items():
        new_predicted[predicted_labels == k] = v

    # VISUALIZE ORIGINAL, TARGET AND PREDICTED
    original_labels_array = np.array(original_labels)
    original_labels_rgb = generate_semantic_rgb(original_labels_array)
    predicted_rgb = generate_semantic_rgb(new_predicted)

    # PLOT IMAGES
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)

    # ORIGINAL IMAGE
    ax11.set_title('Original')
    ax11.imshow(original_image)
    # FULL SEGMENTATION
    ax21.set_title('Semantic: Original')
    ax21.imshow(original_labels_rgb)

    # INPUT SEGMENTATION - RECONSTRUCTED
    ax12.set_title('Semantic: Input')

    # PREDICTED SEGMENTATION
    ax22.set_title('Semantic: Predicted')
    ax22.imshow(predicted_rgb)

    plt.show()
