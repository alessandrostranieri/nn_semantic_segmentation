import os

import matplotlib.pyplot as plt
import numpy as np

from sem_seg.data.generator import DataGenerator
from sem_seg.models.deeplabv3plus import Deeplabv3
from sem_seg.utils.labels import generate_semantic_rgb
from sem_seg.utils.paths import MODELS_DIR, KITTI_BASE_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':

    labels = [4, 5, 6, 7, 8, 9]

    # LOAD THE MODEL
    model = Deeplabv3(weights=None, input_shape=(128, 128, 3), num_classes=len(labels), OS=8)
    model.load_weights(str(MODELS_DIR / 'model_weights.h5'))

    # GET A VALIDATION BATCH
    validation_generator: DataGenerator = DataGenerator(KITTI_BASE_DIR,
                                                        'val.txt',
                                                        target_size=(128, 128),
                                                        batch_size=1,
                                                        active_labels=labels)
    input_images, input_labels = validation_generator[0]

    # GENERATE A PREDICTION
    predicted: np.ndarray = model.predict(input_images)
    predicted_labels: np.ndarray = np.argmax(predicted.squeeze(), -1)
    print(f'Predicted label shape: {predicted_labels.shape}')

    # WE MAP VALUES TO THE RIGHT LABELS
    replacements = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9}
    new_predicted: np.ndarray = np.zeros_like(predicted_labels, dtype=int)
    for k, v in replacements.items():
        new_predicted[predicted_labels == k] = v

    # VISUALIZE ORIGINAL, TARGET AND PREDICTED
    original_image: np.ndarray = input_images.squeeze()
    print(f'Original image shape: {original_image.shape}')

    original_labels: np.ndarray = np.argmax(input_labels.squeeze(), -1)
    print(f'Original labels shape: {original_labels.shape}')

    # RGB LABEL IMAGES
    original_labels_rgb: np.ndarray = generate_semantic_rgb(original_labels, labels=labels)
    predicted_labels_rgb: np.ndarray = generate_semantic_rgb(predicted_labels, labels=labels)

    # PLOT IMAGES
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)

    ax11.imshow(original_image)
    ax12.imshow(original_labels_rgb)
    ax21.imshow(predicted_labels_rgb)

    plt.show()
