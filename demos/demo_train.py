import os

from keras.optimizers import Adam, Optimizer

from sem_seg.data.generator import DataGenerator
from sem_seg.models.deeplabv3plus import Deeplabv3
from sem_seg.utils.paths import KITTI_BASE_DIR, MODELS_DIR

import pathlib as pl

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    # CREATE
    labels = [4, 5, 6, 7, 8, 9]
    model = Deeplabv3(weights=None, input_shape=(128, 128, 3), num_classes=len(labels), OS=8)

    # COMPILE THE MODEL
    loss: str = 'categorical_crossentropy'
    optimizer: Optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6)
    metrics = ['categorical_accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # TRAIN
    num_epochs: int = 1
    batch_size: int = 4
    train_generator = DataGenerator(KITTI_BASE_DIR, 'train.txt', (128, 128), batch_size=batch_size, labels=labels)
    validation_generator = DataGenerator(KITTI_BASE_DIR, 'val.txt', (128, 128), batch_size=batch_size, labels=labels)

    model.fit_generator(generator=train_generator, verbose=2, validation_data=validation_generator)

    # SAVE THE MODEL
    print(f'Saving model...')
    model_path: pl.Path = MODELS_DIR / 'model_weights.h5'
    model.save_weights(str(model_path))