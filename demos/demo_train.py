import os

from keras.optimizers import Adam, Optimizer
from keras.losses import sparse_categorical_crossentropy

from sem_seg.data.generator import DataGenerator
from sem_seg.models.deeplabv3plus import Deeplabv3
from sem_seg.utils.paths import KITTI_BASE_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    # CREATE
    classes = [4, 5, 6, 7, 8, 9]
    model = Deeplabv3(weights=None, input_shape=(128, 128, 3), num_classes=len(classes), OS=8)

    # COMPILE THE MODEL
    loss: str = 'categorical_crossentropy'
    optimizer: Optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6)
    metrics = ['categorical_accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(model.summary())

    # TRAIN
    num_epochs: int = 1
    batch_size: int = 4
    train_generator = DataGenerator(KITTI_BASE_DIR, 'train.txt', (128, 128), batch_size=batch_size, classes=classes)

    model.fit_generator(generator=train_generator, verbose=2)
