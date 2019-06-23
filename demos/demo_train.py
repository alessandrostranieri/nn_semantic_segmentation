from typing import Tuple, List

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, Optimizer

from sem_seg.data.data_source import DataSource, KittiDataSource, CityscapesDataSource
from sem_seg.data.generator import DataGenerator
from sem_seg.models.deeplabv3plus import Deeplabv3
from sem_seg.models.losses import categorical_crossentropy_with_logits
from sem_seg.utils.paths import KITTI_BASE_DIR, MODELS_DIR, CITYSCAPES_BASE_DIR

import pathlib as pl
import pandas as pd

if __name__ == '__main__':
    labels = [0,  # UNLABELLED
              7,  # ROAD
              21,  # VEGETATION
              24,  # PERSON
              26]  # CAR

    image_shape: Tuple[int, int] = (256, 256)
    image_array_shape: Tuple[int, int, int] = image_shape + (3,)

    model = Deeplabv3(weights=None, input_shape=image_array_shape, num_classes=len(labels), OS=8)

    # COMPILE THE MODEL
    # loss: str = 'categorical_crossentropy'
    loss = categorical_crossentropy_with_logits
    optimizer: Optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6)
    metrics = ['categorical_accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # TRAIN
    num_epochs: int = 50
    batch_size: int = 4
    data_sources: List[DataSource] = [CityscapesDataSource(CITYSCAPES_BASE_DIR)]
    train_generator = DataGenerator(data_sources=data_sources, phase='train', target_size=image_shape,
                                    batch_size=batch_size,
                                    active_labels=labels)
    validation_generator = DataGenerator(data_sources=data_sources, phase='val', target_size=image_shape,
                                         batch_size=batch_size,
                                         active_labels=labels)

    model_path: pl.Path = MODELS_DIR / 'model_weights.h5'
    model_checkpoint = ModelCheckpoint(str(model_path), save_best_only=True, save_weights_only=True, verbose=1)
    early_stopping = EarlyStopping(patience=5, verbose=1)
    history = model.fit_generator(generator=train_generator,
                                  epochs=num_epochs,
                                  callbacks=[model_checkpoint, early_stopping],
                                  verbose=2,
                                  validation_data=validation_generator)

    # SAVE HISTORY
    print(f'Saving History')
    history_df: pd.DataFrame = pd.DataFrame(history.history)
    history_df.to_csv(MODELS_DIR / 'history.csv')
