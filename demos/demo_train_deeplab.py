import pathlib as pl
from typing import Tuple, List

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, Optimizer

from sem_seg.data.data_source import DataSource, KittiDataSource
from sem_seg.data.generator import DataGenerator
from sem_seg.data.transformations import Crop
from sem_seg.models.deeplabv3plus import Deeplabv3
from sem_seg.models.losses import categorical_crossentropy_with_logits
from sem_seg.utils.paths import MODELS_DIR, KITTI_BASE_DIR

if __name__ == '__main__':
    labels = [0,  # UNLABELLED
              7,  # ROAD
              21,  # VEGETATION
              24,  # PERSON
              26]  # CAR

    image_shape: Tuple[int, int] = (64, 64)
    image_array_shape: Tuple[int, int, int] = image_shape + (3,)

    model = Deeplabv3(weights=None, input_shape=image_array_shape, num_classes=len(labels), OS=8)

    # COMPILE THE MODEL
    loss = categorical_crossentropy_with_logits
    optimizer: Optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6)
    metrics = ['categorical_accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # TRAIN
    num_epochs: int = 1
    batch_size: int = 4
    data_sources: List[DataSource] = [KittiDataSource(KITTI_BASE_DIR, limit=4)]
    train_generator = DataGenerator(data_sources=data_sources, phase='train', target_size=image_shape,
                                    batch_size=batch_size,
                                    transformation=Crop(image_shape),
                                    active_labels=labels)
    validation_generator = DataGenerator(data_sources=data_sources, phase='val', target_size=image_shape,
                                         batch_size=batch_size,
                                         transformation=Crop(image_shape),
                                         active_labels=labels)

    model_path: pl.Path = MODELS_DIR / 'demo' / 'model_weights.h5'
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
    history_df.to_csv(MODELS_DIR / 'demo' / 'history.csv')
