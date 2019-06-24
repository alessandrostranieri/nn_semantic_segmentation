from typing import Tuple, List

from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from keras.optimizers import Optimizer, Adam

from sem_seg.data.data_source import DataSource, CityscapesDataSource
from sem_seg.data.generator import DataGenerator
from sem_seg.models.unet import unet
from sem_seg.utils.labels import CityscapesLabels, generate_semantic_rgb
from sem_seg.utils.paths import CITYSCAPES_BASE_DIR, MODELS_DIR

import pandas as pd
import pathlib as pl
import numpy as np

from PIL import Image

if __name__ == '__main__':

    # PARAMETERS
    image_size: Tuple[int, int] = (128, 128)
    input_size: Tuple[int, int, int] = image_size + (3,)
    batch_size: int = 4
    num_epochs: int = 1
    patience: int = 1

    # CREATE DATA GENERATOR
    data_sources: List[DataSource] = [CityscapesDataSource(CITYSCAPES_BASE_DIR, limit=4)]
    train_generator = DataGenerator(data_sources=data_sources, phase='train', target_size=image_size,
                                    batch_size=batch_size,
                                    active_labels=CityscapesLabels.ALL)
    validation_generator = DataGenerator(data_sources=data_sources, phase='val', target_size=image_size,
                                         batch_size=batch_size,
                                         active_labels=CityscapesLabels.ALL)

    # CREATE UNET
    model = unet(input_size=input_size, num_classes=len(CityscapesLabels.ALL))
    loss = 'categorical_crossentropy'
    optimizer: Optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6)
    metrics = ['categorical_accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    # TRAIN MODEL
    model_path: pl.Path = MODELS_DIR / 'demo' / 'model.h5'
    model_checkpoint = ModelCheckpoint(str(model_path), save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    history = model.fit_generator(generator=train_generator,
                                  epochs=num_epochs,
                                  callbacks=[model_checkpoint, early_stopping],
                                  verbose=2,
                                  validation_data=validation_generator)

    # SAVE HISTORY
    print(f'Saving History')
    history_df: pd.DataFrame = pd.DataFrame(history.history)
    history_df.to_csv(MODELS_DIR / 'demo' / 'history.csv')

    # LOAD MODEL
    loaded_model: Model = load_model(str(MODELS_DIR / 'demo' / 'model.h5'))

    # PREDICT
    predicted_batch = loaded_model.predict(validation_generator[0][0])
    predicted_single = predicted_batch[0]

    # SAVE FOR INSPECTION
    predicted_semantic = np.argmax(predicted_single, axis=-1)
    predicted_semantic_rgb = generate_semantic_rgb(predicted_semantic)
    predicted_pil: Image.Image = Image.fromarray(predicted_semantic_rgb)
    predicted_pil.save(str(MODELS_DIR / 'demo' / 'predicted.png'))