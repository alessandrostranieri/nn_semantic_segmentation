import pathlib as pl
from argparse import ArgumentParser
from shutil import copyfile
from typing import List, Dict

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sem_seg.data.generator import DataGenerator
from sem_seg.models.unet import unet
from sem_seg.utils.configuration import Configuration
from sem_seg.utils.labels import CityscapesLabels
from sem_seg.utils.paths import MODELS_DIR

# PARSE ARGUMENTS
parser: ArgumentParser = ArgumentParser(description='Train Script')
parser.add_argument('config_file', type=str, help='Path to configuration file')
args = parser.parse_args()

config_file_path: pl.Path = pl.Path(args.config_file)
assert config_file_path.exists(), f'Configuration file not found'
configuration: Configuration = Configuration(config_file_path)

# CHECK SAVING FOLDER AND COPY CONFIGURATION
save_full_path: pl.Path = MODELS_DIR / configuration.save_dir
if not configuration.save_over:
    assert not save_full_path.exists(), f'{save_full_path} already exists. Please delete it or choose another directory'
save_full_path.mkdir(exist_ok=True)
copyfile(str(config_file_path), str(save_full_path / 'config.yml'))

# CREATE DATA GENERATOR
train_generator = DataGenerator(data_sources=configuration.datasets,
                                phase='train',
                                transformation=configuration.transformation,
                                target_size=configuration.input_size,
                                batch_size=configuration.batch_size,
                                active_labels=CityscapesLabels.ALL,
                                random_seed=configuration.random_seed)
train_generator.summary()
validation_generator = DataGenerator(data_sources=configuration.datasets,
                                     phase='val',
                                     transformation=configuration.transformation,
                                     target_size=configuration.input_size,
                                     batch_size=configuration.batch_size,
                                     active_labels=CityscapesLabels.ALL,
                                     random_seed=configuration.random_seed)

# CREATE UNET
class_layouts: Dict[str, List[int]] = dict()
for ds in configuration.datasets:
    class_layouts[ds.get_name()] = CityscapesLabels.ALL
model = unet(input_size=configuration.input_size + (3,), class_layouts=class_layouts)
# TRAINING OPTIONS
losses: Dict[str, str] = {}
for ds in configuration.datasets:
    losses[ds.get_name()] = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
model.compile(optimizer=configuration.optimizer, loss=losses, metrics=metrics)
model.summary()

# TRAIN MODEL
model_path: pl.Path = save_full_path / 'model.h5'
model_checkpoint = ModelCheckpoint(str(model_path), save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience=configuration.patience, verbose=1)
history = model.fit_generator(generator=train_generator,
                              epochs=configuration.num_epochs,
                              callbacks=[model_checkpoint, early_stopping],
                              verbose=2,
                              validation_data=validation_generator)

# SAVE HISTORY
print(f'Saving History')
history_df: pd.DataFrame = pd.DataFrame(history.history)
history_df.to_csv(save_full_path / 'history.csv')
