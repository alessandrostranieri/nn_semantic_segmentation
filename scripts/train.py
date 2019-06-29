import pathlib as pl
from argparse import ArgumentParser
from typing import Tuple, List, Dict

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Optimizer, Adam, SGD

from sem_seg.data.data_source import DataSource, KittiDataSource, CityscapesDataSource
from sem_seg.data.generator import DataGenerator
from sem_seg.data.transformations import RandomCrop
from sem_seg.models.unet import unet
from sem_seg.utils.labels import CityscapesLabels
from sem_seg.utils.paths import KITTI_BASE_DIR, CITYSCAPES_BASE_DIR, MODELS_DIR

# PARSE ARGUMENTS
parser: ArgumentParser = ArgumentParser(description='Train Script')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--patience', type=int, default=10, help='Early Stopping Patience')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of Epochs')
parser.add_argument('--input_size', type=int, default=256, help='Target size of input images')
parser.add_argument('--random_seed', type=int, default=42, help='Seed for data generation reproducibility')
parser.add_argument('--dataset', type=str, default='cityscapes', nargs='+', help='Datasets to use as source')
parser.add_argument('--limit', type=int, nargs='?', default=None,
                    help='Whether data-sources should limit the files produced(Useful during testing)')
parser.add_argument('--optim', type=str, default='sgd',
                    help='Optimizer: adam or sgd')
parser.add_argument('--save_over', action='store_true', default=False,
                    help='When input, model is overwritten')
parser.add_argument('--save_dir', type=str, required=True,
                    help='Folder inside the model directory where to save')

args = parser.parse_args()

print('Summary of Arguments')
print(args)

# SHORT PARAMETERS
image_size: Tuple[int, int] = (args.input_size, args.input_size)
input_size: Tuple[int, int, int] = image_size + (3,)
batch_size: int = args.batch_size
num_epochs: int = args.num_epochs
patience: int = args.patience
random_seed: int = args.random_seed
datasets: List[str] = args.dataset
limit: int = args.limit
save_dir: str = args.save_dir
save_over: bool = args.save_over
optim: str = args.optim

# CHECKS
save_full_path: pl.Path = MODELS_DIR / save_dir
if not save_over:
    assert not save_full_path.exists(), f'{save_full_path} already exists. Please delete it or choose another directory'
save_full_path.mkdir(exist_ok=True)

# CREATE DATA GENERATOR
data_sources_dict = {'kitti': KittiDataSource(KITTI_BASE_DIR, limit=limit),
                     'cityscapes': CityscapesDataSource(CITYSCAPES_BASE_DIR, limit=limit)}
data_sources: List[DataSource] = []
for data_source in datasets:
    data_sources.append(data_sources_dict[data_source])
train_generator = DataGenerator(data_sources=data_sources,
                                phase='train',
                                transformation=RandomCrop(target_size=image_size, random_seed=random_seed),
                                target_size=image_size,
                                batch_size=batch_size,
                                active_labels=CityscapesLabels.ALL,
                                random_seed=random_seed)
validation_generator = DataGenerator(data_sources=data_sources,
                                     phase='val',
                                     transformation=RandomCrop(target_size=image_size, random_seed=random_seed),
                                     target_size=image_size,
                                     batch_size=batch_size,
                                     active_labels=CityscapesLabels.ALL,
                                     random_seed=random_seed)

# CREATE UNET
class_layouts: Dict[str, List[int]] = dict()
for ds in datasets:
    class_layouts[ds] = CityscapesLabels.ALL
model = unet(input_size=input_size, class_layouts=class_layouts)
losses: Dict[str, str] = dict()
for ds in datasets:
    losses[ds] = 'categorical_crossentropy'
optimizers: Dict[str, Optimizer] = {'adam': Adam(),
                                    'sgd': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)}
optimizer: Optimizer = optimizers[optim]
metrics = ['categorical_accuracy']
model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
model.summary()

# TRAIN MODEL
model_path: pl.Path = save_full_path / 'model.h5'
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
history_df.to_csv(save_full_path / 'history.csv')
