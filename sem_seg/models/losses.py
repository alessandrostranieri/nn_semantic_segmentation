# noinspection PyPep8Naming
from tensorflow.python.keras import backend as K


def categorical_crossentropy_with_logits(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)
