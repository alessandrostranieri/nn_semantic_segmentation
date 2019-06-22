from tensorflow.python.keras.losses import categorical_crossentropy


def categorical_crossentropy_with_logits(y_true, y_pred):
    return categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)