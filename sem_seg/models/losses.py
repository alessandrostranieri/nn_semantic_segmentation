from tensorflow.python.keras.losses import categorical_crossentropy


def categorical_crossentropy_with_logits(y_true, y_pred):
    return categorical_crossentropy(target=y_true, output=y_pred, from_logits=True)