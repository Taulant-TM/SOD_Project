from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K

EPS = 1e-7

def soft_iou_batch(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = EPS) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [1, 2, 3]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true + y_pred, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5, smooth: float = EPS) -> tf.Tensor:
    y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)
    axes = [1, 2, 3]
    inter = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
    union = tf.reduce_sum(y_true + y_pred_bin, axis=axes) - inter
    iou = (inter + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def precision_recall_f1(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5, smooth: float = EPS):
    y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    axes = [1, 2, 3]
    tp = tf.reduce_sum(y_true * y_pred_bin, axis=axes)
    fp = tf.reduce_sum((1 - y_true) * y_pred_bin, axis=axes)
    fn = tf.reduce_sum(y_true * (1 - y_pred_bin), axis=axes)

    precision = tf.reduce_mean((tp + smooth) / (tp + fp + smooth))
    recall = tf.reduce_mean((tp + smooth) / (tp + fn + smooth))
    f1 = 2.0 * (precision * recall) / (precision + recall + smooth)
    return precision, recall, f1


def combined_bce_iou_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    iou = soft_iou_batch(y_true, y_pred)
    loss = bce + 0.5 * (1.0 - iou)
    return loss


def _conv_block(x, filters: int, kernel_size: int = 3, batch_norm: bool = False):
    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = layers.ReLU()(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = layers.ReLU()(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    return x


def build_sod_model(input_shape: Tuple[int, int, int] = (128, 128, 3),
                    base_filters: int = 32,
                    depth: int = 3,
                    batch_norm: bool = True,
                    dropout_rate: float = 0.1) -> Model:
    inputs = layers.Input(shape=input_shape)
    skips = []
    x = inputs
    for d in range(depth):
        filters = base_filters * (2 ** d)
        x = _conv_block(x, filters, batch_norm=batch_norm)
        skips.append(x)
        x = layers.MaxPooling2D((2, 2))(x)

    filters = base_filters * (2 ** depth)
    x = _conv_block(x, filters, batch_norm=batch_norm)
    if dropout_rate and dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    for d in reversed(range(depth)):
        filters = base_filters * (2 ** d)
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        skip = skips[d]
        if skip.shape[1] != x.shape[1] or skip.shape[2] != x.shape[2]:
            skip = layers.Resizing(x.shape[1], x.shape[2])(skip)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, filters, batch_norm=batch_norm)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same", name="saliency_mask")(x)
    model = Model(inputs=inputs, outputs=outputs, name="SOD_EncoderDecoder")
    return model


def get_compiled_model(input_shape=(128, 128, 3),
                    base_filters=32,
                    depth=3,
                    batch_norm=True,
                    dropout_rate=0.1,
                    lr=1e-3):
    model = build_sod_model(input_shape=input_shape, base_filters=base_filters,
                            depth=depth, batch_norm=batch_norm, dropout_rate=dropout_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def iou_for_compile(y_true, y_pred):
        return iou_metric(y_true, y_pred, threshold=0.5)

    def prec_for_compile(y_true, y_pred):
        p, _, _ = precision_recall_f1(y_true, y_pred, threshold=0.5)
        return p

    def rec_for_compile(y_true, y_pred):
        _, r, _ = precision_recall_f1(y_true, y_pred, threshold=0.5)
        return r

    model.compile(optimizer=optimizer, loss=combined_bce_iou_loss,
                metrics=[iou_for_compile, prec_for_compile, rec_for_compile])
    return model


if __name__ == "__main__":
    m = get_compiled_model()
    m.summary()
