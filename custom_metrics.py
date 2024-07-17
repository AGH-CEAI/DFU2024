import tensorflow as tf
import keras


def dice(y_true, y_pred):
    c = tf.reduce_sum(y_true * y_pred)
    a = tf.reduce_sum(y_true)
    b = tf.reduce_sum(y_pred)

    dice = 2 * c / (a + b)
    return dice


def plus_jaccard_distance_loss(build_in_loss):
    def jaccard_sth(y_true, y_pred):
        jaccard_loss = jaccard_distance_loss(y_true, y_pred)
        sth_loss = build_in_loss(y_true, y_pred)
        return 0.8*jaccard_loss + 0.2*sth_loss
    return jaccard_sth


def jaccard_distance_loss(y_true, y_pred, smooth=1e-4):
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred))
    jac = intersection / (sum_ - intersection + smooth)
    return 1 - jac


def dice_coef(y_true, y_pred):
    smooth = 1.
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr
