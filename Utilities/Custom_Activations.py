import tensorflow as tf


def split_relu(x):
    y1 = tf.nn.relu(x)
    y2 = tf.nn.relu(-x)
    y = tf.concat([y1, y2], axis=-1)
    return y
