import numpy as np
import tensorflow as tf
from tensorflow import keras


def check_not_nan(arrays):
    if isinstance(arrays, list):
        for i in range(len(arrays)):
            assert (not np.isnan(arrays[i].numpy()).any())
    else:
        assert(not np.isnan(arrays).any())


class Sequential_with_Gradient_Tape(keras.Sequential):
    def __init__(self):
        super(Sequential_with_Gradient_Tape, self).__init__()

    def train_step(self, data):
        x_train, y_train = data
        with tf.GradientTape() as tape:
            y_pred = self(x_train, training=True)
            loss = self.compiled_loss(y_train, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y_train, y_pred)
        if self.run_eagerly:
            check_not_nan(loss)
            check_not_nan(gradients)
            check_not_nan(self.trainable_variables)
        return {m.name: m.result() for m in self.metrics}