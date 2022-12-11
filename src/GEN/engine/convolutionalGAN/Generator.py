from abc import ABC

import matplotlib
from keras import layers

matplotlib.use('Agg')
import tensorflow as tf

from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER


class Generator(ABC):
    def __init__(self, config: dict):
        self._verbose = config["verbose"]
        self.noise_dim = config["training"]["noise_dim"]

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.5)
        self.model = self.build()

    def build(self):
        layer_1_x = max(int(SCALED_NOTES_NUMBER / 2), 1)
        layer_1_y = max(int(TIME_DIVISION / 2), 1)
        layer_2_x = max(int(SCALED_NOTES_NUMBER / 4), 1)
        layer_2_y = max(int(TIME_DIVISION / 4), 1)
        model = tf.keras.Sequential()

        # https://analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/#:~:text=In%20any%20neural%20network%2C%20a,in%20artificial%20neural%20network%20networks.
        # Dense layer: Neurons of the layer that is deeply connected with its preceding layer which means the neurons
        # of the layer are connected to every neuron of its preceding layer.
        model.add(layers.Dense(layer_2_x * layer_2_y * 256, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())

        # Leaky Rectified Linear Unit (ReLU) layer: A leaky ReLU layer performs a threshold operation, where any
        # input value less than zero is multiplied by a fixed scalar.
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((layer_2_x, layer_2_y, 256)))
        assert model.output_shape == (None, layer_2_x, layer_2_y, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, layer_2_x, layer_2_y, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, layer_1_x, layer_1_y, 64)

        # model.add(layers.Dense(64))
        model.add(layers.LeakyReLU())
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, SCALED_NOTES_NUMBER, TIME_DIVISION, 1)

        if self._verbose:
            model.summary()

        return model

    def loss(self, fake_output):
        # computes cross entropy between fake output and best output (which is everything = 1)
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

