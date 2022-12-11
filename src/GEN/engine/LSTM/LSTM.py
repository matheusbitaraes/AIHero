import os

import matplotlib
import numpy as np

from src.GEN.engine.AIHeroGEN import AIHeroGEN
from src.utils.AIHeroGlobals import SCALED_NOTES_CLASSES, TIME_DIVISION, SCALED_NOTES_NUMBER

matplotlib.use('Agg')
import tensorflow as tf
from keras import layers

from src.utils.AIHeroHelper import HarmonicFunction


# https://www.youtube.com/watch?v=IrPhMM_RUmg
class LSTM(AIHeroGEN):
    def __init__(self, config: dict, harmonic_function: HarmonicFunction = HarmonicFunction.TONIC):
        super().__init__(config, harmonic_function)
        self._model_name = 'lstm'
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self._verbose = config["verbose"]

        self.model = self._build()

        # overwrite arguments
        self.checkpoint_dir = f'{config["checkpoint"]["checkpoint_folder"]}/{self._model_name}/{self.harmonic_function.name}'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        # Create a callback that saves the model's weights
        self._cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_prefix,
                                                               save_weights_only=True,
                                                               verbose=1)

        if self._should_use_checkpoint:
            self.load_from_checkpoint()

    def get_name(self):
        return self._model_name

    def load_from_checkpoint(self):
        if self._verbose:
            print(f"loading for {self._model_name} [{self.harmonic_function.name}]...")
        try:
            self.model.load_weights(self.checkpoint_prefix)
            there_is_a_checkpoint = True
        except:
            there_is_a_checkpoint = False

        if self.should_verbose():
            if there_is_a_checkpoint:
                print("Checkpoint loaded!")
            else:
                print("no checkpoint found")

    def generate_prediction(self, new_seed: bool = False, size: int = 1):
        seed = [48, 48, 48, 48, 48, 48, 48, 48]  # seed is a sequence of 8 "no note" states
        sequence_size = TIME_DIVISION
        steps = sequence_size - len(seed)

        seeds = np.repeat(seed, size).reshape([size, len(seed)])
        for _ in range(steps):
            # seed = seed[-max_sequence_length:]
            onehot_seeds = tf.keras.utils.to_categorical(seeds, num_classes=SCALED_NOTES_CLASSES)
            # onehot_seeds = onehot_seeds[np.newaxis, ...]

            # prediction
            probabilities = self.model.predict(onehot_seeds)

            output_int = self._sample_with_temperature(probabilities, 1)

            seeds = np.concatenate((seeds, output_int), axis=1)

        prediction = -1 * np.ones([size, SCALED_NOTES_NUMBER, sequence_size, 1])
        for i in range(seeds.shape[0]): # this method can be optimized
            for j in range(seeds.shape[1] - 1):
                if seeds[i, j] != 48:
                    prediction[i, int(seeds[i, j]), j, 0] = 1

        return prediction

    def train(self, should_generate_gif: bool = False, prefix: str = "", num_epochs: int = None):
        inputs, targets = self.training_data.get_as_LSTM_training_sequences()  # num_samples, midi_notes, time_step, ?

        batch_size = min(self.BATCH_SIZE, np.int(np.int(np.ceil(inputs.shape[0] * self.BATCH_PERCENTAGE))))
        self.model.fit(inputs,
                       targets,
                       epochs=self.num_epochs,
                       batch_size=batch_size,
                       verbose=self._verbose,
                       callbacks=[self._cp_callback]
                       )

        if self._should_use_checkpoint:
            self.model.save_weights(self.checkpoint_prefix)

        if self._should_use_checkpoint:
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def _build(self):
        model = tf.keras.models.Sequential()

        # seq_length, input_size. If shape is (None, SCALED_NOTES_NUMBER), we can use many time divisions
        model.add(tf.keras.Input(shape=(None, SCALED_NOTES_CLASSES)))

        # N, 128 (if return_sequences=true, will be N, length, 128. good for stacking multiple rnns)
        model.add(layers.LSTM(300))
        model.add(layers.Dropout(0.3))  # ver se precisa
        model.add(layers.Dense(SCALED_NOTES_CLASSES, activation="softmax"))

        model.compile(loss="sparse_categorical_crossentropy", optimizer=self._optimizer)

        if self._verbose:
            model.summary()

        return model

    def _sample_with_temperature(self, probabilities, temperature=1):
        # predictions = np.log(probabilities) / temperature
        # probabilities = np.exp(predictions) / np.sum(np.exp(predictions)) #todo: reimplement for many samples at once

        choices = range(probabilities.shape[1])
        indexes = np.zeros([probabilities.shape[0], 1])
        for i in range(probabilities.shape[0]):
            indexes[i, :] = int(np.random.choice(choices, p=probabilities[i]))

        return indexes
