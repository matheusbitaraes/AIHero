import time

import numpy as np
from tensorflow import keras

from src.GEN.engine.augmentation.AugmentationEngine import AugmentationEngine
from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER, SCALED_NOTES_CLASSES
from src.utils.AIHeroHelper import HarmonicFunction


class GANTrainingData:
    def __init__(self, config, harmonic_function=HarmonicFunction.TONIC, data=None):
        if data is not None:
            self._ai_hero_data = data
        else:
            self._ai_hero_data = AIHeroData()
            file_directory = config["training"]["train_data_folder"]
            # self._ai_hero_data.load_from_midi_files(glob(f"{file_directory}/part_{melodic_part.name}_*"))
            if harmonic_function is None:
                self._ai_hero_data.load_spr_from_checkpoint(file_directory)
            else:
                self._ai_hero_data.load_spr_from_checkpoint(file_directory, harmonic_function.name)
            self._ai_hero_data.remove_blank_bars()

        # data augmentation
        augmentation_config = config["data_augmentation"]
        if augmentation_config["enabled"]:
            self.augmentation_engine = AugmentationEngine(
                augmentation_config["data_augmentation_strategy_pipeline"])
            if config["verbose"]:
                start = time.time()
                start_size = self._ai_hero_data.get_spr_as_matrix().shape[0]
                print("Augmenting dataset...")

            self.replicate(self._ai_hero_data.get_spr_as_matrix().shape[0] * augmentation_config["replication_factor"])
            self.augment()

            if config["verbose"]:
                end_size = self._ai_hero_data.get_spr_as_matrix().shape[0]
                print(f"dataset augmented from {start_size} samples to {end_size} samples in {time.time() - start}s")
        else:
            self.augmentation_engine = AugmentationEngine()

    def get_as_matrix(self, harmonic_function=None):
        # returns a matrix of dim [SAMPLES_SIZE, MIDI_NOTES, TIME_DIVISION, 1]
        return self._ai_hero_data.get_spr_as_matrix(harmonic_function)

    def get_as_single_matrix(self, harmonic_function=None):
        # returns a matrix of dim [MIDI_NOTES, TIME_DIVISION * SAMPLES_SIZE]
        matrix = self._ai_hero_data.get_spr_as_matrix(harmonic_function)
        single_matrix = np.concatenate((matrix[0, :, :, 0], matrix[1, :, :, 0]), axis=1)
        for i in range(2, matrix.shape[0]):
            single_matrix = np.concatenate((single_matrix, matrix[i, :, :, 0]), axis=1)

        return single_matrix

    def transform_single_matrix_into_integer_array(self, single_matrix):
        # get a single_matrix with dim [MIDI_NOTES, TIME_DIVISION * SAMPLES_SIZE] and
        # transform it into a vector of lenght [TIME_DIVISION * SAMPLES_SIZE] where each value is the position where MIDI_NOTES is > 0

        # Output vector with dimensions [TIME_DIVISION * SAMPLES_SIZE]
        int_array = np.zeros(single_matrix.shape[1])

        # Iterate over the columns of the input matrix
        for i in range(single_matrix.shape[1]):
            # Get the current column
            column = single_matrix[:, i]
            # Get the index of the first non-zero element in the column
            index = np.argmax(column)

            # Set the corresponding element in the output vector to the index
            if index == 0:
                int_array[i] = SCALED_NOTES_NUMBER  # mark "no note" as SCALED_NOTES_NUMBER position
            else:
                int_array[i] = index

        return int_array.astype(int)

    def get_as_LSTM_training_sequences(self, harmonic_function=None, sequence_window=int(TIME_DIVISION / 4)):
        matrix = self.get_as_single_matrix(harmonic_function)
        int_array = self.transform_single_matrix_into_integer_array(matrix)

        inputs = []
        targets = []
        for i in range(int_array.shape[0] - sequence_window):
            inputs.append(int_array[i:i + sequence_window])
            targets.append(int_array[i + sequence_window])

        vocab_size = SCALED_NOTES_CLASSES
        inputs = keras.utils.to_categorical(inputs, num_classes=vocab_size)
        targets = np.array(targets)

        return inputs, targets

    def augment(self):
        self._ai_hero_data.augment(self.augmentation_engine)

    def replicate(self, final_size):
        self._ai_hero_data.replicate(final_size)

    def print_on_terminal(self):
        self._ai_hero_data.print_on_terminal()
        pass

    @property
    def ai_hero_data(self):
        return self._ai_hero_data

    def get_distinct_harmonic_functions(self):
        return self._ai_hero_data.get_distinct_harmonic_functions()

    def get_spr_encoded_with_hm(self):
        return self._ai_hero_data.encode_spr_with_hm()
