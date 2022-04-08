# class that contains all functions for encoding and decoding dataset functions
import os
import traceback
from glob import glob

import matplotlib
import numpy as np

from src.data.handlers.PrettyMIDIHandler import PrettyMIDIHandler
from src.utils.AIHeroEnums import get_harmonic_function_of_chord

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_RANGE, \
    SCALED_NOTES_NUMBER


class AIHeroData:
    def __init__(self):
        self._transposition_factor = None  # [N X t], N = num bars, t = time division, value = int
        self._harmonic_function = None  # [N], N = num bars, value = harmonic function (dominant, subdominant, tonic)
        self._spr_data = None  # [N X note_number X t]
        self._midi_handler = PrettyMIDIHandler()

    def load_from_pop909_dataset(self, dataset_path=""):
        midi_paths = [d for d in os.listdir(dataset_path)]
        for path in midi_paths:
            filename = dataset_path + path
            files = glob(f"{filename}/*.mid")
            for file in files:
                print(f"loading from file {file}...")
                data, chord_array = self._midi_handler.load_from_pop909_file(file)
                self.add_data(spr_data=data, chord_array=chord_array)

    def save_data(self, dir="", prefix=""):
        np.save(f"{dir}/{prefix}spr_data", self._spr_data)
        np.save(f"{dir}/{prefix}chord_data", self._transposition_factor)

    def load_spr_from_checkpoint(self, path=".", prefix=None):
        if prefix is None:
            sprs = glob(f"{path}/*spr_data.npy")
            chords = glob(f"{path}/*chord_data.npy")
            for i in range(len(sprs)):
                spr = np.load(sprs[i])
                chord = np.load(chords[i])
                self.add_data(spr_data=spr, chord_array=chord)

        else:
            spr = np.load(f"{path}/{prefix}spr_data.npy")
            chords = np.load(f"{path}/{prefix}chord_data.npy")
            self.add_data(spr_data=spr, chord_array=chords)

    def load_from_midi_files(self, files):
        for file in files:
            self._load_from_midi_file(file)

    def _load_from_midi_file(self, file):
        try:
            spr_data, chord_array = self._midi_handler.load_from_midi_file(file)
            self.add_data(spr_data, chord_array)
        except Exception as e:
            print(f"error converting MIDI into file: {e}")
            print(traceback.format_exc())

    def add_data(self, spr_data, chord_array):
        self._add_spr(spr_data)
        self._add_transposition_factor(chord_array)
        self._build_harmonic_function()

    def _add_spr(self, spr_data):
        if self._spr_data is None:
            self._spr_data = spr_data
        else:
            self._spr_data = np.concatenate((self._spr_data, spr_data), axis=0)

    def _add_transposition_factor(self, chord_array):
        if self._transposition_factor is None:
            self._transposition_factor = chord_array
        else:
            self._transposition_factor = np.concatenate((self._transposition_factor, chord_array), axis=0)

    def _add_harmonic_function(self, hm_array):
        if self._harmonic_function is None:
            self._harmonic_function = hm_array
        else:
            self._harmonic_function = np.concatenate((self._harmonic_function, hm_array), axis=0)

    def remove_blank_bars(self):
        old_spr = self._spr_data
        old_tf = self._transposition_factor
        old_hm = self._harmonic_function
        self._spr_data = None
        self._transposition_factor = None
        self._harmonic_function = None
        all_empty_number = TIME_DIVISION * SCALED_NOTES_NUMBER * -1
        not_empty_indexes = []
        for i in range(old_spr.shape[0]):
            note_sum_value = int(sum(sum(old_spr[i, :, :, 0])))
            if note_sum_value != all_empty_number:
                not_empty_indexes.append(i)
        self._add_spr(old_spr[not_empty_indexes, :, :, :])
        self._add_transposition_factor(old_tf[not_empty_indexes, :])
        self._add_harmonic_function(old_hm[not_empty_indexes])

    def load_from_EVO_melody_raw(self, bars):
        composition_list = []
        composition = []
        loaded_tracks = []
        loaded_bars = []
        for bar_tuple in bars:
            data = bar_tuple[0].reshape([1, SCALED_NOTES_NUMBER, TIME_DIVISION, 1])
            chord = np.repeat(bar_tuple[1], TIME_DIVISION).reshape([1, TIME_DIVISION])
            self.add_data(data, chord)
        loaded_tracks.append(loaded_bars)
        composition.append(loaded_tracks)
        composition_list.append(composition)

    def load_from_GAN_melody_raw(self, bars):
        for bar_tuple in bars:
            self._add_spr(bar_tuple[0])
            self._add_transposition_factor(np.repeat(int(bar_tuple[1]), TIME_DIVISION).reshape((1, TIME_DIVISION)))
        self._build_harmonic_function()

    def get_spr(self):
        return self._spr_data

    def get_spr_as_matrix(self, harmonic_function=None):
        if harmonic_function is None:
            return self._spr_data
        else:
            return self._spr_data[self._harmonic_function == int(harmonic_function), :, :, :]

    def _set_spr_matrix(self, matrix, chord_list):
        self._transposition_factor = chord_list
        compositions = []
        composition = []
        tracks = []
        bars = []
        for i in range(0, matrix.shape[0]):
            bars.append(matrix[i, :, :, 0])
        tracks.append(bars)
        composition.append(tracks)
        compositions.append(composition)
        self.set_spr(compositions)
        self._build_harmonic_function()

    def export_spr_as_image(self, path='', file_name="teste", title="Piano Roll of Melody"):
        data = self.get_spr_as_matrix()
        num_bars = data.shape[0]
        concat_data = np.ndarray((SCALED_NOTES_NUMBER, TIME_DIVISION * num_bars))
        a = 0
        b = TIME_DIVISION
        for i in range(num_bars):
            concat_data[:, a:b] = data[i, :, :, 0]
            a = b
            b = b + TIME_DIVISION
        plt.imshow(concat_data, cmap='Blues')
        plt.axis([0, num_bars * TIME_DIVISION, SCALED_NOTES_RANGE[0],
                  SCALED_NOTES_RANGE[1]])  # necessary for inverting y axis
        plt.ylabel("MIDI Notes")
        plt.xlabel("Time Division")
        plt.title(title)
        plt.savefig(f'{file_name}.png', dpi=900)

    def export_as_midi(self, file_name="teste"):
        try:
            self._midi_handler.export_as_midi(self._spr_data, self._transposition_factor, file_name)
        except Exception as e:
            print(f"Failed exporting as midi: {e}")
            print(traceback.format_exc())

    def append_track_and_export_as_midi(self, midi_file, file_name="teste"):
        try:
            self._midi_handler.append_track_and_export_as_midi(self._spr_data, self._transposition_factor, midi_file,
                                                               file_name)
        except Exception as e:
            print(f"Failed exporting as midi: {e}")
            print(traceback.format_exc())

    def append_track_from_file(self, midi_file):
        spr, chords = self._midi_handler.append_track_from_file(self._spr_data, self._transposition_factor, midi_file)
        self._spr_data = spr
        self._transposition_factor = chords
        self._build_harmonic_function()

    def split_into_train_test(self, train_test_ratio=.7):
        all_data = self.get_spr_as_matrix()
        chord_list = np.array(self._transposition_factor)
        size = all_data.shape[0]
        train_data_size = int(size * train_test_ratio)
        random_order = np.random.permutation(size)
        train_spr_data = all_data[random_order[0:train_data_size], :, :, :]
        train_chords = chord_list[random_order[0:train_data_size]]
        test_spr_data = all_data[random_order[train_data_size:], :, :, :]
        test_chords = chord_list[random_order[train_data_size:]]

        train_data = AIHeroData()
        test_data = AIHeroData()
        train_data.set_spr_matrix(train_spr_data, train_chords)
        test_data.set_spr_matrix(test_spr_data, test_chords)

        return train_data, test_data

    def augment(self, engine):
        augmented_data = engine.augment(self._spr_data)
        total_size = augmented_data.shape[0] / self._transposition_factor.shape[0]
        self._spr_data = augmented_data
        self._transposition_factor = np.repeat(self._transposition_factor, total_size, axis=0)
        self._build_harmonic_function()

    def replicate(self, final_size):
        num_repeat = np.round(final_size / self._spr_data.shape[0]) + 1
        repeated_data = np.repeat(self._spr_data, num_repeat, axis=0)
        repeated_chords = np.repeat(self._transposition_factor, num_repeat, axis=0)
        self._spr_data = repeated_data[0:final_size, :, :, :]
        self._transposition_factor = repeated_chords[0:final_size, :]
        self._build_harmonic_function()

    def execute_function_on_data(self, f):
        data, chords = f(self._spr_data, self._transposition_factor)
        self._spr_data = data
        self._transposition_factor = chords
        self._build_harmonic_function()

    def _build_harmonic_function(self):
        chord_array = self._transposition_factor
        hm = np.zeros(chord_array.shape[0])
        i = 0
        for chord in chord_array:
            numerical_value = int(round(np.mean(chord)))
            hm[i] = get_harmonic_function_of_chord(numerical_value)
            i += 1
        self._harmonic_function = hm

    def get_distinct_harmonic_functions(self):
        distinct = set(self._harmonic_function)
        return list(distinct)

    @property
    def harmonic_function(self):
        return self._harmonic_function

    def encode_spr_with_hm(self):
        data = np.zeros(self._spr_data.shape)
        hm = self._harmonic_function
        for i in range(self._spr_data.shape[0]):
            data[i, :, :, :] = self._spr_data[i, :, :, :] * hm[i]
        return data
