# class that contains all functions for encoding and decoding dataset functions
import mingus.core.notes as notes
import numpy as np
from matplotlib import pyplot as plt
from mingus.containers import Bar, Note, Track, Composition
from mingus.midi import midi_file_in, midi_file_out

from src.EVO.resources import *

MIDI_NOTES_NUMBER = 108


class AIHeroData:
    def __init__(self, data=None, mingus_composition_data=None, piano_roll=None):
        self.__data = None
        self.__mingus_composition_list = None  # list of mingus Composition() object
        self.__piano_roll_data = None
        # composition_list -> track_list -> bar_list -> bar (note list)
        self.bpm = None

        if data is not None:
            self.set_data(data)

        if mingus_composition_data is not None:
            self.set_data(mingus_composition_data)

        if piano_roll is not None:
            self.set_data(piano_roll)

    def load_from_midi_files(self, train_files):
        # convert midi data into the used class
        compositions = []
        for file in train_files:
            try:
                compositions.append(midi_file_in.MIDI_to_Composition(file))
            except Exception as e:
                print(f"error converting MIDI into mingus file: {e}")
        self.set_mingus_composition(compositions)

    def load_from_GAN_melody_raw(self, bars):
        composition_list = []
        composition = []
        loaded_tracks = []
        loaded_bars = []
        for bar in bars:
            for i in range(bar.shape[0]):
                loaded_bars.append(bar[i, :, :, 0])
        loaded_tracks.append(loaded_bars)
        composition.append(loaded_tracks)
        composition_list.append(composition)
        self.set_piano_roll(composition_list)

    def get_data(self):
        return self.__data

    def get_piano_roll(self):
        return self.__piano_roll_data

    def get_mingus_composition(self):
        return self.__mingus_composition_list

    def set_data(self, data):
        self.__data = data
        self.__mingus_composition_list = self.convert_data(convert_into="mingus_composition")
        self.__piano_roll_data = self.convert_data(convert_into="piano_roll")

    def set_piano_roll(self, composition):
        self.__piano_roll_data = composition
        self.__data = self.convert_piano_roll(convert_into="data")
        self.__mingus_composition_list = self.convert_data(convert_into="mingus_composition")

    def set_mingus_composition(self, compositions):
        self.__mingus_composition_list = compositions
        self.__data = self.convert_mingus_composition(compositions, convert_into="data")
        self.__piano_roll_data = self.convert_data(convert_into="piano_roll")

    def get_piano_roll_as_matrix(self):
        i = 0
        data_list = []
        for composition in self.get_piano_roll():
            for tracks in composition:
                for bars in tracks:
                    for bar in bars:
                        data_list.append(bar)
                        i = i + 1
        return np.reshape(data_list, (i, MIDI_NOTES_NUMBER, 32, 1))

    def get_data_as_matrix(self):  # dim: [num_melody, num_fingers, num_fuses, 1]
        i = 0
        data_list = []
        for composition in self.get_data():
            for melody in composition:
                for compass in melody:
                    data = -1 * np.ones((8, 32, 1))
                    data[0, :, 0] = np.asmatrix(
                        compass)  # todo: melhorar isto para nao precisar fixar esse num_fingers como 8!!!
                    data_list.append(data)
                    i = i + 1
        return np.reshape(data_list, (i, 8, 32, 1))

    def convert_mingus_composition(self, composition_list, convert_into="data"):
        compositions = []
        for composition in composition_list:
            self.bpm = composition[1]
            converted_composition = []  # lista de tracks
            converted_tracks = []  # lista de Track
            for track in composition[0].tracks:
                # self.track_intrument.append(track.instrument) # todo fazer algo desse tipo
                converted_track = []
                for bar in track.bars:
                    if convert_into == "piano_roll":
                        converted_bar = None  # Implement when needed
                    else:
                        converted_bar = convert_bar_to_data(bar)
                    converted_track.append(converted_bar)
                converted_tracks.append(converted_track)
            converted_composition.append(converted_tracks)
            compositions.append(converted_composition)
        return compositions

    def convert_data(self, convert_into):
        data = self.get_data()
        compositions = []
        if convert_into == "mingus_composition":
            for composition in data:
                c = Composition()
                for tracks in composition:
                    for bars in tracks:
                        t = Track()
                        for bar in bars:
                            b = Bar()
                            b.key.name = 'C'  # por enquanto sempre em C
                            idx = 0
                            while idx < len(bar):
                                note_duration = 1
                                while idx < len(bar) - 1 and bar[idx] == bar[idx + 1]:
                                    note_duration = note_duration + 1
                                    idx = idx + 1
                                if bar[idx] != -1:
                                    octave, note_int = get_octave_and_note(bar[idx])
                                    new_note = Note(notes.int_to_note(note_int), octave=octave, velocity=90)
                                    b.place_notes(new_note, 32 / note_duration)
                                else:
                                    b.place_rest(32 / note_duration)
                                idx = idx + 1
                            t.add_bar(b)
                        c.add_track(t)
                composition_tuple = (c, self.bpm)
                compositions.append(composition_tuple)

        if convert_into == "piano_roll":
            for composition in data:
                composition_converted = []
                for tracks in composition:
                    tracks_converted = []
                    for bars in tracks:
                        bars_converted = []
                        for bar in bars:
                            piano_roll = -1 * np.ones((MIDI_NOTES_NUMBER, len(bar)))
                            for i in range(0, len(bar)):
                                if bar[i] != -1:
                                    piano_roll[int(bar[i]), i] = 1
                            bars_converted.append(piano_roll)
                        tracks_converted.append(bars_converted)
                    composition_converted.append(tracks_converted)
                compositions.append(composition_converted)
        return compositions

    def convert_piano_roll(self, convert_into):
        if convert_into == "data":
            compositions_converted = []
            for composition in self.get_piano_roll():
                composition_converted = []
                for tracks in composition:
                    tracks_converted = []
                    for bars in tracks:
                        bars_converted = []
                        for bar in bars:
                            data = -1 * np.ones(32)
                            id_matrix = np.matrix(np.where(bar > 0))
                            for idx in range(id_matrix.shape[1]):
                                data[id_matrix[1, idx]] = id_matrix[0, idx]
                            bars_converted.append(data)
                        tracks_converted.append(bars_converted)
                    composition_converted.append(tracks_converted)
                compositions_converted.append(composition_converted)
            return compositions_converted

    def export_as_image(self, path='', file_name="teste", title="Piano Roll of Melody"):
        data = self.get_piano_roll_as_matrix()
        num_bars = data.shape[0]
        concat_data = np.ndarray((MIDI_NOTES_NUMBER, 32 * num_bars))
        a = 0
        b = 32
        for i in range(num_bars):
            concat_data[:, a:b] = data[i, :, :, 0]
            a = b
            b = b + 32
        plt.imshow(concat_data, cmap='Blues')
        plt.axis([0, num_bars * 32, 0, 100])  # necessary for inverting y axis
        plt.ylabel("MIDI Notes")
        plt.xlabel("Fuses")
        plt.title(title)
        plt.savefig(f'{file_name}.png', dpi=900)

    def export_as_midi(self, path='', file_name="teste"):
        for composition in self.get_mingus_composition():
            midi_file_out.write_Composition(f"{file_name}.mid", composition[0])

    def append_base_track(self, midi_file):
        base_composition = midi_file_in.MIDI_to_Composition(midi_file)
        base_track = base_composition[0].tracks[0]
        new_compositions = self.get_mingus_composition()
        for new_composition in new_compositions:
            new_composition[0].add_track(base_track)

        self.set_mingus_composition(new_compositions)


def convert_name_into_number(name):
    return note_reference[name]


def convert_bar_to_data(bar):
    # bar = [current beat, duration, notes]
    converted_notes = np.zeros(32) - 1
    key_number = note_reference[bar.key.key]  # para adequar bar ao key
    for note_container in bar.bar:
        notes = note_container[2]
        if notes:
            note = notes[0]  # considerando apenas a primeira nota do conjunto de notas (limitação do modelo)
            midi_note = convert_name_into_number(note.name) + (
                    note.octave + 1) * 12 - key_number  # vai transladar as notas para ficarem todas no mesmo key, que é C.
            current_beat = note_container[0]
            duration = note_container[1]
            begin_at = int(32 * current_beat)
            num_fuses = int(32 / duration)
            # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
            end_at = min(begin_at + num_fuses, 32)
            for i in range(begin_at, end_at):
                converted_notes[i] = int(midi_note)

    return converted_notes


def get_octave_and_note(note):
    octave = int(note / 12) - 1
    note_int = int(note % 12)
    return octave, note_int
