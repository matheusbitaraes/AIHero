# class that contains all functions for encoding and decoding dataset functions
from mingus.midi import midi_file_in, fluidsynth
import mingus.core.notes as notes
from mingus.containers import Bar, Note, Track, Composition
from src.resources import *
import numpy as np


def convert_name_into_number(name):
    return note_reference[name]


def convert_notes(bar):
    # bar = [current beat, duration, notes]
    converted_notes = np.zeros(32) - 1
    key_number = get_key_number(bar)  # para adequar bar ao key
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
                converted_notes[i] = midi_note

    return converted_notes


def get_key_number(bar):
    return note_reference[bar.key.key]


def convert_bars(bars):
    converted_bars = []
    for bar in bars:
        converted_bar = convert_notes(bar)
        converted_bars.append(converted_bar)

    return converted_bars


def convert_tracks(tracks):
    converted_tracks = []
    for track in tracks:
        bars = track.bars
        converted_tracks.append(convert_bars(bars))
    return converted_tracks


def convert_compositions_indo_data(dataset_composition_list):
    datasets = []
    for composition_tuple in dataset_composition_list:
        tracks = composition_tuple[0].tracks
        datasets.append(convert_tracks(tracks))
    return datasets


# def convert_data_into_notes(data):
#     note = Note()
#     return notes
# def convert_data_into_bars(data):
# return bars

def get_octave_and_note(note):
    octave = int(note / 12) - 1
    note_int = int(note % 12)
    return octave, note_int


def convert_data_into_bar(data):
    b = Bar()
    b.key.name = 'C'  # por enquanto sempre em C
    idx = 0
    while idx < len(data):
        note_duration = 1
        while idx < len(data) - 1 and data[idx] == data[idx + 1]:
            note_duration = note_duration + 1
            idx = idx + 1
        if data[idx] != -1:
            octave, note_int = get_octave_and_note(data[idx])
            new_note = Note(notes.int_to_note(note_int), octave=octave, velocity=50)
            b.place_notes(new_note, 32 / note_duration)
        else:
            b.place_rest(32 / note_duration)
        idx = idx + 1
    return b


def convert_data_into_track(data):
    t = Track()
    for bar_data in data:
        t.add_bar(convert_data_into_bar(bar_data))
    return t


def convert_data_into_composition(data):
    c = Composition()
    for track_data in data:
        c.add_track(convert_data_into_track(track_data))
    return c


def convert_data_into_compositions(data):
    dataset_composition_list = []
    for composition_tuple in data:
        dataset_composition_list.append(convert_data_into_composition(composition_tuple))
    return dataset_composition_list


class AIHeroData:
    def __init__(self, data=None, mingus_composition_data=None):
        self.__data = data
        self.__mingus_composition_data = mingus_composition_data

    def load_from_midi_files(self, train_files):
        # convert midi data into the used class
        compositions = []
        for file in train_files:
            try:
                compositions.append(midi_file_in.MIDI_to_Composition(file))
            except:
                print("error converting MIDI into mingus file!")

        self.__mingus_composition_data = compositions
        self.__data = convert_compositions_indo_data(compositions)

    def get_data(self):
        return self.__data

    def get_composition(self):
        return self.__mingus_composition_data

    def set_data(self, data):
        self.__data = data
        self.__mingus_composition_data = convert_data_into_compositions(data)

    def set_compositions(self, compositions):
        self.__data = convert_compositions_indo_data(compositions)
        self.__mingus_composition_data = compositions
