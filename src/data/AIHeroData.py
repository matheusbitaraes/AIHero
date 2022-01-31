# class that contains all functions for encoding and decoding dataset functions
import traceback

import mingus.core.notes as notes
import numpy as np
from matplotlib import pyplot as plt
from mingus.containers import Bar, Note, Track, Composition, NoteContainer
from mingus.midi import midi_file_in, midi_file_out

from src.EVO.resources.resources import *
from src.utils.AIHeroGlobals import MIDI_NOTES_NUMBER, TIME_DIVISION, CENTRAL_NOTE_NUMBER, SCALED_NOTES_RANGE, \
    SCALED_NOTES_NUMBER


class AIHeroData:
    def __init__(self, data=None, mingus_composition_data=None, pr=None):
        # AIHeroData is a list of composition. Each composition has tracks. Each track has bars. Each bar has Notes
        self.chord_list = []
        # self._data = None
        self._mingus_composition_list = None  # list of mingus Composition() object
        # self._pr_data = None
        self._spr_data = None
        self.bpm = None

        # if data is not None:
        #     self.set_data(data)

        if mingus_composition_data is not None:
            self.set_data(mingus_composition_data)

        # if pr is not None:
        #     self.set_data(pr)

    def load_from_midi_files(self, train_files):
        # convert midi data into the used class
        compositions = []
        for file in train_files:
            try:
                composition = midi_file_in.MIDI_to_Composition(file)
                chord = get_chord_from_filename(file)
                compositions.append(self.add_chord_to_composition(composition, chord))
            except Exception as e:
                print(f"error converting MIDI into mingus file: {e}")
                print(traceback.format_exc())
        self.set_mingus_compositions(compositions)

    def load_from_EVO_melody_raw(self, bars):
        composition_list = []
        composition = []
        loaded_tracks = []
        loaded_bars = []
        for bar_tuple in bars:
            loaded_bars.append(bar_tuple[0])
            self.chord_list.append(bar_tuple[1])
        loaded_tracks.append(loaded_bars)
        composition.append(loaded_tracks)
        composition_list.append(composition)
        self.set_spr(composition_list)

    def load_from_GAN_melody_raw(self, bars):
        composition_list = []
        composition = []
        loaded_tracks = []
        loaded_bars = []
        for bar_tuple in bars:
            bar = bar_tuple[0]
            for i in range(bar.shape[0]):
                loaded_bars.append(bar[i, :, :, 0])
                self.chord_list.append(bar_tuple[1])
        loaded_tracks.append(loaded_bars)
        composition.append(loaded_tracks)
        composition_list.append(composition)
        self.set_spr(composition_list)

    def get_data(self):
        return self._data

    def get_pr(self):
        return self._pr_data

    def get_mingus_compositions(self):
        return self._mingus_composition_list

    def get_spr(self):
        return self._spr_data

    def get_spr_as_matrix(self):
        i = 0
        data_list = []
        for composition in self.get_spr():
            for tracks in composition:
                for bars in tracks:
                    for bar in bars:
                        data_list.append(bar)
                        i = i + 1
        return np.reshape(data_list, (i, SCALED_NOTES_NUMBER, TIME_DIVISION, 1))

    # def set_data(self, data):
    #     self._data = data
    #     self._mingus_composition_list = self.revert_data()
    #     self._pr_data = self.convert_mingus_composition(convert_into="pr")
    #     self._spr_data = self.convert_mingus_composition(convert_into="spr")

    # def set_pr(self, composition):
    #     # self._pr_data = composition
    #     self._mingus_composition_list = self.revert_pr()
    #     self._spr_data = self.convert_mingus_composition(convert_into="spr")
    #     self._data = self.convert_mingus_composition(convert_into="data")

    def set_mingus_compositions(self, compositions, chord_list=None):
        if chord_list is not None:
            self.chord_list = chord_list
        self._mingus_composition_list = compositions
        # self.export_as_midi()
        # self._pr_data = self.convert_mingus_composition(convert_into="pr")
        self._spr_data = self.convert_mingus_composition(convert_into="spr")
        # self._data = self.convert_mingus_composition(convert_into="data")

    def set_spr(self, compositions):
        self._spr_data = compositions
        self._mingus_composition_list = self.revert_spr()
        # self._pr_data = self.convert_mingus_composition(convert_into="pr")
        # self._data = self.convert_mingus_composition(convert_into="data")

    def set_spr_matrix(self, matrix, chord_list):
        self.chord_list = chord_list
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

    # def sanitize(self):
    #     pr_data = self.get_pr()
    #     clean_pr_data = []
    #     for composition in pr_data:
    #         clean_composition = []
    #         for tracks in composition:
    #             clean_tracks = []
    #             for bars in tracks:
    #                 clean_bars = []
    #                 for bar in bars:
    #                     if not is_empty(bar):
    #                         clean_bars.append(bar)
    #                 clean_tracks.append(clean_bars)
    #             clean_composition.append(clean_tracks)
    #         clean_pr_data.append(clean_composition)
    #     self.set_pr(clean_pr_data)

    def convert_mingus_composition(self, convert_into="data"):
        compositions = []
        for composition in self._mingus_composition_list:
            self.bpm = composition[1]
            converted_composition = []  # lista de tracks
            converted_tracks = []  # lista de Track
            for track in composition[0].tracks:
                # self.track_intrument.append(track.instrument) # todo fazer algo desse tipo
                i = 0
                if convert_into == "spr":
                    converted_track = convert_track_to_spr(track)
                elif convert_into == "pr":
                    converted_track = convert_track_to_pr(track)  # Implement when needed
                else:
                    converted_track = convert_track_to_data(track)  # Implement when needed
                converted_tracks.append(converted_track)
            converted_composition.append(converted_tracks)
            compositions.append(converted_composition)
        return compositions

    # def revert_data(self):
    #     data = self.get_data()
    #     compositions = []
    #     for composition in data:
    #         c = Composition()
    #         for tracks in composition:
    #             for bars in tracks:
    #                 t = Track()
    #                 for bar in bars:
    #                     b = Bar()
    #                     b.key.name = 'C'  # por enquanto sempre em C
    #                     idx = 0
    #                     while idx < len(bar):
    #                         note_duration = 1
    #                         while idx < len(bar) - 1 and bar[idx] == bar[idx + 1]:
    #                             note_duration = note_duration + 1
    #                             idx = idx + 1
    #                         if bar[idx] != -1:
    #                             octave, note_int = get_octave_and_note(bar[idx])
    #                             new_note = Note(notes.int_to_note(note_int), octave=octave, velocity=90)
    #                             b.place_notes(new_note, TIME_DIVISION / note_duration)
    #                         else:
    #                             b.place_rest(TIME_DIVISION / note_duration)
    #                         idx = idx + 1
    #                     t.add_bar(b)
    #                 c.add_track(t)
    #         composition_tuple = (c, self.bpm)
    #         compositions.append(composition_tuple)
    #     return compositions
    #
    # def revert_pr(self):
    #     compositions_converted = []
    #     key_id = 0
    #     for composition in self.get_pr():
    #         c = Composition()
    #         for tracks in composition:
    #             for bars in tracks:
    #                 t = Track()
    #                 for matrix in bars:
    #                     b = Bar()
    #                     # b.key.name = self.chord_list[key_id] todo: fazer isso
    #                     # key_id += 1
    #                     i = 0
    #                     while i < matrix.shape[1]:
    #                         notes_matrix = np.where(matrix[:, i] > 0)
    #                         there_is_note = False
    #                         note_duration = 1
    #                         n = NoteContainer()
    #                         n.empty()
    #                         for note in notes_matrix[0]:
    #                             octave, note_int = get_octave_and_note(note)
    #                             # print(f"place note {note}, {note_int}, {notes.int_to_note(note_int)} in fuse {i}")
    #                             new_note = Note(notes.int_to_note(note_int), octave=octave, velocity=90)
    #                             n.add_notes(new_note)
    #                             there_is_note = True
    #                         if there_is_note:
    #                             # print(f"place notes {n}, with duration {TIME_DIVISION / note_duration}")
    #                             b.place_notes(n, TIME_DIVISION / note_duration)
    #                         else:
    #                             # print(f"place rest, with duration {TIME_DIVISION / note_duration}")
    #                             b.place_rest(TIME_DIVISION / note_duration)
    #                         i += 1
    #                     b = unite_notes(b)
    #                     t.add_bar(b)
    #                 c.add_track(t)
    #         compositions_converted.append((c, self.bpm))
    #     return compositions_converted

    def revert_spr(self):
        compositions_converted = []
        max_key_id = len(self.chord_list)
        key_id = 0
        for composition in self.get_spr():
            c = Composition()
            for tracks in composition:
                for bars in tracks:
                    t = Track()
                    for matrix in bars:
                        b = Bar()
                        chord = self.chord_list[key_id]
                        b.key.name = chord
                        i = 0
                        while i < matrix.shape[1]:
                            b.place_rest(TIME_DIVISION)
                            notes_matrix = np.where(matrix[:, i] > 0)[0]
                            there_is_note = False
                            note_duration = 1
                            n = NoteContainer()
                            n.empty()
                            for note in notes_matrix:
                                octave, note_int = revert_spr_note(note, chord)
                                # print(f"place note {note}, {note_int}, {notes.int_to_note(note_int)} in fuse {i}")
                                new_note = Note(notes.int_to_note(note_int), octave=octave, velocity=90)
                                n.add_notes(new_note)
                                there_is_note = True

                            if there_is_note:
                                b.remove_last_entry()
                                b.place_notes(n, TIME_DIVISION)

                            i += 1
                        b = unite_notes(b)
                        t.add_bar(b)
                        key_id += 1
                        if key_id == max_key_id:
                            key_id = 0
                    c.add_track(t)
            compositions_converted.append((c, self.bpm))
        return compositions_converted

    # def export_pr_as_image(self, path='', file_name="teste", title="Piano Roll of Melody"):
    #     data = self.get_pr_as_matrix()
    #     num_bars = data.shape[0]
    #     concat_data = np.ndarray((MIDI_NOTES_NUMBER, TIME_DIVISION * num_bars))
    #     a = 0
    #     b = TIME_DIVISION
    #     for i in range(num_bars):
    #         concat_data[:, a:b] = data[i, :, :, 0]
    #         a = b
    #         b = b + TIME_DIVISION
    #     plt.imshow(concat_data, cmap='Blues')
    #     plt.axis([0, num_bars * TIME_DIVISION, 0, 100])  # necessary for inverting y axis
    #     plt.ylabel("MIDI Notes")
    #     plt.xlabel("Time Division")
    #     plt.title(title)
    #     plt.savefig(f'{file_name}.png', dpi=900)

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

    def export_as_midi(self, path='', file_name="teste"):
        i = 0
        for composition in self.get_mingus_compositions():
            i += 1
            if i == 1:
                midi_file_out.write_Composition(f"{file_name}.mid", composition[0])
            else:
                midi_file_out.write_Composition(f"{file_name}_{i}.mid", composition[0])

    def append_base_track(self, midi_file):
        base_composition = midi_file_in.MIDI_to_Composition(midi_file)
        base_track = base_composition[0].tracks[0]
        new_compositions = self.get_mingus_compositions()
        tempo = 80  # todo FAZER ISSO FUNCIONAR E SINCRONIZAR AS MELODIAS PARA O MESMO TEMPO
        for i in range(len(new_compositions)):
            new_compositions[i] = (new_compositions[i][0], tempo)
            new_compositions[i][0].add_track(base_track)

        self.set_mingus_compositions(new_compositions)
        self.revert_spr()

    def split_into_train_test(self, train_test_ratio=.7):
        all_data = self.get_spr_as_matrix()
        chord_list = np.array(self.chord_list)
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

    def add_chord_to_composition(self, mingus_composition, chord):
        composition = mingus_composition[0]
        for track in composition.tracks:
            for bar_list in track.bars:
                bar_list.chord = chord
                bar_list.key.key = chord
                self.chord_list.append(chord)
        return mingus_composition

    def augment(self, engine):
        augmented_data = engine.augment(self.get_spr_as_matrix())
        total_size = augmented_data.shape[0] / len(self.chord_list)
        self.set_spr_matrix(augmented_data, chord_list=np.tile(self.chord_list, int(total_size)))

    def replicate(self, final_size):
        data = self.get_spr_as_matrix()
        num_repeat = np.round(final_size / data.shape[0]) + 1
        repeated_data = np.repeat(self.get_spr_as_matrix(), num_repeat, axis=0)
        self.set_spr_matrix(repeated_data[0:final_size, :, :, :], chord_list=self.chord_list)

    def print_on_terminal(self):
        data = self.get_spr_as_matrix()
        # all_data = data.reshape([SCALED_NOTES_NUMBER, TIME_DIVISION * data.shape[0], data.shape[3]])
        for bar_id in range(data.shape[0]):
            print("\n")
            bar = data[bar_id, :, :, :]
            for i in range(bar.shape[0] - 1, 0, -1):
                row = bar[i, :, 0]
                str_row = ""
                for j in range(len(row)):
                    if j % TIME_DIVISION == 0:
                        str_row += "|"
                    if j % (TIME_DIVISION / 4) == 0:
                        str_row += "."
                    if row[j] == 1:
                        str_val = "-"
                    else:
                        str_val = " "
                    str_row += str_val
                print(f"{str_row}")
        pass


def convert_name_into_number(name):
    return note_reference[name]


def convert_bar_to_pr(bar):
    # bar = [current beat, duration, notes]
    converted_notes = np.zeros((MIDI_NOTES_NUMBER, TIME_DIVISION)) - 1
    key_number = note_reference[bar.key.key]  # para adequar bar ao key
    for note_container in bar.bar:
        notes = note_container[2]
        if notes:
            for note in notes:
                midi_note = convert_name_into_number(note.name) + (
                        note.octave + 1) * 12 - key_number  # vai transladar as notas para ficarem todas no mesmo key, que é C.
                current_beat = note_container[0]
                duration = note_container[1]
                begin_at = int(TIME_DIVISION * current_beat)
                num_time_steps = int(TIME_DIVISION / duration)
                # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
                end_at = min(begin_at + num_time_steps, TIME_DIVISION)
                converted_notes[midi_note, range(begin_at, end_at)] = 1

    return converted_notes


def convert_bar_to_spr(bar):  # bar = [current beat, duration, notes]
    # bar = [current beat, duration, notes]
    converted_notes = np.zeros((SCALED_NOTES_NUMBER, TIME_DIVISION)) - 1
    key_number = note_reference[bar.key.key]  # para adequar bar ao key
    for note_container in bar.bar:
        notes = note_container[2]
        if notes:
            for note in notes:
                # change notes for key C and make C4(note 60) as the middle of the piano_roll matrix
                note_number = convert_name_into_number(note.name)
                note_octave_factor = 12 * (note.octave + 1)
                central_note = CENTRAL_NOTE_NUMBER
                normalization_factor = int((SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]) / 2)
                midi_note = note_number + note_octave_factor - key_number - central_note + normalization_factor
                current_beat = note_container[0]
                duration = note_container[1]
                begin_at = int(np.round(TIME_DIVISION * current_beat))
                num_time_steps = max(1, int(np.round(TIME_DIVISION / duration)))
                # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
                end_at = min(begin_at + num_time_steps, TIME_DIVISION)
                if SCALED_NOTES_RANGE[1] > midi_note >= SCALED_NOTES_RANGE[0]:
                    converted_notes[midi_note, range(begin_at, end_at)] = 1
    converted_notes[12, 0] = 1
    return converted_notes


def convert_bar_to_data(bar):
    # bar = [current beat, duration, notes]
    converted_notes = np.zeros(TIME_DIVISION) - 1
    key_number = note_reference[bar.key.key]  # para adequar bar ao key
    for note_container in bar.bar:
        notes = note_container[2]
        if notes:
            note = notes[0]  # considerando apenas a primeira nota do conjunto de notas (limitação do modelo)
            midi_note = convert_name_into_number(note.name) + (
                    note.octave + 1) * 12 - key_number  # vai transladar as notas para ficarem todas no mesmo key, que é C.
            current_beat = note_container[0]
            duration = note_container[1]
            begin_at = int(TIME_DIVISION * current_beat)
            num_fuses = int(TIME_DIVISION / duration)
            # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
            end_at = min(begin_at + num_fuses, TIME_DIVISION)
            for i in range(begin_at, end_at):
                converted_notes[i] = int(midi_note)

    return converted_notes


def revert_spr_note(note, chord):
    normalization_factor = int((SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]) / 2)
    chord_normalization = notes.note_to_int(chord)  # factor that will translate phrase according to the chord
    pr_note = CENTRAL_NOTE_NUMBER + note - normalization_factor + chord_normalization
    return get_octave_and_note(pr_note)


def get_octave_and_note(note):
    octave = int(note / 12) - 1
    note_int = int(note % 12)
    return octave, note_int


def is_empty(bar):
    return bar.min() == bar.max()


def get_chord_from_filename(file):
    return file.replace(".mid", "").split("_")[-1]


def is_same_note(base_note, next_note):
    return (base_note is None and next_note is None) or \
           (base_note is not None and next_note is not None and base_note == next_note)


def unite_notes(bar):
    bars = bar.bar
    output_bar = Bar()
    note_index = 2
    i = 0
    while i < len(bars):
        duration = 1
        while i < len(bar) - 1 and is_same_note(bars[i][note_index], bars[i + 1][note_index]):
            duration += 1
            i += 1
        note = bars[i][note_index]
        # print(f"note: {note}, duration: {duration}")
        if note:
            output_bar.place_notes(note, TIME_DIVISION / duration)
        else:
            output_bar.place_rest(TIME_DIVISION / duration)
        i += 1
    return output_bar


def convert_mido_to_piano_roll(mido_data):
    composition = Composition()
    ticks_per_beat = mido_data.ticks_per_beat
    beats = 4
    total_ticks = ticks_per_beat * beats
    for track in mido_data.tracks:
        b = Bar()
        time = 0
        for bar in track:
            time += bar.time
            if not bar.is_meta:
                note = bar.note
                if bar.type == 'note_on':
                    print(f'note_on {note} on {time}, {time / total_ticks} ticks')
                elif bar.type == 'note_off':
                    print(f'note_off {note} on {time / total_ticks} ticks')

    return composition


def convert_track_to_spr(track):
    max_size = len(track.bars)
    converted_notes = np.zeros((SCALED_NOTES_NUMBER, TIME_DIVISION * max_size)) - 1
    previous_position = 0
    for bar in track.bars:
        # bar = [current beat, duration, notes]
        key_number = note_reference[bar.key.key]  # para adequar bar ao key
        for note_container in bar.bar:
            notes = note_container[2]
            if notes:
                for note in notes:
                    # change notes for key C and make C4(note 60) as the middle of the piano_roll matrix
                    note_number = convert_name_into_number(note.name)
                    note_octave_factor = 12 * (note.octave + 1)
                    central_note = CENTRAL_NOTE_NUMBER
                    normalization_factor = int((SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]) / 2)
                    midi_note = note_number + note_octave_factor - key_number - central_note + normalization_factor

                    # get time position
                    current_beat = previous_position + note_container[0]
                    duration = note_container[1]
                    begin_at = int(np.round(TIME_DIVISION * current_beat))
                    num_time_steps = max(1, int(np.round(TIME_DIVISION / duration)))
                    # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
                    end_at = min(begin_at + num_time_steps, TIME_DIVISION*max_size)
                    if SCALED_NOTES_RANGE[1] > midi_note >= SCALED_NOTES_RANGE[0]:
                        converted_notes[midi_note, range(begin_at, end_at)] = 1
        previous_position += bar.current_beat
    converted_track = []
    for i in range(0, TIME_DIVISION * max_size, TIME_DIVISION):
        bar = converted_notes[:, i:i+TIME_DIVISION]
        if np.max(bar) != -1:
            converted_track.append(bar)
    return converted_track


def convert_track_to_pr(track):
    return []


def convert_track_to_data(track):
    return []
