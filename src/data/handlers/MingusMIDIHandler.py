import numpy as np
from mingus.containers import Composition, Track, Bar, NoteContainer, Note
from mingus.core import notes
from mingus.midi import midi_file_in, midi_file_out

from src.EVO.resources.resources import note_reference
from src.data.handlers.MIDIHandler import MIDIHandlerInterface
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER, CENTRAL_NOTE_NUMBER, SCALED_NOTES_RANGE, \
    MIDI_NOTES_NUMBER


class MingusMIDIHandler(MIDIHandlerInterface):

    def load_from_midi_file(self, file):
        composition = midi_file_in.MIDI_to_Composition(file)
        chord = get_chord_from_filename(file)
        composition = add_chord_to_composition(composition, chord)
        spr = get_as_spr(composition)
        chord_array = np.repeat(chord, TIME_DIVISION)
        chords = np.repeat(chord_array, spr.shape[0]).reshape((spr.shape[0], TIME_DIVISION))
        return spr, chords

    def export_as_midi(self, spr, chord_array, file_name="teste"):
        composition = revert_spr_into_composition(spr, chord_array)
        midi_file_out.write_Composition(f"{file_name}.mid", composition[0])

    def append_track_from_file(self, spr, chords, midi_file):
        composition = revert_spr_into_composition(spr, chords)
        base_composition = midi_file_in.MIDI_to_Composition(midi_file)

        base_track = base_composition[0].tracks[0]
        composition[0].add_track(base_track)
        return get_as_spr(composition), chords

    def append_track_and_export_as_midi(self, spr, chord_array, midi_file, file_name="teste"):
        composition = revert_spr_into_composition(spr, chord_array)
        base_composition = midi_file_in.MIDI_to_Composition(midi_file)

        base_track = base_composition[0].tracks[0]
        composition[0].add_track(base_track)
        midi_file_out.write_Composition(f"{file_name}.mid", composition[0])


def get_as_spr(composition):
    i = 0
    data_list = []
    for track in composition[0].tracks:
        for bar in track:
            data_list.append(convert_bar_to_spr(bar))
            i = i + 1
    return np.reshape(data_list, (i, SCALED_NOTES_NUMBER, TIME_DIVISION, 1))


def add_chord_to_composition(mingus_composition, chord):
    composition = mingus_composition[0]
    if isinstance(chord, str):
        for track in composition.tracks:
            for bar_list in track.bars:
                bar_list.chord = chord
                bar_list.key.key = chord
    else:
        for track in composition.tracks:
            i = 0
            for bar_list in track.bars:
                bar_list.chord = chord[i, 0]
                bar_list.key.key = chord[i, 0]
                i += 1
    return mingus_composition


def revert_spr_into_composition(spr, chord_array):
    compositions_converted = []
    max_key_id = chord_array.shape[0]
    key_id = 0
    c = Composition()
    t = Track()
    for matrix in spr:
        b = Bar()
        b.key.name = chord_array[key_id, 0]
        i = 0
        while i < matrix.shape[1]:
            b.place_rest(TIME_DIVISION)
            notes_matrix = np.where(matrix[:, i] > 0)[0]
            there_is_note = False
            note_duration = 1
            n = NoteContainer()
            n.empty()
            for note in notes_matrix:
                octave, note_int = revert_spr_note(note, chord_array[key_id, i])
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
    return c, 120


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
                    end_at = min(begin_at + num_time_steps, TIME_DIVISION * max_size)
                    if SCALED_NOTES_RANGE[1] > midi_note >= SCALED_NOTES_RANGE[0]:
                        converted_notes[midi_note, range(begin_at, end_at)] = 1
        previous_position += bar.current_beat
    converted_track = []
    for i in range(0, TIME_DIVISION * max_size, TIME_DIVISION):
        bar = converted_notes[:, i:i + TIME_DIVISION]
        if np.max(bar) != -1:
            converted_track.append(bar)
    return converted_track
