import math

import numpy as np
import pretty_midi as pyd
from pretty_midi import key_name_to_key_number

from src.data.handlers.MIDIHandler import MIDIHandlerInterface
# from src.data.handlers.MingusMIDIHandler import revert_spr_into_composition
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER, CENTRAL_NOTE_NUMBER, SCALED_NOTES_RANGE


class PrettyMIDIHandler(MIDIHandlerInterface):

    def load_from_midi_file(self, file, chords=None):
        composition = pyd.PrettyMIDI(file)
        if chords is None:
            chord = int(get_chords_from_filename(file))
            chord_array = np.repeat(chord, TIME_DIVISION)

            # todo: este codigo abaixo está duplicado. pode ser melhorado isso
            beat_indexes = composition.get_beats()
            beat_indexes_trans = beat_indexes[1:]
            slices = beat_indexes_trans - beat_indexes[:-1]
            num_slices = len(slices)
            chords = np.repeat(chord_array, num_slices).reshape((num_slices, TIME_DIVISION))

        return get_as_spr(composition, chords)

    def load_from_pop909_file(self, file):
        path = file.rsplit('/', 1)[0]
        data = pyd.PrettyMIDI(file)
        data = extract_melody(data)
        chord_midi = get_from_filename(path, 'chord_midi')
        key_audio = get_from_filename(path, 'key_audio')
        return get_pop909_as_spr(data, chord_midi, key_audio)

    def export_as_midi(self, spr, chord_array, file_name="teste"):
        composition = revert_spr_into_composition(spr, chord_array)
        composition.write(f"{file_name}.mid", )

    def append_track_and_export_as_midi(self, spr, chord_array, midi_file, file_name="teste"):
        composition = revert_spr_into_composition(spr, chord_array)
        base_composition = pyd.PrettyMIDI(midi_file, resolution=composition.resolution, initial_tempo=120)
        composition.instruments.append(base_composition.instruments[0])
        composition.write(f"{file_name}.mid")


def revert_spr_into_composition(spr, chords):
    piano_roll = spr_to_piano_roll(spr, chords)
    composition = piano_roll_to_pretty_midi(piano_roll, fs=int(TIME_DIVISION * 2 / 4))
    return composition


def spr_to_piano_roll(spr, chords):
    piano_roll = np.zeros([128, spr.shape[0] * TIME_DIVISION])
    for i in range(spr.shape[0]):
        for j in range(TIME_DIVISION):
            normalization_factor = int((SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]) / 2)
            chord_normalization = int(chords[i, j])
            a = CENTRAL_NOTE_NUMBER - normalization_factor - chord_normalization
            b = CENTRAL_NOTE_NUMBER + normalization_factor - chord_normalization
            piano_roll[a:b, i * TIME_DIVISION + j] = spr[i, :, j, 0]
    piano_roll[piano_roll == -1] = 0  # regularization of empty notes
    piano_roll[piano_roll == 1] = 100  # inserting constant velocity on every note
    return piano_roll


def piano_roll_to_pretty_midi(piano_roll, fs=100,
                              program=0):  # copiado de https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pyd.PrettyMIDI()
    instrument = pyd.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pyd.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def transpose_chord_number(chord_number, key_number):
    diff = chord_number - key_number
    if diff < 0:
        diff += 12
    return diff


def get_key_number(key):
    key_number = key_name_to_key_number(key[2].replace(":", " "))  # 0-11: major 12-23: minor
    if key_number >= 12:  # minor
        key_number -= 9
        if key_number >= 12:
            key_number -= 12
    return key_number


def get_pop909_as_spr(composition, chord_midi, key_audio):
    fs = 1000  # 16
    beat_indexes = composition.get_beats() * fs
    beat_indexes_trans = beat_indexes[1:]
    slices = beat_indexes_trans - beat_indexes[:-1]
    num_slices = len(slices)
    piano_roll = composition.get_piano_roll(fs=fs,
                                            pedal_threshold=None)  # todo: quando tem uma nota prolongada ants, a proxima nota (se for a mesma), some.
    pr = np.zeros([128, int(round(TIME_DIVISION * num_slices * 4 / 4))])
    chord_list = np.zeros([pr.shape[1]])
    for i in range(num_slices):

        # pega o time signature dessa parte
        num_of_divisions = get_num_of_divisions(composition, beat_indexes_trans[i])

        # muda escala do piano_roll para minha divisão de tempo
        downsampling_rate = round(slices[i] / num_of_divisions)
        id_begin = int(np.floor(beat_indexes[i]))
        id_end = int(np.ceil(beat_indexes_trans[i]))
        pr_beat = piano_roll[:, id_begin:id_end]
        max_y = pr_beat.shape[1] - 1
        j = 0

        # pega o tom
        key = get_key(key_audio, id_begin, id_end, fs)
        key_number = get_key_number(key)
        while j < num_of_divisions:
            a = j * downsampling_rate
            b = min(int(a + downsampling_rate), max_y)

            if key_number != 0:
                pr[:-key_number, num_of_divisions * i + j] = np.sum(pr_beat[key_number:, a:b],
                                                                    axis=1)  # armazena já convertido para key C
            else:
                pr[:, num_of_divisions * i + j] = np.sum(pr_beat[key_number:, a:b],
                                                         axis=1)  # armazena já convertido para key C
            j += 1

        # Armazena acorde transposto
        chord = get_chord(chord_midi, id_begin, id_end, fs)
        chord_number = key_name_to_key_number(chord[2].split(":")[0])
        translated_chord_number = transpose_chord_number(chord_number, key_number)
        chord_list[num_of_divisions * i: num_of_divisions * i + j] = translated_chord_number

        # print(f"key: {key[2]}[{key_number}] | {chord[2]}[{chord_number}] -> key: {key_number_to_key_name(0)}[0] |"
        #       f" {key_number_to_key_name(translated_chord_number)}[{translated_chord_number}]")

    spr, chord_array = scale_piano_roll(pr, chord_list)

    return spr, chord_array


def get_as_spr(composition, chord_array):
    fs = 1000  # 16
    beat_indexes = composition.get_beats() * fs
    beat_indexes_trans = beat_indexes[1:]
    slices = beat_indexes_trans - beat_indexes[:-1]
    num_slices = len(slices)
    piano_roll = composition.get_piano_roll(fs=fs,
                                            pedal_threshold=None)  # todo: quando tem uma nota prolongada ants, a proxima nota (se for a mesma), some.
    pr = np.zeros([128, int(round(TIME_DIVISION * num_slices))])
    chord_list = np.zeros([pr.shape[1]])
    for i in range(num_slices):

        # pega o time signature dessa parte
        num_of_divisions = get_num_of_divisions(composition, beat_indexes_trans[i])

        # muda escala do piano_roll para minha divisão de tempo
        downsampling_rate = round(slices[i] / num_of_divisions)
        id_begin = int(np.floor(beat_indexes[i]))
        id_end = int(np.ceil(beat_indexes_trans[i]))
        pr_beat = piano_roll[:, id_begin:id_end]
        max_y = pr_beat.shape[1] - 1
        j = 0

        # colocar o tom como C
        key_number = 0
        while j < num_of_divisions:
            a = j * downsampling_rate
            b = min(int(a + downsampling_rate), max_y)

            if key_number != 0:
                pr[:-key_number, num_of_divisions * i + j] = np.sum(pr_beat[key_number:, a:b],
                                                                    axis=1)  # armazena já convertido para key C
            else:
                pr[:, num_of_divisions * i + j] = np.sum(pr_beat[key_number:, a:b],
                                                         axis=1)  # armazena já convertido para key C
            j += 1

        # Armazena acorde transposto
        if chord_array.shape[0] > i:
            chord_value = chord_array[i, j - 1]
        else:
            chord_value = 0
        chord_list[num_of_divisions * i: num_of_divisions * i + j] = chord_value

        # print(f"key: {key[2]}[{key_number}] | {chord[2]}[{chord_number}] -> key: {key_number_to_key_name(0)}[0] |"
        #       f" {key_number_to_key_name(translated_chord_number)}[{translated_chord_number}]")

    spr, chord_array = scale_piano_roll(pr, chord_list)

    return spr, chord_array


def get_num_of_divisions(composition, current_time):
    # converte os time signatures em 4/4 e retorna a divisão a ser usada
    time_signature = get_time_signature(composition, current_time)
    factor = 4 * time_signature.denominator / time_signature.numerator
    num_of_divisions = int(TIME_DIVISION / factor)
    return num_of_divisions


def get_time_signature(composition, current_time):
    for ts in composition.time_signature_changes:
        if ts.time < current_time:
            return ts


def get_key(key_audio, id_begin, id_end, fs):
    filtered_keys = [key for key in key_audio if is_inside_slice(key, id_begin, id_end, fs)]
    if len(filtered_keys) > 0:
        return filtered_keys[0]
    else:
        return get_closest_key(key_audio, id_begin, id_end, fs)


def get_chord(chord_midi, id_begin, id_end, fs):
    filtered_chords = [chord for chord in chord_midi if is_inside_slice(chord, id_begin, id_end, fs)]
    if len(filtered_chords) > 0 and filtered_chords[0][2] != 'N':
        return filtered_chords[0]
    else:
        return get_closest_chord(chord_midi, id_begin, id_end, fs)


def get_name_from_chord_midi(chord_midi):
    if chord_midi == None:
        return None
    return chord_midi[2].split(":")[0]


def is_inside_slice(chord, i_a, i_b, fs):
    cond1 = int(float(chord[0]) * fs) <= i_a
    cond2 = int(float(chord[1]) * fs) >= i_b
    return cond1 and cond2


def get_closest_chord(chord_audio, i_a, i_b, fs):
    best = {
        "chord": None,
        "diff": math.inf
    }
    center = (i_b + i_a) / 2
    for chord in chord_audio:
        chord_center = fs * (float(chord[0]) + float(chord[1])) / 2
        diff = abs(chord_center - center)
        if diff < best["diff"] and chord[2] != 'N':
            best["chord"] = chord
            best["diff"] = diff
    return best["chord"]


def get_closest_key(key_audio, i_a, i_b, fs):
    best = {
        "key": None,
        "diff": math.inf
    }
    center = (i_b + i_a) / 2
    for key in key_audio:
        key_center = fs * (float(key[0]) + float(key[1])) / 2
        diff = abs(key_center - center)
        if diff < best["diff"]:
            best["key"] = key
            best["diff"] = diff
    return best["key"]


def get_chord_array(key_changes):
    pass


def scale_piano_roll(piano_roll_data, chord_transpose_array):
    num_bars = int(piano_roll_data.shape[1] / TIME_DIVISION)
    spr = np.zeros([num_bars, SCALED_NOTES_NUMBER, TIME_DIVISION, 1])
    chord_array = np.zeros([num_bars, TIME_DIVISION])
    central_note = CENTRAL_NOTE_NUMBER
    normalization_factor = int((SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]) / 2)
    i = 0
    while i < num_bars:
        for j in range(TIME_DIVISION):
            # x_a = int(i*TIME_DIVISION)
            idx = int(i * TIME_DIVISION + j)
            transpose_amount = int(chord_transpose_array[idx])
            y_a = central_note - normalization_factor - transpose_amount
            y_b = central_note + normalization_factor - transpose_amount
            spr[i, :, j, 0] = piano_roll_data[y_a:y_b, idx]
            chord_array[i, j] = transpose_amount
        i += 1
    spr[spr > 0] = 1
    spr[spr <= 0] = -1
    return spr, chord_array


def extract_melody(entire_data):
    melody_data = pyd.PrettyMIDI()
    melody_data.instruments.append(entire_data.instruments[0])
    melody_data.key_signature_changes = entire_data.key_signature_changes
    melody_data.lyrics = entire_data.lyrics
    melody_data.resolution = entire_data.resolution
    melody_data.time_signature_changes = entire_data.time_signature_changes
    return melody_data


def get_from_filename(filename, param):
    contents = []
    with open(f'{filename}/{param}.txt') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            contents.append(line.split('\t'))
    return contents


def get_chords_from_filename(file):
    try:
        chord = file.replace(".mid", "").split("_")[-1]
        return chord
    except:
        print("WARNING: filename does not have chord information. COnsidering it is all in C")
        return 0


def get_chords(composition):
    pass

# def add_chord_to_composition(mingus_composition, chord):
#     composition = mingus_composition[0]
#     if isinstance(chord, str):
#         for track in composition.tracks:
#             for bar_list in track.bars:
#                 bar_list.chord = chord
#                 bar_list.key.key = chord
#     else:
#         for track in composition.tracks:
#             i = 0
#             for bar_list in track.bars:
#                 bar_list.chord = chord[i, 0]
#                 bar_list.key.key = chord[i, 0]
#                 i += 1
#     return mingus_composition
#
#
# def revert_spr_into_composition(spr, chord_array):
#     compositions_converted = []
#     max_key_id = chord_array.shape[0]
#     key_id = 0
#     c = Composition()
#     t = Track()
#     for matrix in spr:
#         b = Bar()
#         b.key.name = chord_array[key_id, 0]
#         i = 0
#         while i < matrix.shape[1]:
#             b.place_rest(TIME_DIVISION)
#             notes_matrix = np.where(matrix[:, i] > 0)[0]
#             there_is_note = False
#             note_duration = 1
#             n = NoteContainer()
#             n.empty()
#             for note in notes_matrix:
#                 octave, note_int = revert_spr_note(note, chord_array[key_id, i])
#                 # print(f"place note {note}, {note_int}, {notes.int_to_note(note_int)} in fuse {i}")
#                 new_note = Note(notes.int_to_note(note_int), octave=octave, velocity=90)
#                 n.add_notes(new_note)
#                 there_is_note = True
#
#             if there_is_note:
#                 b.remove_last_entry()
#                 b.place_notes(n, TIME_DIVISION)
#             i += 1
#         b = unite_notes(b)
#         t.add_bar(b)
#         key_id += 1
#         if key_id == max_key_id:
#             key_id = 0
#     c.add_track(t)
#     return c, 120
#
#
# def convert_name_into_number(name):
#     return note_reference[name]
#
#
# def convert_bar_to_pr(bar):
#     # bar = [current beat, duration, notes]
#     converted_notes = np.zeros((MIDI_NOTES_NUMBER, TIME_DIVISION)) - 1
#     key_number = note_reference[bar.key.key]  # para adequar bar ao key
#     for note_container in bar.bar:
#         notes = note_container[2]
#         if notes:
#             for note in notes:
#                 midi_note = convert_name_into_number(note.name) + (
#                         note.octave + 1) * 12 - key_number  # vai transladar as notas para ficarem todas no mesmo key, que é C.
#                 current_beat = note_container[0]
#                 duration = note_container[1]
#                 begin_at = int(TIME_DIVISION * current_beat)
#                 num_time_steps = int(TIME_DIVISION / duration)
#                 # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
#                 end_at = min(begin_at + num_time_steps, TIME_DIVISION)
#                 converted_notes[midi_note, range(begin_at, end_at)] = 1
#
#     return converted_notes
#
#
# def convert_bar_to_spr(bar):  # bar = [current beat, duration, notes]
#     # bar = [current beat, duration, notes]
#     converted_notes = np.zeros((SCALED_NOTES_NUMBER, TIME_DIVISION)) - 1
#     key_number = note_reference[bar.key.key]  # para adequar bar ao key
#     for note_container in bar.bar:
#         notes = note_container[2]
#         if notes:
#             for note in notes:
#                 # change notes for key C and make C4(note 60) as the middle of the piano_roll matrix
#                 note_number = convert_name_into_number(note.name)
#                 note_octave_factor = 12 * (note.octave + 1)
#                 central_note = CENTRAL_NOTE_NUMBER
#                 normalization_factor = int((SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]) / 2)
#                 midi_note = note_number + note_octave_factor - key_number - central_note + normalization_factor
#                 current_beat = note_container[0]
#                 duration = note_container[1]
#                 begin_at = int(np.round(TIME_DIVISION * current_beat))
#                 num_time_steps = max(1, int(np.round(TIME_DIVISION / duration)))
#                 # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
#                 end_at = min(begin_at + num_time_steps, TIME_DIVISION)
#                 if SCALED_NOTES_RANGE[1] > midi_note >= SCALED_NOTES_RANGE[0]:
#                     converted_notes[midi_note, range(begin_at, end_at)] = 1
#     converted_notes[12, 0] = 1
#     return converted_notes
#
#
# def convert_bar_to_data(bar):
#     # bar = [current beat, duration, notes]
#     converted_notes = np.zeros(TIME_DIVISION) - 1
#     key_number = note_reference[bar.key.key]  # para adequar bar ao key
#     for note_container in bar.bar:
#         notes = note_container[2]
#         if notes:
#             note = notes[0]  # considerando apenas a primeira nota do conjunto de notas (limitação do modelo)
#             midi_note = convert_name_into_number(note.name) + (
#                     note.octave + 1) * 12 - key_number  # vai transladar as notas para ficarem todas no mesmo key, que é C.
#             current_beat = note_container[0]
#             duration = note_container[1]
#             begin_at = int(TIME_DIVISION * current_beat)
#             num_fuses = int(TIME_DIVISION / duration)
#             # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
#             end_at = min(begin_at + num_fuses, TIME_DIVISION)
#             for i in range(begin_at, end_at):
#                 converted_notes[i] = int(midi_note)
#
#     return converted_notes
#
#
# def revert_spr_note(note, chord):
#     normalization_factor = int((SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]) / 2)
#     chord_normalization = notes.note_to_int(chord)  # factor that will translate phrase according to the chord
#     pr_note = CENTRAL_NOTE_NUMBER + note - normalization_factor + chord_normalization
#     return get_octave_and_note(pr_note)
#
#
# def get_octave_and_note(note):
#     octave = int(note / 12) - 1
#     note_int = int(note % 12)
#     return octave, note_int
#
#
# def is_empty(bar):
#     return bar.min() == bar.max()
#
#
# def get_chord_from_filename(file):
#     return file.replace(".mid", "").split("_")[-1]
#
#
# def is_same_note(base_note, next_note):
#     return (base_note is None and next_note is None) or \
#            (base_note is not None and next_note is not None and base_note == next_note)
#
#
# def unite_notes(bar):
#     bars = bar.bar
#     output_bar = Bar()
#     note_index = 2
#     i = 0
#     while i < len(bars):
#         duration = 1
#         while i < len(bar) - 1 and is_same_note(bars[i][note_index], bars[i + 1][note_index]):
#             duration += 1
#             i += 1
#         note = bars[i][note_index]
#         # print(f"note: {note}, duration: {duration}")
#         if note:
#             output_bar.place_notes(note, TIME_DIVISION / duration)
#         else:
#             output_bar.place_rest(TIME_DIVISION / duration)
#         i += 1
#     return output_bar
#
#
# def convert_track_to_spr(track):
#     max_size = len(track.bars)
#     converted_notes = np.zeros((SCALED_NOTES_NUMBER, TIME_DIVISION * max_size)) - 1
#     previous_position = 0
#     for bar in track.bars:
#         # bar = [current beat, duration, notes]
#         key_number = note_reference[bar.key.key]  # para adequar bar ao key
#         for note_container in bar.bar:
#             notes = note_container[2]
#             if notes:
#                 for note in notes:
#                     # change notes for key C and make C4(note 60) as the middle of the piano_roll matrix
#                     note_number = convert_name_into_number(note.name)
#                     note_octave_factor = 12 * (note.octave + 1)
#                     central_note = CENTRAL_NOTE_NUMBER
#                     normalization_factor = int((SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]) / 2)
#                     midi_note = note_number + note_octave_factor - key_number - central_note + normalization_factor
#
#                     # get time position
#                     current_beat = previous_position + note_container[0]
#                     duration = note_container[1]
#                     begin_at = int(np.round(TIME_DIVISION * current_beat))
#                     num_time_steps = max(1, int(np.round(TIME_DIVISION / duration)))
#                     # print(f" begin: {begin_at}\n end: {begin_at + num_fuses - 1}\n bar: {bar.bar} \n container: {note_container}\n conv_notes:{converted_notes}")
#                     end_at = min(begin_at + num_time_steps, TIME_DIVISION * max_size)
#                     if SCALED_NOTES_RANGE[1] > midi_note >= SCALED_NOTES_RANGE[0]:
#                         converted_notes[midi_note, range(begin_at, end_at)] = 1
#         previous_position += bar.current_beat
#     converted_track = []
#     for i in range(0, TIME_DIVISION * max_size, TIME_DIVISION):
#         bar = converted_notes[:, i:i + TIME_DIVISION]
#         if np.max(bar) != -1:
#             converted_track.append(bar)
#     return converted_track
