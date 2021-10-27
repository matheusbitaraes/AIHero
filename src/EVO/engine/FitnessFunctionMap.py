import numpy as np
from mingus.core import notes

from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER
import mingus.core.chords as chords


class FitnessFunctionMap:
    def __init__(self, name):
        self.f = get_function_by_name(name)

    def eval(self, input_values):
        return self.f(input_values)


def get_function_by_name(name):
    if name == "notes_on_same_chord_key":
        return notes_on_same_chord_key
    elif name == "notes_on_beat_rate":
        return notes_on_beat_rate
    elif name == "intervals_percentage":
        return intervals_percentage
    elif name == "note_repetitions_rate":
        return note_repetitions_rate
    elif name == "pitch_proximity":
        return pitch_proximity_rate
    elif name == "note_sequence_rate":
        return note_sequence_rate
    else:
        return none_function


def notes_on_same_chord_key(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]
    chord_notes = get_chord_notes(input_values)

    # function
    total_notes = len(note_sequence[note_sequence == 1])
    chord_note_lines = note_sequence[chord_notes, :]
    notes_on_chord = len(chord_note_lines[chord_note_lines == 1])
    if total_notes != 0:
        return weight * notes_on_chord / total_notes
    else:
        return 0


def notes_on_beat_rate(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]

    # strategy for reducing notes to its fist value appearing on row. So, [-1, -1, 1, 1, 1, 1] will be transformed
    # into [-1, -1, 1, -1, -1, -1] because we are interested only in the beginning of the note
    translated_note_sequence = np.zeros(note_sequence.shape)
    translated_note_sequence[:, 1:] = note_sequence[:, 0: note_sequence.shape[1] - 1]
    translated_note_sequence[:, 0] = note_sequence[:, -1]
    note_sequence = note_sequence - translated_note_sequence - 1

    total_notes = len(note_sequence[note_sequence == 1])
    on_beat_positions = range(0, TIME_DIVISION, int(TIME_DIVISION / 4))
    columns_on_beat = note_sequence[:, on_beat_positions]
    notes_on_beat = len(columns_on_beat[columns_on_beat == 1])
    if total_notes != 0:
        return weight * notes_on_beat / total_notes
    else:
        return 0


def intervals_percentage(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]

    PERCENTAGE_BOUNDARIES = [0.2, 0.8]

    note_sums = np.sum(note_sequence, 0)  # sum over y axis

    total_intervals = len(note_sums[note_sums == -1 * SCALED_NOTES_NUMBER])
    if total_intervals != 0:
        interval_percentage = total_intervals/TIME_DIVISION
        normalized_perc = (interval_percentage - PERCENTAGE_BOUNDARIES[0]) /(PERCENTAGE_BOUNDARIES[1] - PERCENTAGE_BOUNDARIES[0])
        return weight * min(0, max(1, normalized_perc))
    else:
        return weight


def note_repetitions_rate(input_values):
    return 0


def pitch_proximity_rate(input_values):
    return 0


def note_sequence_rate(input_values):
    return 0


def none_function(input_values):
    return 0


def get_chord_notes(melody_specs):
    chord = melody_specs["chord"]
    key = melody_specs["key"]
    note_list = chords.triad(chord, key)
    note_numbers = []
    for n in note_list:
        note_int = notes.note_to_int(n)
        note = int(note_int + SCALED_NOTES_NUMBER / 2)
        note_numbers.append(note)
        note_numbers.append(note + 12)
        note_numbers.append(note - 12)
    return note_numbers
