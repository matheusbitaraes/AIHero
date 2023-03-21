import mingus.core.chords as chords
import numpy as np
from mingus.core import notes

from src.EVO.resources.resources import note_reference
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER, CENTRAL_NOTE_NUMBER


class FitnessFunctionMap:
    def __init__(self):
        self.map = {
            "notes_on_same_chord_key": notes_on_same_chord_key,
            "notes_on_beat_rate": notes_on_beat_rate,
            "note_on_density": note_on_density,
            "note_variety_rate": note_variety_rate,
            "single_notes_rate": single_notes_rate,  # notas unicas, que não são triades ou duplas
            "notes_out_of_scale_rate": notes_out_of_scale_rate,
        }

    def eval(self, input_values):
        name = input_values["name"]
        return self.map[name](input_values)

    def keys(self):
        return self.map.keys()


def notes_on_same_chord_key(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]
    chord_notes = get_chord_notes(input_values)

    # function
    total_notes = len(note_sequence[note_sequence == 1])
    chord_note_lines = note_sequence[chord_notes, :].copy()
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
    translated_note_sequence[:, 1:] = note_sequence[:, 0: note_sequence.shape[1] - 1].copy()
    translated_note_sequence[:, 0] = note_sequence[:, -1].copy()
    note_sequence = note_sequence - translated_note_sequence - 1

    total_notes = len(note_sequence[note_sequence == 1])
    on_beat_positions = range(0, TIME_DIVISION, int(TIME_DIVISION / 4))
    columns_on_beat = note_sequence[:, on_beat_positions]
    notes_on_beat = len(columns_on_beat[columns_on_beat == 1])
    if total_notes != 0:
        return weight * notes_on_beat / total_notes
    else:
        return 0


def note_on_density(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]

    PERCENTAGE_BOUNDARIES = [0.2, 0.8]

    note_sums = np.sum(note_sequence, 0)  # sum over y axis

    total_note_on = len(note_sums[note_sums != -1 * SCALED_NOTES_NUMBER])
    if total_note_on != 0:
        note_on_percentage = total_note_on / TIME_DIVISION
        normalized_value = (note_on_percentage - PERCENTAGE_BOUNDARIES[0]) / (
                PERCENTAGE_BOUNDARIES[1] - PERCENTAGE_BOUNDARIES[0])
        normalized_perc = max(0, min(1, normalized_value))
        return weight * normalized_perc
    else:
        return 0


def note_variety_rate(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]

    PERCENTAGE_BOUNDARIES = [0.05, 0.15]

    note_sums = np.sum(note_sequence, 1)  # sum over x axis

    total_rows_with_notes = len(note_sums[note_sums != -1 * TIME_DIVISION])
    if total_rows_with_notes != 0:
        note_variety_percentage = total_rows_with_notes / SCALED_NOTES_NUMBER
        normalized_value = (note_variety_percentage - PERCENTAGE_BOUNDARIES[0]) / (
                PERCENTAGE_BOUNDARIES[1] - PERCENTAGE_BOUNDARIES[0])
        normalized_perc = max(0, min(1, normalized_value))
        return weight * normalized_perc
    else:
        return 0


def single_notes_rate(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]

    PERCENTAGE_BOUNDARIES = [0.3, 1]

    note_sums = np.sum(note_sequence, 0)  # sum over y axis

    total_note_on = len(note_sums[note_sums != -1 * SCALED_NOTES_NUMBER])
    total_single_notes = len(note_sums[note_sums == -1 * (SCALED_NOTES_NUMBER - 2)])
    if total_single_notes != 0:
        note_on_percentage = total_single_notes / total_note_on
        normalized_value = (note_on_percentage - PERCENTAGE_BOUNDARIES[0]) / (
                PERCENTAGE_BOUNDARIES[1] - PERCENTAGE_BOUNDARIES[0])
        normalized_perc = max(0, min(1, normalized_value))
        return weight * normalized_perc
    else:
        return 0

    return 0


def pitch_proximity_rate(input_values):
    note_center = input_values["value"] - CENTRAL_NOTE_NUMBER + SCALED_NOTES_NUMBER/2
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]
    note_range = 10
    note_sums = np.sum(note_sequence, 1)  # sum over x axis
    total_rows_with_notes = len(note_sums[note_sums != -1 * TIME_DIVISION])
    a = max(0, int(note_center-note_range))
    b = min(SCALED_NOTES_NUMBER, int(note_center+note_range))
    range_filtered = note_sums[a:b]
    notes_inside_range = len(range_filtered[range_filtered != -1 * TIME_DIVISION])

    if total_rows_with_notes == 0:
        return 0
    return weight * notes_inside_range/total_rows_with_notes


def notes_out_of_scale_rate(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]

    return 0


def note_repetitions_rate(input_values):
    return 0


def note_sequence_rate(input_values):
    return 0


def get_chord_notes(melody_specs):
    chord_number = melody_specs["chord"]
    chord = 'C'
    for note in note_reference.keys():
        if note_reference[note] == abs(int(chord_number)):
            chord = note
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
