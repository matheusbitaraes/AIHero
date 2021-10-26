from src.utils.AIHeroGlobals import TIME_DIVISION


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
        return pitch_proximity
    elif name == "note_sequence_rate":
        return note_sequence_rate
    else:
        return none_function


def notes_on_same_chord_key(input_values):
    # input values
    weight = input_values["weight"]
    note_sequence = input_values["note_sequence"]
    chord_notes = input_values["chord_notes"]

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

    total_notes = len(note_sequence[note_sequence == 1])
    on_beat_positions = range(0, TIME_DIVISION, int(TIME_DIVISION / 4))
    columns_on_beat = note_sequence[:, on_beat_positions]
    notes_on_beat = len(columns_on_beat[columns_on_beat == 1])
    if total_notes != 0:
        return weight * notes_on_beat / total_notes
    else:
        return 0


def intervals_percentage(input_values):
    return 0


def note_repetitions_rate(input_values):
    return 0


def pitch_proximity(input_values):
    return 0


def note_sequence_rate(input_values):
    return 0


def none_function(input_values):
    return 0
