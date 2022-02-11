import numpy as np

from utils.AIHeroGlobals import TIME_DIVISION


class FifthNoteAddStrategy:  # add note five semitons below the not being played
    def __init__(self):
        fuse = TIME_DIVISION / 32
        self._note_change_scale = int(2 * fuse)
        self._time_lower_bound = int(1 * fuse)
        self._time_upper_bound = int(8 * fuse)

    def apply(self, matrix):
        empty_note_row = TIME_DIVISION * -1
        rows_with_note = np.where(np.sum(matrix, 1) != empty_note_row)[0]
        row_id = np.random.choice(rows_with_note)
        self.add_fifth_note(matrix[row_id, :], matrix[row_id - 5, :])
        return matrix

    def add_fifth_note(self, row, fifth):
        note_on = np.where(row == 1)[0]
        intervals = get_intervals(note_on)
        relevant_intervals = np.where(intervals < -1)[0]
        if len(relevant_intervals) > 0:  # there are more notes on row
            random_interval = np.random.choice(relevant_intervals)
            note_off_id = note_on[random_interval]
            note_on_id = note_on[0]
            for id in range(random_interval - 1, -1, -1):
                if intervals[id] != -1:
                    note_on_id = note_on[id + 1]
                    break
        else:  # there is only one note
            note_off_id = note_on[-1]
            note_on_id = note_on[0]
        fifth[note_on_id:note_off_id + 1] = 1


def get_intervals(notes):
    notes_transposed = np.ones(notes.shape)
    notes_transposed[0:-1] = notes[1:]
    notes_transposed[-1] = notes[-1] + 1
    return notes - notes_transposed
