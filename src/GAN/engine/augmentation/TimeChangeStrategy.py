import numpy as np

from utils.AIHeroGlobals import SCALED_NOTES_RANGE, NOTES_IN_OCTAVE, TIME_DIVISION, SCALED_NOTES_NUMBER


class TimeChangeStrategy:  # change duration of a random note, between a stablished note size
    # todo fazer o tempo aumentado cair dentro de multiplos do time_division/8
    def __init__(self):
        self._note_change_scale = int(TIME_DIVISION / 8)
        self.note_off_interval = range(0, TIME_DIVISION, self._note_change_scale)

    def apply(self, matrix):
        empty_note_row = TIME_DIVISION * -1
        rows_with_note = np.where(np.sum(matrix, 1) != empty_note_row)[0]
        row_id = np.random.choice(rows_with_note)
        self.change_note_time(matrix[row_id, :])
        return matrix

    def change_note_time(self, row):
        incr_or_decr = np.sign(np.random.rand() - 0.5)
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

        note_off_on_interval = incr_or_decr * (self.note_off_interval - note_off_id)
        only_positives = [i for i in note_off_on_interval if i > 0]
        if len(only_positives) > 0:
            new_note_id = np.where(note_off_on_interval == min(only_positives))[0][0]
            new_note_off_id = self.note_off_interval[new_note_id]
        else:
            new_note_off_id = note_off_id
        if note_on_id < new_note_off_id:  # only time changes and no not erasing
            row[note_on_id:note_off_id + 1] = -1
            row[note_on_id:new_note_off_id] = 1


def get_intervals(notes):
    notes_transposed = np.ones(notes.shape)
    notes_transposed[0:-1] = notes[1:]
    notes_transposed[-1] = notes[-1] + 1
    return notes - notes_transposed
