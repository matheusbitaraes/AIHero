import numpy as np

from src.utils.AIHeroGlobals import SCALED_NOTES_RANGE, NOTES_IN_OCTAVE, TIME_DIVISION, SCALED_NOTES_NUMBER


class NoteJoinStrategy:  # join same notes in sequence. If there is none, duplicate the matrix
    def apply(self, matrix):
        empty_note_row = TIME_DIVISION * -1
        projection = np.sum(matrix, 0)
        rows_with_note = np.where(np.sum(matrix, 1) != empty_note_row)[0]
        for i in rows_with_note:
            gap_found, cols = get_note_gap(matrix[i, :], projection)
            if gap_found:
                matrix[i, cols] = 1
        return matrix


class OctaveChangeStrategy:  # move 1 octave up or down, if possible
    def apply(self, matrix, up_or_down=None):
        if up_or_down is None:
            up_or_down = np.sign(np.random.rand() - 0.5)

        success, transposed_matrix = transpose_octave(matrix, up_or_down)
        if not success:
            success, transposed_matrix = transpose_octave(matrix, -up_or_down)

        return transposed_matrix


class TimeChangeStrategy:  # change duration of a random note, between a stablished note size
    def __init__(self):
        self._change_probability = 0.3
        fuse = TIME_DIVISION/32
        self._note_change_scale = int(2 * fuse)
        self._time_lower_bound = int(1 * fuse)
        self._time_upper_bound = int(8 * fuse)

    def apply(self, matrix):
        empty_note_row = TIME_DIVISION * -1
        rows_with_note = np.where(np.sum(matrix, 1) != empty_note_row)[0]
        for i in rows_with_note:
            if np.random.rand() <= self._change_probability:
                self.change_note_time(matrix[i, :])
        return matrix

    def change_note_time(self, row):
        incr_or_decr = np.sign(np.random.rand() - 0.5)
        note_on = np.where(row == 1)[0]
        intervals = get_intervals(note_on)
        relevant_intervals = np.where(intervals < -1)[0]
        if len(relevant_intervals) > 0:
            inter = np.random.choice(relevant_intervals)
            note_off_id = note_on[inter]
            note_on_id = note_on[0]
            for id in range(inter-1, 0, -1):
                if intervals[id] != -1:
                    note_on_id = note_on[id]
            time_delta = np.random.choice(range(self._time_lower_bound, self._time_upper_bound, self._note_change_scale))
            new_note_off_id = max(0, int(note_off_id + incr_or_decr*time_delta))
            row[note_on_id:note_off_id + 1] = -1
            row[note_on_id:new_note_off_id] = 1


class NoteSplitStrategy:  # Split large notes in sequence
    def apply(self, matrix):
        return matrix


def get_intervals(notes):
    notes_transposed = np.ones(notes.shape)
    notes_transposed[0:-1] = notes[1:]
    notes_transposed[-1] = notes[-1] + 1
    return notes - notes_transposed


def is_gap_without_other_notes(projection):
    return np.sum(projection) == len(projection) * -1 * SCALED_NOTES_NUMBER


def get_note_gap(row, projection):
    note_on = np.where(row == 1)[0]
    intervals = get_intervals(note_on)
    relevant_intervals = np.where(intervals < -1)[0]
    for i in relevant_intervals:
        gap_start = note_on[i]
        gap_end = note_on[i + 1]
        if is_gap_without_other_notes(projection[gap_start + 1: gap_end]):
            return True, range(gap_start + 1, gap_end)
    return False, []


def there_is_note(row):
    return sum(row) > -TIME_DIVISION


def get_highest_note(matrix):
    for i in range(matrix.shape[0] - 1, 0, -1):
        if there_is_note(matrix[i, :]):
            return i
    return SCALED_NOTES_RANGE[0]


def is_note_above_lower_boundary(note):
    return note - NOTES_IN_OCTAVE >= SCALED_NOTES_RANGE[0]


def is_note_below_higher_boundary(note):
    return note + NOTES_IN_OCTAVE < SCALED_NOTES_RANGE[1]


def get_lowest_note(matrix):
    for i in range(matrix.shape[0] - 1):
        if there_is_note(matrix[i, :]):
            return i
    return SCALED_NOTES_RANGE[1]


def transpose_octave(matrix, up_or_down):
    transposed_matrix = -1 * np.ones(matrix.shape)
    transposed = False
    if up_or_down >= 0:
        highest_note = get_highest_note(matrix)
        if is_note_below_higher_boundary(highest_note):
            transposed_matrix[12:highest_note + 1 + 12, :] = matrix[0:highest_note + 1, :]
            transposed = True
    else:
        lowest_note = get_lowest_note(matrix)
        if is_note_above_lower_boundary(lowest_note):
            transposed_matrix[lowest_note - 12:-1 - 12, :] = matrix[lowest_note:-1, :]
            transposed = True

    return transposed, transposed_matrix
