import numpy as np

from src.utils.AIHeroGlobals import SCALED_NOTES_RANGE, NOTES_IN_OCTAVE, TIME_DIVISION


class NoteJoinStrategy:
    def apply(self, matrix):

        return matrix


class OctaveChangeStrategy:
    def apply(self, matrix, up_or_down=None):
        if up_or_down is None:
            up_or_down = np.sign(np.random.rand() - 0.5)

        success, transposed_matrix = transpose_octave(matrix, up_or_down)
        if not success:
            success, transposed_matrix = transpose_octave(matrix, -up_or_down)

        return transposed_matrix


class TimeChangeStrategy:
    def apply(self, matrix):
        return matrix


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
