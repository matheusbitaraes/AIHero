import unittest

import numpy as np

from src.GAN.engine.augmentation.NoteJoinStrategy import NoteJoinStrategy
from src.GAN.engine.augmentation.OctaveChangeStrategy import OctaveChangeStrategy
from src.GAN.engine.augmentation.TimeChangeStrategy import TimeChangeStrategy
from utils.AIHeroGlobals import SCALED_NOTES_NUMBER, TIME_DIVISION


class AugmentationStrategiesTest(unittest.TestCase):
    def test_octave_change(self):
        # initialize data
        matrix_list = initialize_octave_matrix_list()
        expected_matrix_list = initialize_octave_expected_list()
        ud_list = [1, -1, 1]
        for i in range(len(matrix_list)):
            result = OctaveChangeStrategy().apply(matrix_list[i], up_or_down=ud_list[i])
            self.assertTrue((result == expected_matrix_list[i]).all())  # add assertion here

    def test_note_join(self):
        # initialize data
        matrix_list = initialize_note_matrix_list()
        expected_matrix_list = initialize_note_expected_list()
        for i in range(len(matrix_list)):
            result = NoteJoinStrategy().apply(matrix_list[i])
            self.assertTrue((result == expected_matrix_list[i]).all())  # add assertion here

    def test_time_change(self):
        # initialize data
        matrix_list = initialize_time_matrix_list()
        expected_matrix_list = initialize_time_expected_list()
        for i in range(len(matrix_list)):
            result = TimeChangeStrategy().apply(matrix_list[i])
            self.assertTrue((result == expected_matrix_list[i]).all())  # add assertion here


def initialize_octave_matrix_list():
    matrix_list = []

    # upper limited: can only be transposed down
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[37, 0:5] = 1
    matrix[36, 5:16] = 1
    matrix[30, 32:40] = 1
    matrix[25, 35:64] = 1
    matrix_list.append(matrix)

    # lower limited: can only be transposed up
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[30, 0:5] = 1
    matrix[15, 5:16] = 1
    matrix[10, 32:40] = 1
    matrix[5, 35:64] = 1
    matrix_list.append(matrix)

    # not limited
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[24, 0:5] = 1
    matrix[22, 5:16] = 1
    matrix[20, 32:40] = 1
    matrix[15, 35:64] = 1
    matrix_list.append(matrix)

    return matrix_list


def initialize_octave_expected_list():
    matrix_list = []

    # upper limited: can only be transposed down
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[37 - 12, 0:5] = 1
    matrix[36 - 12, 5:16] = 1
    matrix[30 - 12, 32:40] = 1
    matrix[25 - 12, 35:64] = 1
    matrix_list.append(matrix)

    # lower limited: can only be transposed up
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[30 + 12, 0:5] = 1
    matrix[15 + 12, 5:16] = 1
    matrix[10 + 12, 32:40] = 1
    matrix[5 + 12, 35:64] = 1
    matrix_list.append(matrix)

    # not limited
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[24 + 12, 0:5] = 1
    matrix[22 + 12, 5:16] = 1
    matrix[20 + 12, 32:40] = 1
    matrix[15 + 12, 35:64] = 1
    matrix_list.append(matrix)

    return matrix_list


def initialize_time_matrix_list():
    matrix_list = []

    # upper limited: can only be transposed down
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[37, 0:5] = 1
    matrix[10, 5:16] = 1
    matrix[30, 32:40] = 1
    matrix[25, 35:64] = 1
    matrix_list.append(matrix)

    # lower limited: can only be transposed up
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[30, 0:5] = 1
    matrix[15, 5:16] = 1
    matrix[10, 32:40] = 1
    matrix[5, 35:64] = 1
    matrix_list.append(matrix)

    # not limited
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[24, 0:5] = 1
    matrix[22, 5:16] = 1
    matrix[20, 32:40] = 1
    matrix[15, 35:64] = 1
    matrix_list.append(matrix)

    return matrix_list


def initialize_time_expected_list():
    matrix_list = []

    # upper limited: can only be transposed down
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[37 - 12, 0:5] = 1
    matrix[36 - 12, 5:16] = 1
    matrix[30 - 12, 32:40] = 1
    matrix[25 - 12, 35:64] = 1
    matrix_list.append(matrix)

    # lower limited: can only be transposed up
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[30 + 12, 0:5] = 1
    matrix[15 + 12, 5:16] = 1
    matrix[10 + 12, 32:40] = 1
    matrix[5 + 12, 35:64] = 1
    matrix_list.append(matrix)

    # not limited
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[24 + 12, 0:5] = 1
    matrix[22 + 12, 5:16] = 1
    matrix[20 + 12, 32:40] = 1
    matrix[15 + 12, 35:64] = 1
    matrix_list.append(matrix)

    return matrix_list


def initialize_note_matrix_list():
    matrix_list = []

    # upper limited: can only be transposed down
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[37, 0:5] = 1
    matrix[37, 10:15] = 1
    matrix[30, 32:40] = 1
    matrix[25, 35:64] = 1
    matrix_list.append(matrix)

    # lower limited: can only be transposed up
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[30, 0:5] = 1
    matrix[15, 5:16] = 1
    matrix[30, 32:40] = 1
    matrix[15, 35:64] = 1
    matrix_list.append(matrix)

    # not limited
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[24, 0:5] = 1
    matrix[24, 7:9] = 1
    matrix[22, 10:16] = 1
    matrix[15, 35:64] = 1
    matrix_list.append(matrix)

    return matrix_list


def initialize_note_expected_list():
    matrix_list = []

    # upper limited: can only be transposed down
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[37, 0:15] = 1
    matrix[30, 32:40] = 1
    matrix[25, 35:64] = 1
    matrix_list.append(matrix)

    # lower limited: can only be transposed up
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[30, 0:5] = 1
    matrix[15, 5:16] = 1
    matrix[30, 32:40] = 1
    matrix[15, 35:64] = 1
    matrix_list.append(matrix)

    # not limited
    matrix = -1 * np.ones([SCALED_NOTES_NUMBER, TIME_DIVISION])
    matrix[24, 0:9] = 1
    matrix[22, 10:16] = 1
    matrix[15, 35:64] = 1
    matrix_list.append(matrix)

    return matrix_list


if __name__ == '__main__':
    unittest.main()
