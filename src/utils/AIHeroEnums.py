from enum import Enum


# class MelodicPart(Enum):
#     X = 'RELAXATION'
#     Y = 'TENSION'
#     Z = 'RETAKE'
#
#     def get_from_value(self, value):
#         if value == "RELAXATION":
#             return MelodicPart.X
#         if value == "TENSION":
#             return MelodicPart.Y
#         if value == "RETAKE":
#             return MelodicPart.X


class HarmonicFunction(Enum):
    TONIC = 1
    DOMINANT = 2
    SUBDOMINANT = 3

    def get_from_value(self, value):
        if value == 1:
            return HarmonicFunction.TONIC
        if value == 2:
            return HarmonicFunction.DOMINANT
        if value == 3:
            return HarmonicFunction.SUBDOMINANT


def get_harmonic_function_of_chord(chord_value):
    # I - TONICA
    # II - SUBDOMINANTE
    # III - TONICA
    # IV - SUBDOMINANTE
    # V - DOMINANTE
    # VI - TONICA
    # VII - DOMINANTE
    # 0 - fora do campo harmonico, 1 - tonica, 2 - subdominante, 3 - dominante
    table = {
        -11: 3,  # B
        -10: 0,  # A#
        -9: 1,  # A
        -8: 0,  # G#
        -7: 3,  # G
        -6: 0,  # F#
        -5: 2,  # F
        -4: 1,  # E
        -3: 0,  # D#
        -2: 2,  # D
        -1: 0,  # C#
        0: 1,  # C
        1: 0,  # C#
        2: 2,  # D
        3: 0,  # D#
        4: 1,  # E
        5: 2,  # F
        6: 0,  # F#
        7: 3,  # G
        8: 0,  # G#
        9: 1,  # A
        10: 0,  # A#
        11: 3,  # B
    }
    return table[chord_value]