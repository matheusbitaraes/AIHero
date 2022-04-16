from enum import Enum
from src.model.ApiModels import HarmonySpecs

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
from src.utils.AIHeroGlobals import FACTOR_TO_HARMONIC_FUNCTION, CHORD_STRING_TO_FACTOR


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
    return FACTOR_TO_HARMONIC_FUNCTION[chord_value]


def convert_chord_into_factor(chord_string, key_string):
    chord_transpose = CHORD_STRING_TO_FACTOR[chord_string.split(":")[0]]
    key_transpose = CHORD_STRING_TO_FACTOR[key_string]
    factor = chord_transpose - key_transpose
    if factor > 0:
        return factor - 12
    else:
        return factor


def build_harmony_specs_from_input(specs_input):
    harmony_specs = []
    for spec in specs_input:
        harmony_specs.append(HarmonySpecs(transposition_factor=convert_chord_into_factor(spec.chord, spec.key),
                                          key=spec.key,
                                          tempo=spec.tempo))
    return harmony_specs
