from unittest import TestCase

from src.model.ApiModels import HarmonySpecsInput, HarmonySpecs
from src.utils.AIHeroHelper import build_harmony_specs_from_input


class TestAIHeroHelper(TestCase):

    def test_build_harmony_specs_from_input(self):
        input = [HarmonySpecsInput(chord="C", key="C", tempo=120),
                 HarmonySpecsInput(chord="F", key="C", tempo=120),
                 HarmonySpecsInput(chord="G", key="C", tempo=120),
                 HarmonySpecsInput(chord="G", key="G", tempo=120),
                 HarmonySpecsInput(chord="D", key="G", tempo=120),
                 HarmonySpecsInput(chord="C", key="G", tempo=120),
                 HarmonySpecsInput(chord="A#", key="A#", tempo=120),
                 HarmonySpecsInput(chord="Eb", key="A#", tempo=120)]

        expected = [HarmonySpecs(transposition_factor=0, key="C", tempo=120),
                    HarmonySpecs(transposition_factor=-5, key="C", tempo=120),
                    HarmonySpecs(transposition_factor=-7, key="C", tempo=120),
                    HarmonySpecs(transposition_factor=0, key="G", tempo=120),
                    HarmonySpecs(transposition_factor=-7, key="G", tempo=120),
                    HarmonySpecs(transposition_factor=-5, key="G", tempo=120),
                    HarmonySpecs(transposition_factor=0, key="A#", tempo=120),
                    HarmonySpecs(transposition_factor=-5, key="A#", tempo=120)]

        value = build_harmony_specs_from_input(input)
        for i in range(len(value)):
            print(f'check [{expected[i]}] == [{value[i]}], for input [{input[i]}]')
            assert expected[i] == value[i]
