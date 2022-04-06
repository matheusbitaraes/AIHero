from json import load
from unittest import TestCase

from src.data.AIHeroData import AIHeroData
from src.model.ApiModels import MelodyRequestInput, MelodySpecs, HarmonySpecs, FitnessFunction
from src.service.AIHeroService import AIHeroService
from src.utils.AIHeroGlobals import TIME_DIVISION


def get_harmony_specs():
    return [HarmonySpecs(melodic_part="RELAXATION", chord=0, key="C", tempo=80),
            HarmonySpecs(melodic_part="TENSION", chord=-5, key="C", tempo=80),
            HarmonySpecs(melodic_part="RELAXATION", chord=0, key="C", tempo=80),
            HarmonySpecs(melodic_part="RELAXATION", chord=0, key="C", tempo=80),
            HarmonySpecs(melodic_part="RELAXATION", chord=-5, key="C", tempo=80),
            HarmonySpecs(melodic_part="TENSION", chord=-5, key="C", tempo=80),
            HarmonySpecs(melodic_part="RELAXATION", chord=0, key="C", tempo=80),
            HarmonySpecs(melodic_part="RELAXATION", chord=0, key="C", tempo=80),
            HarmonySpecs(melodic_part="TENSION", chord=-7, key="C", tempo=80),
            HarmonySpecs(melodic_part="TENSION", chord=-5, key="C", tempo=80),
            HarmonySpecs(melodic_part="RELAXATION", chord=0, key="C", tempo=80),
            HarmonySpecs(melodic_part="RETAKE", chord=-7, key="C", tempo=80)]


def get_evo_specs():
    return [FitnessFunction(key="notes_on_same_chord_key", name="", description="", value=1),
            FitnessFunction(key="notes_on_beat_rate", name="", description="", value=0),
            FitnessFunction(key="note_on_density", name="", description="", value=0.5),
            FitnessFunction(key="note_variety_rate", name="", description="", value=0.7),
            FitnessFunction(key="single_notes_rate", name="", description="", value=0),
            FitnessFunction(key="notes_out_of_scale_rate", name="", description="", value=0)]


def build_request_body(source):
    harmony_specs = get_harmony_specs()
    evolutionary_specs = get_evo_specs()
    melody_specs = MelodySpecs(harmony_specs=harmony_specs, evolutionary_specs=evolutionary_specs)
    request_input = MelodyRequestInput(source=source, melody_specs=melody_specs)
    return request_input


def build_big_request_body(source):
    harmony_specs = get_harmony_specs()
    big_harmony_specs = []
    for i in range(5):
        big_harmony_specs.extend(harmony_specs)
    evolutionary_specs = get_evo_specs()
    melody_specs = MelodySpecs(harmony_specs=big_harmony_specs, evolutionary_specs=evolutionary_specs)
    request_input = MelodyRequestInput(source=source, melody_specs=melody_specs)
    return request_input


def get_config():
    with open("src/test/test_config.json") as config_file:
        return load(config_file)


class TestAIHeroData(TestCase):

    def test_load_pop909_spr_and_with_chord_on_different_keys(self):
        # load a file
        data = AIHeroData()
        data.load_from_pop909_dataset(dataset_path="src/test/resources/pop909-subsample/")

        # add a chord being played on every SPR
        data.execute_function_on_data(add_one_note)

        # export file
        data.export_as_midi("testeeeee")
        assert 1 == 1

    def test_load_manual_spr_and_with_chord_on_different_keys(self):
        # load a file
        data = AIHeroData()
        data.load_spr_from_checkpoint("src/GAN/data/train/manual")

        # add a chord being played on every SPR
        data.execute_function_on_data(add_one_note)

        # export file
        data.export_as_midi("src/test/test_load_manual_spr_and_with_chord_on_different_keys")
        assert 1 == 1

    def test_make_a_blues_sequence_from_train_data(self):
        input = build_big_request_body("train")
        harmony_specs = input.melody_specs.harmony_specs
        ai_hero_service = AIHeroService(get_config())
        data = ai_hero_service.generate_compositions_with_train_data(harmony_specs)

        # add a chord being played on every SPR
        data.execute_function_on_data(add_one_note)

        data.export_as_midi("src/test/test_make_a_blues_sequence_from_train_data")

    def test_make_a_blues_sequence_from_gan_data(self):
        input = build_big_request_body("gan")
        harmony_specs = input.melody_specs.harmony_specs
        ai_hero_service = AIHeroService(get_config())
        data = ai_hero_service.generate_GAN_compositions(harmony_specs, melody_id="id")

        # add a chord being played on every SPR
        data.execute_function_on_data(add_one_note)

        data.export_as_midi("src/test/test_make_a_blues_sequence_from_train_data")

    def test_make_a_blues_sequence_from_evo_data(self):
        input = build_request_body("evo")
        ai_hero_service = AIHeroService(get_config())
        data = ai_hero_service.generate_compositions(input.melody_specs.harmony_specs,
                                                     input.melody_specs.evolutionary_specs)

        # add a chord being played on every SPR
        data.execute_function_on_data(add_one_note)

        data.export_as_midi("src/test/test_make_a_blues_sequence_from_evo_data")


def add_one_note(data, chords):
    data[:, 12, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    data[:, 16, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    data[:, 19, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    return data, chords
