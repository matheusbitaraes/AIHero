from json import load

from src.model.ApiModels import MelodyRequestInput, MelodySpecs, HarmonySpecs, FitnessFunction
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
    return [FitnessFunction(key="notes_on_same_chord_key", name="", description="", weight=1),
            FitnessFunction(key="notes_on_beat_rate", name="", description="", weight=1),
            FitnessFunction(key="note_on_density", name="", description="", weight=1),
            FitnessFunction(key="note_variety_rate", name="", description="", weight=1),
            FitnessFunction(key="single_notes_rate", name="", description="", weight=1),
            FitnessFunction(key="notes_out_of_scale_rate", name="", description="", value=-1)]


def build_request_body(source="evo"):
    harmony_specs = get_harmony_specs()
    evolutionary_specs = get_evo_specs()
    melody_specs = MelodySpecs(harmony_specs=harmony_specs, evolutionary_specs=evolutionary_specs)
    request_input = MelodyRequestInput(source=source, melody_specs=melody_specs)
    return request_input


def build_big_request_body(source="evo"):
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


def add_chord(data, chords):
    data[:, 0, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    # data[:, 4, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    # data[:, 7, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    return data, chords
