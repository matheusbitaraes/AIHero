from json import load as jload

from src.service.AIHeroService import AIHeroService
from src.synth.AISynth import AISynth

# MAIN (simulando as requisições feitas pelo front)

# gera dados fictícios de entrada (gerados pelo front)
from src.utils.AIHeroEnums import MelodicPart

with open('config.json') as config_file:
    config = jload(config_file)


api_input = [
    {
        "melodic_part": "RELAXATION",
        "chord": "C",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "TENSION",
        "chord": "F",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "RELAXATION",
        "chord": "C",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "RELAXATION",
        "chord": "C",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "RELAXATION",
        "chord": "F",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "TENSION",
        "chord": "F",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "RELAXATION",
        "chord": "C",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "RELAXATION",
        "chord": "C",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "TENSION",
        "chord": "G",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "TENSION",
        "chord": "F",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "RELAXATION",
        "chord": "C",
        "key": "C",
        "tempo": 120,
    },
    {
        "melodic_part": "RETAKE",
        "chord": "G",
        "key": "C",
        "tempo": 120,
    }
]

# Acessa módulo AIHERO pedindo a melodia especificada
ai_hero_service = AIHeroService(config)
# ai_hero_data_test = ai_hero_service.generate_compositions_with_train_data(api_input)
# ai_hero_data_test.append_base_track(midi_file="resources/super_simple_base.mid")
# ai_hero_data_test.export_as_midi(file_name="resources/exported_melody_test")
#
# ai_hero_data = ai_hero_service.generate_GAN_compositions(api_input)
# ai_hero_data.append_base_track(midi_file="resources/super_simple_base.mid")
# ai_hero_data.export_as_midi(file_name="resources/exported_melody_gan")
#
ai_hero_data = ai_hero_service.generate_compositions(api_input)
ai_hero_data.export_spr_as_image(file_name="resources/exported_image")
ai_hero_data.append_base_track(midi_file="resources/super_simple_base.mid")
ai_hero_data.export_as_midi(file_name="resources/exported_melody_evo")

