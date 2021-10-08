from json import load as jload

from src.Service.AIHeroService import AIHeroService
from src.Synth.AISynth import AISynth

# MAIN (simulando as requisições feitas pelo front)

# gera dados fictícios de entrada (gerados pelo front)
from src.utils.AIHeroEnums import MelodicPart

with open('config.json') as config_file:
    config = jload(config_file)


bars = [
        {'melodic_part': MelodicPart.X, 'chord': 'C7'},
        {'melodic_part': MelodicPart.Y, 'chord': 'F7'},
        {'melodic_part': MelodicPart.X, 'chord': 'C7'},
        {'melodic_part': MelodicPart.X, 'chord': 'C7'},
        {'melodic_part': MelodicPart.X, 'chord': 'F7'},
        {'melodic_part': MelodicPart.Y, 'chord': 'F7'},
        {'melodic_part': MelodicPart.X, 'chord': 'C7'},
        {'melodic_part': MelodicPart.X, 'chord': 'C7'},
        {'melodic_part': MelodicPart.Y, 'chord': 'G7'},
        {'melodic_part': MelodicPart.Y, 'chord': 'F7'},
        {'melodic_part': MelodicPart.X, 'chord': 'C7'},
        {'melodic_part': MelodicPart.Z, 'chord': 'G7'},
]
# Acessa módulo AIHERO pedindo a melodia especificada
ai_hero_service = AIHeroService(config)
ai_hero_data_test = ai_hero_service.generate_compositions_with_train_data(bars)
ai_hero_data_test.export_spr_as_image(file_name="Resources/exported_image_test")
ai_hero_data_test.append_base_track(midi_file="Resources/blues_base.mid")
ai_hero_data_test.export_as_midi(file_name="Resources/exported_melody_test")

ai_hero_data = ai_hero_service.generate_compositions(bars)
ai_hero_data.export_spr_as_image(file_name="Resources/exported_image")
ai_hero_data.append_base_track(midi_file="Resources/blues_base.mid")
ai_hero_data.export_as_midi(file_name="Resources/exported_melody")

# Executa a melodia (o que seria feito pelo front)
# ai_hero_synth_service = AISynth(config)
# ai_hero_synth_service.play_compositions(ai_hero_data.get_mingus_composition())

