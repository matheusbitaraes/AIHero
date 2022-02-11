from json import load as jload

from service.AIHeroService import AIHeroService

# MAIN (simulando as requisições feitas pelo front)

# gera dados fictícios de entrada (gerados pelo front)
from utils.AIHeroGlobals import DEFAULT_MELODY_REQUEST

with open('src/config.json') as config_file:
    config = jload(config_file)


api_input = DEFAULT_MELODY_REQUEST

# Acessa módulo AIHERO pedindo a melodia especificada
ai_hero_service = AIHeroService(config)
# ai_hero_data_test = ai_hero_service.generate_compositions_with_train_data(api_input, id="test")
# ai_hero_data_test.append_base_track(midi_file="src/resources/blues_base.mid")
# ai_hero_data_test.export_as_midi(file_name="src/resources/exported_melody_test")

# ai_hero_data = ai_hero_service.generate_GAN_compositions(api_input)
# ai_hero_data.append_base_track(midi_file="resources/blues_base.mid")
# ai_hero_data.export_as_midi(file_name="resources/exported_melody_gan")
# # #
ai_hero_data = ai_hero_service.generate_compositions(api_input)
# ai_hero_data.export_spr_as_image(file_name="resources/exported_image")
ai_hero_data.append_base_track(midi_file="resources/blues_base.mid")
ai_hero_data.export_as_midi(file_name="resources/exported_melody_evo")

