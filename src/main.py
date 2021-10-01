from json import load as jload

from src.Service.AIHeroService import AIHeroService
from src.Synth.AISynth import AISynth

# MAIN (simulando as requisições feitas pelo front)

# gera dados fictícios de entrada (gerados pelo front)
from src.utils.AIHeroEnums import MelodicPart

with open('config.json') as config_file:
    config = jload(config_file)


bars = [
        {'melodic_part': MelodicPart.X.value, 'chord': 'C7'},
        {'melodic_part': MelodicPart.Y.value, 'chord': 'F7'},
        {'melodic_part': MelodicPart.X.value, 'chord': 'C7'},
        {'melodic_part': MelodicPart.X.value, 'chord': 'C7'},
        {'melodic_part': MelodicPart.Z.value, 'chord': 'F7'},
        {'melodic_part': MelodicPart.Y.value, 'chord': 'F7'},
        {'melodic_part': MelodicPart.X.value, 'chord': 'C7'},
        {'melodic_part': MelodicPart.X.value, 'chord': 'C7'},
        {'melodic_part': MelodicPart.K.value, 'chord': 'G7'},
        {'melodic_part': MelodicPart.Y.value, 'chord': 'F7'},
        {'melodic_part': MelodicPart.X.value, 'chord': 'C7'},
        {'melodic_part': MelodicPart.J.value, 'chord': 'G7'},
]
# Acessa módulo AIHERO pedindo a melodia especificada
ai_hero_service = AIHeroService(config)
ai_hero_data = ai_hero_service.generate_ai_hero_data(bars)

ai_hero_data.export_as_image(file_name="exported_image")
ai_hero_data.export_as_midi()

# Executa a melodia (o que seria feito pelo front)
# ai_hero_synth_service = AISynth()
# ai_hero_synth_service.play_composition(ai_hero_data.get_mingus_composition())

