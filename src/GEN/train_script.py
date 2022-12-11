from json import load as jload

from src.GEN.service.GENService import GENService
from src.utils.AIHeroHelper import HarmonicFunction

with open('src/config.json') as config_file:
    config = jload(config_file)

gen_service = GENService(config)

# part = MelodicPart.X
# gen_service.train_model(harmonic_function=HarmonicFunction(1), should_generate_gif=True)
gen_service.train_models(should_generate_gif=True)
