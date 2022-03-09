from json import load as jload

from src.GAN.service.GANService import GANService
from src.utils.AIHeroEnums import MelodicPart

with open('config.json') as config_file:
    config = jload(config_file)

gan_service = GANService(config)

part = MelodicPart.Y
# gan_service.train_gan(part=part.value, should_generate_gif=True)
gan_service.train_gans(should_generate_gif=True)
