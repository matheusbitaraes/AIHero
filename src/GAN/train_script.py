# Simulando requisições feitas à futura api de GANs
# Este é o processo ue irá treinar as GANs.
from json import load as jload

from src.GAN.Service.GANService import GANService
from src.utils.AIHeroEnums import MelodicPart

with open('config.json') as config_file:
    config = jload(config_file)

gan_service = GANService(config)

part = MelodicPart.X
gan_service.train_gan(part=part.value, epochs=100, should_generate_gif=True)
# gan_service.train_gans(epochs=20, should_generate_gif=True)