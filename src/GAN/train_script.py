# Simulando requisições feitas à futura api de GANs
# Este é o processo ue irá treinar as GANs.
from json import load as jload

from src.GAN.Service.GANService import GANService
from src.utils.AIHeroEnums import MelodicPart

with open('config.json') as config_file:
    config = jload(config_file)

gan_service = GANService(config)

# part = MelodicPart.Y
# gan_service.train_gan(part=part.value, epochs=200, should_generate_gif=True)
gan_service.train_gans(epochs=200, should_generate_gif=True)