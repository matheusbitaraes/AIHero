# Simulando requisições feitas à futura api de GANs
# Este é o processo ue irá treinar as GANs.
from json import load as jload

from src.GAN.Service.GANService import GANService

with open('config.json') as config_file:
    config = jload(config_file)

gan_service = GANService(config)
gan_service.train_gans(epochs=50, verbose=True, should_generate_gif=True)

