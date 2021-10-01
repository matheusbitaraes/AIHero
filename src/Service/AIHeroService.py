from src.AIHeroData import AIHeroData
from src.GAN.Service.GANService import GANService


class AIHeroService:
    def __init__(self, config):
        self.gan_service = GANService(config)

    def generate_melody(self, melody_specs):
        # inicialmente só vai pegar a gan e gerar um valor dela de acordo com os specs
        # o segundo passo é chamar o código do ai hero e fazer a gan gerar a população do algorimto genético
        gan_specs = melody_specs
        raw_melody = self.gan_service.generate_melody(specs=gan_specs)
        return raw_melody

    def generate_ai_hero_data(self, melody_specs_list):
        ai_hero_data = AIHeroData()
        melody_list = []
        try:
            for melody_specs in melody_specs_list:
                raw_melody = self.generate_melody(melody_specs)
                melody_list.append(raw_melody)
            ai_hero_data.load_from_GAN_melody_raw(melody_list)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
        return ai_hero_data


# def transform_melody_list_into_composition(melody_list):

    # return composition
