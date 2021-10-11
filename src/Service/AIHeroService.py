import random
import traceback

import numpy as np

from src.Data.AIHeroData import AIHeroData
from src.GAN.Data.GANTrainingData import GANTrainingData
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

    def get_random_train_data_as_matrix(self, melody_specs):
        train_data = GANTrainingData(melody_specs['melodic_part'], file_directory="GAN/Data/train").get_as_matrix()
        index = random.sample(range(train_data.shape[0]), 1)[0]
        output = np.zeros((1, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
        output[0, :, :, :] = train_data[index, :, :, :]
        return output

    def generate_compositions(self, melody_specs_list):
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for melody_specs in melody_specs_list:
                raw_melody = self.generate_melody(melody_specs)
                melody_tuples.append((raw_melody, melody_specs["chord"]))
            ai_hero_data.load_from_GAN_melody_raw(melody_tuples)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
            print(traceback.format_exc())
        return ai_hero_data

    def generate_compositions_with_train_data(self, melody_specs_list):
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for melody_specs in melody_specs_list:
                raw_melody = self.get_random_train_data_as_matrix(melody_specs)
                melody_tuples.append((raw_melody, melody_specs["chord"]))
            ai_hero_data.load_from_GAN_melody_raw(melody_tuples)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
            print(traceback.format_exc())
        return ai_hero_data

# def transform_melody_list_into_composition(melody_list):

# return composition
