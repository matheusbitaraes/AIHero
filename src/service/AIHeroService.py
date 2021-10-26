import random
import traceback

import numpy as np

from src.EVO.service.EVOService import EVOService
from src.GAN.data.GANTrainingData import GANTrainingData
from src.GAN.service.GANService import GANService
from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroEnums import MelodicPart


class AIHeroService:
    def __init__(self, config):
        self.gan_service = GANService(config)
        self.evo_service = EVOService(config)

    def generate_GAN_compositions(self, melody_specs_list):
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for melody_specs in melody_specs_list:
                raw_melody = self.gan_service.generate_melody(specs=melody_specs)
                melody_tuples.append((raw_melody, melody_specs["chord"]))
            ai_hero_data.load_from_GAN_melody_raw(melody_tuples)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
            print(traceback.format_exc())
        return ai_hero_data

    def generate_compositions(self, melody_specs_list):
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for melody_specs in melody_specs_list:
                raw_melody = self.evo_service.generate_melody(specs=melody_specs)
                melody_tuples.append((raw_melody, melody_specs["chord"]))
            ai_hero_data.load_from_EVO_melody_raw(melody_tuples)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
            print(traceback.format_exc())
        return ai_hero_data

    def generate_compositions_with_train_data(self, melody_specs_list):
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for melody_specs in melody_specs_list:
                raw_melody = self.gan_service.get_random_train_data(specs=melody_specs)
                melody_tuples.append((raw_melody, melody_specs["chord"]))
            ai_hero_data.load_from_GAN_melody_raw(melody_tuples)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot generate melody from GAN training data: {e}")
            print(traceback.format_exc())
        return ai_hero_data