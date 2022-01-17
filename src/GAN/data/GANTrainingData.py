import time
from glob import glob

from src.GAN.engine.augmentation.AugmentationEngine import AugmentationEngine
from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroEnums import MelodicPart


class GANTrainingData:
    def __init__(self, config, melodic_part=MelodicPart.X, data=None):
        if data is not None:
            self._ai_hero_data = data
        else:
            self._ai_hero_data = AIHeroData()
            file_directory = config["training"]["train_data_folder"]
            self._ai_hero_data.load_from_midi_files(glob(f"{file_directory}/part_{melodic_part.name}_*"))

        # data augmentation
        augmentation_config = config["data_augmentation"]
        if augmentation_config["enabled"]:
            self.augmentation_engine = AugmentationEngine(
                augmentation_config["data_augmentation_strategy_pipeline"])
            if config["verbose"]:
                start = time.time()
                start_size = self._ai_hero_data.get_spr_as_matrix().shape[0]
                print("Augmenting dataset...")

            self.replicate(self._ai_hero_data.get_spr_as_matrix().shape[0] * augmentation_config["replication_factor"])
            self.augment()  # todo: ver se esse é de fato o melhor lugar

            if config["verbose"]:
                end_size = self._ai_hero_data.get_spr_as_matrix().shape[0]
                print(f"dataset augmented from {start_size} samples to {end_size} samples in {time.time()-start}s")
        else:
            self.augmentation_engine = AugmentationEngine()

    def get_as_matrix(self):
        return self._ai_hero_data.get_spr_as_matrix()

    def augment(self):
        self._ai_hero_data.augment(self.augmentation_engine)

    def replicate(self, final_size):
        self._ai_hero_data.replicate(final_size)

    def print_on_terminal(self):
        self._ai_hero_data.print_on_terminal()
        pass

    @property
    def ai_hero_data(self):
        return self._ai_hero_data
