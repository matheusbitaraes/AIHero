from glob import glob

from src.GAN.engine.augmentation.AugmentationEngine import AugmentationEngine
from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroEnums import MelodicPart


class GANTrainingData:
    def __init__(self, config, melodic_part=MelodicPart.X, data=None):
        if data is not None:
            self._ai_hero_data = data

            # data augmentation
            augmentation_config = config["data_augmentation"]
            if augmentation_config["enabled"]:
                self.augmentation_engine = AugmentationEngine(augmentation_config["data_augmentation_strategy_pipeline"])
            else:
                self.augmentation_engine = AugmentationEngine()

        else:
            self._ai_hero_data = AIHeroData()
            file_directory = config["training"]["train_data_folder"]
            self._ai_hero_data.load_from_midi_files(glob(f"{file_directory}/part_{melodic_part.name}_*"))

    def get_as_matrix(self):
        return self._ai_hero_data.get_spr_as_matrix()

    def augment(self):
        self._ai_hero_data.augment(self.augmentation_engine)

    def replicate(self, final_size):
        self._ai_hero_data.replicate(final_size)
