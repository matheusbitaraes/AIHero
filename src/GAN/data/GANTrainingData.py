from src.data.AIHeroData import AIHeroData
from src.GAN.exceptions.GANExceptions import GanTrainingException
from src.utils.AIHeroEnums import MelodicPart
from glob import glob


class GANTrainingData:
    def __init__(self, config, melodic_part=MelodicPart.X):
        self._ai_hero_data = AIHeroData()
        file_directory = config["train_data_folder"]
        self._file_glob = glob(f"{file_directory}/part_{melodic_part.name}_*")

    def get_as_matrix(self):
        self._ai_hero_data.load_from_midi_files(self._file_glob)
        return self._ai_hero_data.get_spr_as_matrix()

    def load(self):
        self._ai_hero_data.load_from_midi_files(self._file_glob)
        return self._ai_hero_data
