from src.Data.AIHeroData import AIHeroData
from src.GAN.Exceptions.GANExceptions import GanTrainingException
from src.utils.AIHeroEnums import MelodicPart
from glob import glob


class GANTrainingData:
    def __init__(self, melodic_part=MelodicPart.X, file_directory="Data/train"):
        self._ai_hero_data = AIHeroData()
        self._file_glob = glob(f"{file_directory}/part_{melodic_part.name}_*")

    def get_as_matrix(self):
        self._ai_hero_data.load_from_midi_files(self._file_glob)
        return self._ai_hero_data.get_spr_as_matrix()

    def load(self):
        self._ai_hero_data.load_from_midi_files(self._file_glob)
        return self._ai_hero_data
