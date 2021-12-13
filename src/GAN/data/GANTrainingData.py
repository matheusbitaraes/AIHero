from glob import glob

from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroEnums import MelodicPart


class GANTrainingData:
    def __init__(self, config, melodic_part=MelodicPart.X, data=None):
        if data is not None:
            self._ai_hero_data = data
        else:
            self._ai_hero_data = AIHeroData()
            file_directory = config["train_data_folder"]
            self._ai_hero_data.load_from_midi_files(glob(f"{file_directory}/part_{melodic_part.name}_*"))

    def get_as_matrix(self):
        return self._ai_hero_data.get_spr_as_matrix()
