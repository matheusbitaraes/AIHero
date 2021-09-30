from src.AIHeroData import AIHeroData
from src.GAN.Exceptions.GANExceptions import GanTrainingException
from src.utils.AIHeroEnums import MelodicPart
from glob import glob


class GANTrainingData:
    def __init__(self, melodic_part=MelodicPart.X):
        self.ai_hero_data = AIHeroData()
        if melodic_part == MelodicPart.X:
            self.file_glob = glob("Data/resources/train/part_x_*")
        elif melodic_part == MelodicPart.Y:
            self.file_glob = glob("Data/resources/train/part_y_*")
        elif melodic_part == MelodicPart.Z:
            self.file_glob = glob("Data/resources/train/part_z_*")
        elif melodic_part == MelodicPart.K:
            self.file_glob = glob("Data/resources/train/part_j_*")
        elif melodic_part == MelodicPart.J:
            self.file_glob = glob("Data/resources/train/part_k_*")
        else:
            raise GanTrainingException("Melodic part does not exist")

    def get(self):
        self.ai_hero_data.load_from_midi_files(self.file_glob)
        return self.ai_hero_data.get_piano_roll_as_matrix()
