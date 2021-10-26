import traceback

from src.GAN.engine.AIHeroGAN import AIHeroGAN
from src.GAN.exceptions.GANExceptions import GanTrainingException
from src.utils.AIHeroEnums import MelodicPart


class GANService:
    def __init__(self, config):
        self._config = config
        self.gans = self.build_gans()

    def train_gans(self, num_seeds=1, epochs=50, verbose=False, should_generate_gif=False):
        for part in MelodicPart:
            if verbose:
                print(f"Training GAN of melodic part: {part.value}")
            try:
                self.train_gan(part=part.value, num_seeds=num_seeds, epochs=epochs, should_generate_gif=should_generate_gif)
            except GanTrainingException:
                print(f"Error training GAN for part {part.value}")

    def train_gan(self, part, num_seeds=1, epochs=50, should_generate_gif=False):
        return self.gans[part].train(num_seeds=num_seeds, epochs=epochs, should_generate_gif=should_generate_gif)

    def generate_melodies(self, specs_list):
        for specs in specs_list:
            return self.generate_melody(specs)

    def generate_melody(self, specs=None, num_melodies=1):
        melodic_part = MelodicPart(specs["melodic_part"])
        try:
            gan = self.gans[melodic_part.value]
            return gan.generate_melody_matrix(num_melodies=num_melodies, new_seed=True)
        except GanTrainingException as e:
            print(f"Exception in GAN Service: Gan for part {melodic_part} is not trained: {e}")
        except Exception as e:
            print(f"exception in GAN Service: {e}")
            print(traceback.format_exc())

    def build_gans(self):
        gan_map = {}
        for part in MelodicPart:
            gan_map[part.value] = AIHeroGAN(part=part, checkpoint_folder=self._config['checkpoint_folder'],
                                            verbose=self._config["verbose"])
        return gan_map

    def get_random_train_data(self, specs=None):
        melodic_part = MelodicPart(specs["melodic_part"])
        try:
            gan = self.gans[melodic_part.value]
            return gan.get_random_train_data()
        except GanTrainingException as e:
            print(f"Exception in GAN Service: Could not get GAN for part {melodic_part}: {e}")
        except Exception as e:
            print(f"exception in GAN Service: {e}")
            print(traceback.format_exc())