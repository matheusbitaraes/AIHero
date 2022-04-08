import traceback

from src.GAN.engine.AIHeroGAN import AIHeroGAN
from src.GAN.exceptions.GANExceptions import GanTrainingException
from src.utils.AIHeroEnums import HarmonicFunction, get_harmonic_function_of_chord


class GANService:
    def __init__(self, config):
        self._config = config
        # self.gan = AIHeroGAN(config)
        self.gans = self.build_gans()

    def train_gans(self, verbose=False, should_generate_gif=False):
        for function in HarmonicFunction:
            if verbose:
                print(f"Training GAN of function: {function}")
            try:
                self.train_gan(harmonic_function=function, should_generate_gif=should_generate_gif)
            except GanTrainingException:
                print(f"Error training GAN for function {function}")
                raise GanTrainingException

    # def train_single_gan(self, verbose=False, should_generate_gif=False):
    #     if verbose:
    #         print(f"Training single GAN")
    #     try:
    #         self.gan.single_gan_train(should_generate_gif=should_generate_gif)
    #     except GanTrainingException:
    #         print(f"Error training single GAN")
    #         raise GanTrainingException

    def train_gan(self, harmonic_function, should_generate_gif=False, num_epochs=None):
        return self.gans[harmonic_function.name].train(should_generate_gif=should_generate_gif, num_epochs=num_epochs)

    def generate_melodies(self, specs_list):
        for specs in specs_list:
            return self.generate_melody(specs)

    def generate_melody(self, specs=None, num_melodies=1, melody_id=""):
        harmonic_function = HarmonicFunction(get_harmonic_function_of_chord(specs.chord))
        try:
            gan = self.gans[harmonic_function.name]
            return gan.generate_melody_matrix(num_melodies=num_melodies, new_seed=True)
        except GanTrainingException as e:
            print(f"Exception in GAN Service: Gan for function {harmonic_function.name} is not trained: {e}")
        except Exception as e:
            print(f"exception in GAN Service: {e}")
            print(traceback.format_exc())

    def build_gans(self):
        gan_map = {}
        for function in HarmonicFunction:
            gan_map[function.name] = AIHeroGAN(self._config, harmonic_function=function)
        return gan_map

    def get_random_train_data(self, specs=None):
        harmonic_function = HarmonicFunction(get_harmonic_function_of_chord(specs.chord))
        try:
            gan = self.gans[harmonic_function.name]
            return gan.get_random_train_data()
        except GanTrainingException as e:
            print(f"Exception in GAN Service: Could not get GAN for part {harmonic_function.name}: {e}")
        except Exception as e:
            print(f"exception in GAN Service: {e}")
            print(traceback.format_exc())
