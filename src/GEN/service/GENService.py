import traceback
from typing import List

from src.GEN.engine.LSTM.LSTM import LSTM
from src.GEN.engine.convolutionalGAN.AIHeroGANConvolutional import AIHeroGANConvolutional
from src.GEN.exceptions.GENExceptions import GENTrainingException
from src.model.ApiModels import HarmonySpecs
from src.utils.AIHeroHelper import HarmonicFunction, get_harmonic_function_of_chord


class GENService:
    def __init__(self, config):
        self._config = config
        self._selected_model = self._get_model_by_config(config)
        self.models = self.build_models()

    def train_models(self, verbose: bool = False, should_generate_gif: bool = False):
        for function in HarmonicFunction:
            if verbose:
                print(f"Training GAN of function: {function}")
            try:
                self.train_model(harmonic_function=function, should_generate_gif=should_generate_gif)
            except GENTrainingException:
                print(f"Error training GAN for function {function}")
                raise GENTrainingException

    def train_model(self, harmonic_function: HarmonicFunction, should_generate_gif: bool = False,
                    num_epochs: bool = None):
        return self.models[harmonic_function.name].train(should_generate_gif=should_generate_gif, num_epochs=num_epochs)

    def generate_melodies(self, specs_list: List[HarmonySpecs]):
        for specs in specs_list:
            return self.generate_melody(specs)

    def generate_melody(self, specs: HarmonySpecs = None, num_melodies: int = 1, melody_id: str = ""):
        harmonic_function = HarmonicFunction(get_harmonic_function_of_chord(specs.transposition_factor))
        try:
            model = self.models[harmonic_function.name]
            return model.generate_melody_matrix(num_melodies=num_melodies, new_seed=True)
        except GENTrainingException as e:
            print(f"Exception in GAN Service: Gan for function {harmonic_function.name} is not trained: {e}")
        except Exception as e:
            print(f"exception in GAN Service: {e}")
            print(traceback.format_exc())

    def build_models(self):
        model_map = {}
        for function in HarmonicFunction:
            model_map[function.name] = self._selected_model(self._config, harmonic_function=function)
        return model_map

    def get_random_train_data(self, specs: HarmonySpecs = None):
        harmonic_function = HarmonicFunction(get_harmonic_function_of_chord(specs.transposition_factor))
        try:
            gan = self.models[harmonic_function.name]
            return gan.get_random_train_data()
        except GENTrainingException as e:
            print(f"Exception in GAN Service: Could not get GAN for part {harmonic_function.name}: {e}")
        except Exception as e:
            print(f"exception in GAN Service: {e}")
            print(traceback.format_exc())

    def _get_model_by_config(self, config):
        lstm_name = LSTM(config).get_name()
        if config["model_name"] == lstm_name:
            return LSTM
        else:
            return AIHeroGANConvolutional
