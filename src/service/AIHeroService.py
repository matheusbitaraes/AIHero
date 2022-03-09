import concurrent.futures
import traceback

from src.EVO.service.EVOService import EVOService
from src.GAN.service.GANService import GANService
from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroHelper import build_harmony_specs_from_input


class AIHeroService:
    def __init__(self, config):
        self._threads_enabled = config["enable_parallelization"]
        self._thread_max_workers = config["thread_max_workers"]
        self.gan_service = GANService(config)
        self.evo_service = EVOService(config)

    def generate_GAN_compositions(self, melody_specs_input, melody_id):
        melody_specs = build_harmony_specs_from_input(melody_specs_input)
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for melody_specs in melody_specs:
                raw_melody = self.gan_service.generate_melody(specs=melody_specs, melody_id=melody_id)
                melody_tuples.append((raw_melody, melody_specs.transposition_factor))
            ai_hero_data.load_from_GAN_melody_raw(melody_tuples)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
            print(traceback.format_exc())
        return ai_hero_data

    def generate_compositions(self, melody_specs_input, evolutionary_specs=None, melody_id=""):
        melody_specs = build_harmony_specs_from_input(melody_specs_input)
        if evolutionary_specs is not None:
            self.evo_service.update_fitness_functions(evolutionary_specs)
        ai_hero_data = AIHeroData()
        melody_tuples = []

        try:
            if self._threads_enabled:
                def execute(spec, index):
                    result = self.evo_service.generate_melody(specs=spec, melody_id=melody_id)
                    return result, spec.transposition_factor, index

                with concurrent.futures.ThreadPoolExecutor(max_workers=self._thread_max_workers) as executor:
                    futures = []
                    idx = 0
                    for spec in melody_specs:
                        futures.append(executor.submit(execute, spec, idx))
                        idx += 1
                    for future in concurrent.futures.as_completed(futures):
                        melody_tuples.append(future.result())
            else:
                for spec in melody_specs:
                    raw_melody = self.evo_service.generate_melody(specs=spec, melody_id=melody_id)
                    melody_tuples.append((raw_melody, spec.transposition_factor))

            ai_hero_data.load_from_EVO_melody_raw(melody_tuples)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
            print(traceback.format_exc())
        print(traceback.format_exc())

        return ai_hero_data

    def generate_compositions_with_train_data(self, melody_specs_input, melody_id=""):
        melody_specs = build_harmony_specs_from_input(melody_specs_input)
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for spec in melody_specs:
                raw_melody = self.gan_service.get_random_train_data(specs=spec)
                melody_tuples.append((raw_melody, spec.transposition_factor))
            ai_hero_data.load_from_GAN_melody_raw(melody_tuples)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot generate melody from GAN training data: {e}")
            print(traceback.format_exc())
        return ai_hero_data
