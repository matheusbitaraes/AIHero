import concurrent.futures
import traceback

from src.EVO.service.EVOService import EVOService
from src.GAN.service.GANService import GANService
from src.data.AIHeroData import AIHeroData


class AIHeroService:
    def __init__(self, config):
        self._threads_enabled = config["enable_parallelization"]
        self.gan_service = GANService(config)
        self.evo_service = EVOService(config)

    def generate_GAN_compositions(self, melody_specs_list, melody_id, harmony_file=None):
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for melody_specs in melody_specs_list:
                raw_melody = self.gan_service.generate_melody(specs=melody_specs, melody_id=melody_id)
                melody_tuples.append((raw_melody, melody_specs.chord))
            ai_hero_data.load_from_GAN_melody_raw(melody_tuples)
            if harmony_file is not None:
                ai_hero_data.append_base_track(harmony_file)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
            print(traceback.format_exc())
        return ai_hero_data

    def generate_compositions(self, harmony_specs, evolutionary_specs=None, melody_id="", harmony_file=None):
        if evolutionary_specs is not None:
            self.evo_service.update_fitness_functions(evolutionary_specs)
        ai_hero_data = AIHeroData()
        melody_tuples = []

        try:
            if self._threads_enabled:
                def execute(melody_spec, index):
                    result = self.evo_service.generate_melody(specs=melody_spec, melody_id=melody_id)
                    return result, melody_spec.chord, index

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = []
                    idx = 0
                    for spec in harmony_specs:
                        futures.append(executor.submit(execute, spec, idx))
                        idx += 1
                    for future in concurrent.futures.as_completed(futures):
                        melody_tuples.append(future.result())
            else:
                for spec in harmony_specs:
                    raw_melody = self.evo_service.generate_melody(specs=spec, melody_id=melody_id)
                    melody_tuples.append((raw_melody, spec.chord))

            ai_hero_data.load_from_EVO_melody_raw(melody_tuples)
            if harmony_file is not None:
                ai_hero_data.append_base_track(harmony_file)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot Generate Melody: {e}")
        print(traceback.format_exc())

        return ai_hero_data

    def generate_compositions_with_train_data(self, melody_specs_list, melody_id="", harmony_file=None):
        ai_hero_data = AIHeroData()
        melody_tuples = []
        try:
            for melody_specs in melody_specs_list:
                raw_melody = self.gan_service.get_random_train_data(specs=melody_specs)
                melody_tuples.append((raw_melody, melody_specs.chord))
            ai_hero_data.load_from_GAN_melody_raw(melody_tuples)
            if harmony_file is not None:
                ai_hero_data.append_base_track(harmony_file)
        except Exception as e:
            print(f"Exception in AI Hero Service: Cannot generate melody from GAN training data: {e}")
            print(traceback.format_exc())
        return ai_hero_data
