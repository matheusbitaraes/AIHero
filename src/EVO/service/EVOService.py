from src.EVO.engine.AIHeroEVO import AIHeroEVO


class EVOService:
    def __init__(self, config):
        self.evolutionary_algorithm = AIHeroEVO(config)

    def generate_melody(self, specs, melody_id=""):
        return self.evolutionary_algorithm.generate_melody(specs, melody_id=melody_id)

    def update_fitness_functions(self, evolutionary_specs):
        self.evolutionary_algorithm.update_fitness_functions(evolutionary_specs)
