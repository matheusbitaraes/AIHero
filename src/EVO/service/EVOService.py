from EVO.engine.AIHeroEVO import AIHeroEVO


class EVOService:
    def __init__(self, config):
        self.evolutionary_algorithm = AIHeroEVO(config)

    def generate_melody(self, specs, melody_id=""):
        return self.evolutionary_algorithm.generate_melody(specs, melody_id=melody_id)
