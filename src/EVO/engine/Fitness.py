from EVO.engine.FitnessFunctionMap import FitnessFunctionMap


class Fitness:
    def __init__(self, config):
        self.function_sets = config["function_sets"]
        self.fitness_function_map = FitnessFunctionMap()
        self.scale = config["scale"]

    def eval(self, note_sequence, melody_specs):
        fitness = 0
        for function_set in self.function_sets:
            input_vars = {
                "name":  function_set["name"],
                "weight": function_set["weight"],
                "chord": melody_specs.chord,
                "key": melody_specs.key,
                "note_sequence": note_sequence,
            }
            fitness += self.fitness_function_map.eval(input_vars)

        return fitness
