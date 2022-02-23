import numpy as np
from EVO.engine.FitnessFunctionMap import FitnessFunctionMap


class Fitness:
    def __init__(self, config):
        self.function_sets = config["function_sets"]
        self.fitness_function_map = FitnessFunctionMap()
        self.scale = config["scale"]

    def eval(self, note_sequence, melody_specs):
        fitness = 0
        fitness_per_function = np.zeros(len(self.function_sets))
        i = 0
        for function_set in self.function_sets:
            if function_set["weight"] != 0:
                input_vars = {
                    "name":  function_set["name"],
                    "weight": function_set["weight"],
                    "chord": melody_specs.chord,
                    "key": melody_specs.key,
                    "note_sequence": note_sequence,
                }
                value = self.fitness_function_map.eval(input_vars)
                fitness_per_function[i] = value
                i += 1
                fitness += value

        return fitness, fitness_per_function

    def get_function_names(self):
        return self.fitness_function_map.keys()

    def update_configs(self, evo_specs):
        new_set = []
        for spec in evo_specs:
            new_set.append({"name": spec.key, "weight": spec.value})
        self.function_sets = new_set
