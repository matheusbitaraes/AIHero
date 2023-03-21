import numpy as np
from src.EVO.engine.FitnessFunctionMap import FitnessFunctionMap


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
                    "value": function_set["value"],
                    "chord": melody_specs.transposition_factor,
                    "key": melody_specs.key,
                    "note_sequence": note_sequence,
                }
                value = self.fitness_function_map.eval(input_vars)
                fitness_per_function[i] = value
                fitness += value
            i += 1

        return fitness, fitness_per_function

    def get_function_names(self):
        fitness_function_names = []
        for function_set in self.function_sets:
            fitness_function_names.append(f'{function_set["name"]} ({function_set["weight"]:.2f})')

        return fitness_function_names

    def update_configs(self, evo_specs):
        new_set = []
        for spec in evo_specs:
            new_set.append({"name": spec.key, "weight": spec.weight, "value": spec.value})
        self.function_sets = new_set

    def get_maximum_possible_value(self):
        fitness = 0
        for function_set in self.function_sets:
            if function_set["weight"] > 0:
                fitness += function_set["weight"]
        return fitness
