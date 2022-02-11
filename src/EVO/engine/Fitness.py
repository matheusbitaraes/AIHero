from EVO.engine.FitnessFunctionMap import FitnessFunctionMap


class Fitness:
    def __init__(self, config):
        self.function_sets = config["function_sets"]
        self.scale = config["scale"]

    def eval(self, note_sequence, melody_specs):
        fitness = 0
        for function_set in self.function_sets:
            input_vars = {
                "chord": melody_specs.chord,
                "key": melody_specs.key,
                "note_sequence": note_sequence,
                "weight": function_set["weight"]
            }
            function = FitnessFunctionMap(function_set["name"])
            fitness += function.eval(input_vars)

        return fitness
