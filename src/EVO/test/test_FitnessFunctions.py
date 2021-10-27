from glob import glob
from unittest import TestCase

from src.EVO.engine.FitnessFunctionMap import notes_on_same_chord_key, notes_on_beat_rate, intervals_percentage, \
    note_repetitions_rate, pitch_proximity_rate
from src.data.AIHeroData import AIHeroData


class TestFitnessFunctions(TestCase):
    # def setUp(self):

    def test_notes_on_same_chord_key(self):
        type_name = "notes_on_same_chord_key"
        func = notes_on_same_chord_key
        FITNESS_HIGH = [1, 1, 1]
        FITNESS_LOW = [0, 0]
        FITNESS_MEDIUM = [0.38095238095238093, 0.6333333333333333]
        fixed_input_values = {
            "weight": 1,
            "chord": "C",
            "key": "C"
        }
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_high*")),
            FITNESS_HIGH)
        self.assertEqual(test_function(function=func, input_dict=fixed_input_values,
                                       file_path=glob(f"resources/{type_name}_medium*")), FITNESS_MEDIUM)
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_low*")),
            FITNESS_LOW)

    def test_notes_on_beat_rate(self):
        type_name = "notes_on_beat_rate"
        func = notes_on_beat_rate
        FITNESS_HIGH = [1, 1, 1]
        FITNESS_LOW = [0, 0]
        FITNESS_MEDIUM = [0.38095238095238093, 0.6333333333333333]
        fixed_input_values = {
            "weight": 1,
            "chord": "C",
            "key": "C"
        }
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_high*")),
            FITNESS_HIGH)
        self.assertEqual(test_function(function=func, input_dict=fixed_input_values,
                                       file_path=glob(f"resources/{type_name}_medium*")), FITNESS_MEDIUM)
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_low*")),
            FITNESS_LOW)

    def test_intervals_percentage(self):
        type_name = "intervals_percentage"
        func = intervals_percentage
        FITNESS_HIGH = [1, 1, 1]
        FITNESS_LOW = [0, 0]
        FITNESS_MEDIUM = [0.38095238095238093, 0.6333333333333333]
        fixed_input_values = {
            "weight": 1,
            "chord": "C",
            "key": "C"
        }
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_high*")),
            FITNESS_HIGH)
        self.assertEqual(test_function(function=func, input_dict=fixed_input_values,
                                       file_path=glob(f"resources/{type_name}_medium*")), FITNESS_MEDIUM)
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_low*")),
            FITNESS_LOW)

    def test_note_repetitions_rate(self):
        type_name = "note_repetitions_rate"
        func = note_repetitions_rate
        FITNESS_HIGH = [1, 1, 1]
        FITNESS_LOW = [0, 0]
        FITNESS_MEDIUM = [0.38095238095238093, 0.6333333333333333]
        fixed_input_values = {
            "weight": 1,
            "chord": "C",
            "key": "C"
        }
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_high*")),
            FITNESS_HIGH)
        self.assertEqual(test_function(function=func, input_dict=fixed_input_values,
                                       file_path=glob(f"resources/{type_name}_medium*")), FITNESS_MEDIUM)
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_low*")),
            FITNESS_LOW)

    def test_pitch_proximity_rate(self):
        type_name = "pitch_proximity_rate"
        func = pitch_proximity_rate
        FITNESS_HIGH = [1, 1, 1]
        FITNESS_LOW = [0, 0]
        FITNESS_MEDIUM = [0.38095238095238093, 0.6333333333333333]
        fixed_input_values = {
            "weight": 1,
            "chord": "C",
            "key": "C"
        }
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_high*")),
            FITNESS_HIGH)
        self.assertEqual(test_function(function=func, input_dict=fixed_input_values,
                                       file_path=glob(f"resources/{type_name}_medium*")), FITNESS_MEDIUM)
        self.assertEqual(
            test_function(function=func, input_dict=fixed_input_values, file_path=glob(f"resources/{type_name}_low*")),
            FITNESS_LOW)


def test_function(function, input_dict, file_path):
    fitness_list = []
    dataset = AIHeroData()
    dataset.load_from_midi_files(file_path)
    input_data = dataset.get_spr_as_matrix()
    for i in range(input_data.shape[0]):
        ns = input_data[i, :, :, 0]
        input_dict["note_sequence"] = ns
        if len(ns[ns == 1]) > 0:
            fitness = function(input_dict)
            fitness_list.append(fitness)
    return fitness_list
