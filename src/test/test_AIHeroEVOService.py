from unittest import TestCase

from src.model.ApiModels import FitnessFunction
from src.service.AIHeroService import AIHeroService
from src.test.utils.test_utils import build_big_request_body, get_config, build_request_body, add_chord


class TestAIHeroService(TestCase):
    _one_time_setup_complete = False

    def before_running_all_tests(self):
        self.ai_hero_service = AIHeroService(get_config())
        self._one_time_setup_complete = True

    def setUp(self):
        if not self._one_time_setup_complete:
            self.before_running_all_tests()

    def test_make_a_blues_sequence_from_train_data(self):
        harmony_specs = build_big_request_body().melody_specs.harmony_specs
        data = self.ai_hero_service.generate_compositions_with_train_data(harmony_specs)

        # add a chord being played on every SPR
        data.execute_function_on_data(add_chord)

        data.export_as_midi("src/test/results/test_make_a_blues_sequence_from_train_data")

    def test_make_a_blues_sequence_from_gan_data(self):
        harmony_specs = build_big_request_body().melody_specs.harmony_specs
        data = self.ai_hero_service.generate_GAN_compositions(harmony_specs, melody_id="id")

        # add a chord being played on every SPR
        data.execute_function_on_data(add_chord)

        data.export_as_midi("src/test/results/test_make_a_blues_sequence_from_train_data")

    def test_make_a_blues_sequence_from_evo_data(self):
        input = build_request_body("evo")
        data = self.ai_hero_service.generate_compositions(input.melody_specs.harmony_specs,
                                                          input.melody_specs.evolutionary_specs)

        # add a chord being played on every SPR
        data.execute_function_on_data(add_chord)

        data.export_as_midi("src/test/results/test_make_a_blues_sequence_from_evo_data")

    def test_make_a_blues_sequence_from_train_data_with_blues_base(self):
        input = build_request_body()
        harmony_specs = input.melody_specs.harmony_specs
        data = self.ai_hero_service.generate_compositions_with_train_data(harmony_specs, melody_id="id")

        data.append_track_and_export_as_midi(midi_file="src/test/resources/blues_base_0.mid",
                                             file_name="src/test/results/test_make_a_blues_sequence_from_train_data_with_blues_base")

    def test_make_a_blues_sequence_from_gan_data_with_blues_base(self):
        input = build_request_body()
        harmony_specs = input.melody_specs.harmony_specs
        data = self.ai_hero_service.generate_GAN_compositions(harmony_specs, melody_id="id")

        data.append_track_and_export_as_midi(midi_file="src/test/resources/blues_base_0.mid",
                                             file_name="src/test/results/test_make_a_blues_sequence_from_gan_data_with_blues_base")

    def test_make_a_blues_sequence_from_evo_data_with_blues_base_no_pitch_control(self):
        input = build_request_body("evo")
        data = self.ai_hero_service.generate_compositions(input.melody_specs.harmony_specs,
                                                          input.melody_specs.evolutionary_specs)

        data.append_track_and_export_as_midi(midi_file="src/test/resources/blues_base_0.mid",
                                             file_name="src/test/results/test_make_a_blues_sequence_from_evo_data_with_blues_base_no_pitch_control")

    def test_make_a_blues_sequence_from_evo_data_with_blues_base_low_pitch(self):
        input = build_request_body("evo")
        input.melody_specs.evolutionary_specs.append(
            FitnessFunction(key="pitch_proximity", name="", description="", weight=2, value=42))
        data = self.ai_hero_service.generate_compositions(input.melody_specs.harmony_specs,
                                                          input.melody_specs.evolutionary_specs)

        data.append_track_and_export_as_midi(midi_file="src/test/resources/blues_base_0.mid",
                                             file_name="src/test/results/test_make_a_blues_sequence_from_evo_data_with_blues_base_low_pitch")

    def test_make_a_blues_sequence_from_evo_data_with_blues_base_middle_pitch(self):
        input = build_request_body("evo")
        input.melody_specs.evolutionary_specs.append(
            FitnessFunction(key="pitch_proximity", name="", description="", weight=2, value=60))
        data = self.ai_hero_service.generate_compositions(input.melody_specs.harmony_specs,
                                                          input.melody_specs.evolutionary_specs)

        data.append_track_and_export_as_midi(midi_file="src/test/resources/blues_base_0.mid",
                                             file_name="src/test/results/test_make_a_blues_sequence_from_evo_data_with_blues_base_middle_pitch")

    def test_make_a_blues_sequence_from_evo_data_with_blues_base_high_pitch(self):
        input = build_request_body("evo")
        input.melody_specs.evolutionary_specs.append(
            FitnessFunction(key="pitch_proximity", name="", description="", weight=2, value=72))
        data = self.ai_hero_service.generate_compositions(input.melody_specs.harmony_specs,
                                                          input.melody_specs.evolutionary_specs)

        data.append_track_and_export_as_midi(midi_file="src/test/resources/blues_base_0.mid",
                                             file_name="src/test/results/test_make_a_blues_sequence_from_evo_data_with_blues_base_high_pitch")
