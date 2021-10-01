from glob import glob
from unittest import TestCase

from src.Data.AIHeroData import AIHeroData


class TestAIHeroData(TestCase):
    def setUp(self):
        self.ai_hero_data = AIHeroData()
        self.ai_hero_data.load_from_midi_files(glob("resources/Blues_Licks_-_Exercice_inhabituel*"))

    def test_image_plot(self):
        data = self.ai_hero_data.get_data()
        piano_roll = self.ai_hero_data.get_piano_roll()
        piano_roll_matrix = self.ai_hero_data.get_piano_roll_as_matrix()
        self.ai_hero_data.generate_piano_roll_image()

    def test_composition_to_data_conversion(self):
        previous_data = self.ai_hero_data.get_data()
        self.ai_hero_data.generate_piano_roll_image()
        converted_composition = self.ai_hero_data.get_mingus_composition()
        self.ai_hero_data.set_mingus_composition(converted_composition)
        converted_data = self.ai_hero_data.get_data()
        self.assertEqual(str(previous_data), str(converted_data))

    def test_composition_to_piano_roll_conversion(self):
        previous_piano_roll = self.ai_hero_data.get_piano_roll()
        converted_composition = self.ai_hero_data.get_mingus_composition()
        self.ai_hero_data.set_mingus_composition(converted_composition)
        converted_piano_roll = self.ai_hero_data.get_piano_roll()
        self.assertEqual(str(previous_piano_roll), str(converted_piano_roll))

    def test_data_to_composition_conversion(self):
        previous_composition = self.ai_hero_data.get_mingus_composition()
        converted_data = self.ai_hero_data.get_data()
        self.ai_hero_data.set_data(converted_data)
        converted_composition = self.ai_hero_data.get_mingus_composition()
        self.assertEqual(str(previous_composition), str(converted_composition))

    def test_data_to_piano_roll_conversion(self):
        previous_piano_roll = self.ai_hero_data.get_piano_roll()
        converted_data = self.ai_hero_data.get_data()
        self.ai_hero_data.set_data(converted_data)
        converted_piano_roll = self.ai_hero_data.get_piano_roll()
        self.assertEqual(str(previous_piano_roll), str(converted_piano_roll))

    def test_load_from_GAN(self):
        pass
