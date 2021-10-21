from glob import glob
from unittest import TestCase

from src.data.AIHeroData import AIHeroData

TRAIN_FILES_DIRECTORY = "resources/part_X_manual_3_C_*"

class TestAIHeroData(TestCase):
    def setUp(self):
        self.ai_hero_data = AIHeroData()
        self.ai_hero_data.load_from_midi_files(glob(TRAIN_FILES_DIRECTORY))

    def test_image_plot(self):
        self.ai_hero_data.export_spr_as_image(file_name="image_export_test")

    def test_midi_export(self):
        self.ai_hero_data.export_as_midi(file_name="loaded_mingus_test")

    def test_composition_to_data_conversion(self):  # todo: create "expected" value
        previous, converted = test_conversion(self.ai_hero_data, "mingus_composition", "data")
        self.assertEqual(str(previous), str(converted))

    def test_composition_to_pr_conversion(self):  # todo: create "expected" value
        previous, converted = test_conversion(self.ai_hero_data, "mingus_composition", "pr")
        self.assertEqual(str(previous), str(converted))

    def test_composition_to_spr_conversion(self):  # todo: create "expected" value
        previous, converted = test_conversion(self.ai_hero_data, "mingus_composition", "spr")
        self.assertEqual(str(previous), str(converted))

    def test_pr_to_composition_conversion(self):
        previous, converted = test_conversion(self.ai_hero_data, "pr", "mingus_composition")
        self.assert_compositions(previous, converted)

    def test_spr_to_composition_conversion(self):
        previous, converted = test_conversion(self.ai_hero_data, "spr", "mingus_composition")
        self.assert_compositions(previous, converted)

    def test_data_to_composition_conversion(self):
        previous, converted = test_conversion(self.ai_hero_data, "data", "mingus_composition")
        self.assert_compositions(previous, converted)

    def test_pr_to_spr_conversion(self):
        previous, converted = test_conversion(self.ai_hero_data, "pr", "spr")
        self.assertEqual(str(previous), str(converted))

    def test_spr_to_pr_conversion(self):
        previous, converted = test_conversion(self.ai_hero_data, "spr", "pr")
        self.assertEqual(str(previous), str(converted))

    def test_load_from_GAN(self):
        pass

    def assert_compositions(self, previous_composition, converted_composition):
        for i in range(len(previous_composition)):
            pc = previous_composition[i]
            cc = converted_composition[i]
            for j in range(len(pc)):
                pt = pc[j]
                ct = cc[j]
                for k in range(len(pt)):
                    pbs = pt[k].bars
                    cbs = ct[k].bars
                    if bars_with_note(pbs) and bars_with_note(cbs):
                        for l in range(len(pbs)):
                            self.assertEqual(str(pbs[l]), str(cbs[l]))


def test_conversion(ai_hero_data, from_type="mingus_composition", to_type="pr"):
    previous_value = None
    converted_value = None

    if to_type == "data":
        previous_value = ai_hero_data.get_data()
    if to_type == "mingus_composition":
        previous_value = ai_hero_data.get_mingus_compositions()
    if to_type == "pr":
        previous_value = ai_hero_data.get_pr()
    if to_type == "spr":
        previous_value = ai_hero_data.get_spr()

    if from_type == "data":
        ai_hero_data.set_data(ai_hero_data.get_data())
    if from_type == "mingus_composition":
        ai_hero_data.set_mingus_compositions(ai_hero_data.get_mingus_compositions())
    if from_type == "pr":
        ai_hero_data.set_pr(ai_hero_data.get_pr())
    if from_type == "spr":
        ai_hero_data.set_spr(ai_hero_data.get_spr())

    if to_type == "data":
        converted_value = ai_hero_data.get_data()
    if to_type == "mingus_composition":
        converted_value = ai_hero_data.get_mingus_compositions()
    if to_type == "pr":
        converted_value = ai_hero_data.get_pr()
    if to_type == "spr":
        converted_value = ai_hero_data.get_spr()
    return previous_value, converted_value


def bars_with_note(bars):
    if len(bars) == 0:
        return False
    for bar in bars:
        if len(bar) > 0:
            return True
