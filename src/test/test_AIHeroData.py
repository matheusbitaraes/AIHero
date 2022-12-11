from unittest import TestCase

from src.data.AIHeroData import AIHeroData
from src.test.utils.test_utils import add_chord


class TestAIHeroData(TestCase):

    def test_load_pop909_spr_and_with_chord_on_different_keys(self):
        # load a file
        data = AIHeroData()
        data.load_from_pop909_dataset(dataset_path="src/test/resources/pop909-subsample/")

        # add a chord being played on every SPR
        data.execute_function_on_data(add_chord)

        # export file
        data.export_as_midi("src/test/results/test_load_pop909_spr_and_with_chord_on_different_keys")
        assert 1 == 1

    def test_load_manual_spr_and_with_chord_on_different_keys(self):
        # load a file
        data = AIHeroData()
        data.load_spr_from_checkpoint("src/GEN/data/train/manual")

        # add a chord being played on every SPR
        data.execute_function_on_data(add_chord)

        # export file
        data.export_as_midi("src/test/results/test_load_manual_spr_and_with_chord_on_different_keys")
        assert 1 == 1


