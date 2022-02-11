import numpy as np

from GAN.engine.augmentation.FifthNoteAddStrategy import FifthNoteAddStrategy
from GAN.engine.augmentation.NoteJoinStrategy import NoteJoinStrategy
from GAN.engine.augmentation.OctaveChangeStrategy import OctaveChangeStrategy
from GAN.engine.augmentation.TimeChangeStrategy import TimeChangeStrategy
from utils.AIHeroGlobals import SCALED_NOTES_NUMBER, TIME_DIVISION


class AugmentationEngine:
    def __init__(self, strategy_pipeline=None):
        if strategy_pipeline is None:
            strategy_pipeline = []
        self.strategy_pipeline = []
        for strategy in strategy_pipeline:
            self.strategy_pipeline.append({"method": get_from_string(strategy["method"]), "factor": strategy["factor"]})

    def augment(self, data):
        for strategy in self.strategy_pipeline:
            augmented_data = self.augment_with_strategy(strategy["method"], data, strategy["factor"])
            data = self.add_data(data, augmented_data)
        return data

    def augment_with_strategy(self, strategy, data, factor):
        size = data.shape[0]
        augmented_data = np.zeros([size * factor, data.shape[1], data.shape[2], data.shape[3]])
        for i in range(size):
            spr = data[i, :, :, 0]
            for j in range(factor):
                augmented_data[(j * size) + i, :, :, 0] = strategy.apply(spr)

        return augmented_data

    def add_data(self, data, new_data):
        total_size = data.shape[0] + new_data.shape[0]
        aggr_data = np.zeros([total_size, SCALED_NOTES_NUMBER, TIME_DIVISION, 1])
        aggr_data[0:data.shape[0], :, :, :] = data
        aggr_data[data.shape[0]:, :, :, :] = new_data
        return aggr_data


def get_from_string(strategy_string):
    if strategy_string == "OctaveChangeStrategy":
        return OctaveChangeStrategy()
    elif strategy_string == "TimeChangeStrategy":
        return TimeChangeStrategy()
    elif strategy_string == "NoteJoinStrategy":
        return NoteJoinStrategy()
    elif strategy_string == "FifthNoteAddStrategy":
        return FifthNoteAddStrategy()
    else:
        return None
