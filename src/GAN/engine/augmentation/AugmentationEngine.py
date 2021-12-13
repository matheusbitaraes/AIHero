import numpy as np

from src.utils.AIHeroGlobals import SCALED_NOTES_NUMBER, TIME_DIVISION


class AugmentationEngine:
    def __init__(self, strategy_list=None, augmentation_size=1):
        if strategy_list is None:
            strategy_list = []
        self.strategy_list = strategy_list
        self.augmentation_size = augmentation_size

    def augment(self, data):
        data_list = []
        data_size = 0
        for strategy in self.strategy_list:
            augmented_data = self.augment_with_strategy(strategy, data)
            data_size += augmented_data.shape[0]
            data_list.append(augmented_data)

        output_data = np.zeros([data_size, SCALED_NOTES_NUMBER, TIME_DIVISION, 1])
        i = 0
        for data in data_list:
            data_len = data.shape[0]
            output_data[i:i + data_len, :, :, :] = data
            i = i + data_len
        return output_data

    def augment_with_strategy(self, strategy, data):
        size = data.shape[0]
        augmented_data = np.zeros([size * self.augmentation_size, data.shape[1], data.shape[2], data.shape[3]])
        for i in range(size):
            spr = data[i, :, :, 0]
            for j in range(self.augmentation_size):
                augmented_data[(j * size) + i, :, :, 0] = strategy.apply(spr)

        return augmented_data
