import random
import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.transform import resize
from scipy.linalg import sqrtm

from src.data.AIHeroData import AIHeroData


class FIDQualityModel:  # quality model using FID (Frechet Inception Distance) metric
    def __init__(self):
        self._inception_model_shape = (299, 299, 3)
        self._fid_model = InceptionV3(include_top=False, pooling='avg', input_shape=self._inception_model_shape)

    # @tf.function
    def calculate_quality(self, data1, data2):
        scaled_data_1 = self.scale_data(data1)
        scaled_data_2 = self.scale_data(data2)
        ready_data1 = preprocess_input(scaled_data_1)
        ready_data2 = preprocess_input(scaled_data_2)

        # calculate activations
        act1 = self._fid_model.predict(ready_data1)
        act2 = self._fid_model.predict(ready_data2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        return fid

    def calculate_qualities(self, dataset1: AIHeroData, dataset2: AIHeroData):
        a = dataset1.get_spr_as_matrix()
        b = dataset2.get_spr_as_matrix()
        fid_array = []
        sample_size = min(a.shape[0], b.shape[0])
        batch_size = int(sample_size * 0.2)
        num_evaluations = 200
        a_sequence = np.arange(sample_size)
        b_sequence = np.arange(sample_size)
        for _ in range(num_evaluations):
            total_percent = (_ + 1) / num_evaluations * 100
            random.shuffle(a_sequence)
            random.shuffle(b_sequence)
            fid_array.append(self.calculate_quality(a[a_sequence[0:batch_size], :, :, 0],
                                                    b[b_sequence[0:batch_size], :, :, 0]))
            print(f'\r FID CALCULATION PROGRESS: [{"#" * int(total_percent):<100}] {total_percent:.1f}%', end='')
        print('')
        return fid_array

    def scale_data(self, data):
        images_list = list()
        for image in data:
            # resize with nearest neighbor interpolation
            new_image = resize(image, self._inception_model_shape, 0)
            # store
            images_list.append(new_image)
        return np.asarray(images_list)
