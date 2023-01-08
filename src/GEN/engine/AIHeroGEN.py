import glob
import os
import random
import time
from abc import ABC, abstractmethod
from datetime import date

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

from src.GEN.data.GANTrainingData import GANTrainingData
from src.GEN.engine.convolutionalGAN.Discriminator import Discriminator
from src.GEN.engine.convolutionalGAN.Generator import Generator
from src.quality.FID.FIDQualityModel import FIDQualityModel
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER, SCALED_NOTES_RANGE
from src.utils.AIHeroHelper import HarmonicFunction

matplotlib.use('Agg')


class AIHeroGEN(ABC):
    def __init__(self, config: dict, harmonic_function: HarmonicFunction = HarmonicFunction.TONIC):
        self._model_name = ""
        self.harmonic_function = harmonic_function

        # placeholder models
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

        # training
        self.noise_dim = config["training"]["noise_dim"]
        self.num_examples_to_generate = config["training"][
            "num_examples_to_generate"]  # number of melodies to be generated
        self.BATCH_PERCENTAGE = config["training"]["batch_percentage_alt"]
        self.BATCH_SIZE = config["training"]["max_batch_size"]
        self.BUFFER_PERCENTAGE = config["training"]["buffer_percentage_alt"]
        self.BUFFER_SIZE = config["training"]["max_buffer_size"]
        self.num_epochs = config["training"]["num_epochs"]

        self.training_data = GANTrainingData(config, harmonic_function=harmonic_function)

        # Private Variables
        self._trained = False
        self._verbose = config["verbose"]
        self._should_use_checkpoint = config["checkpoint"]["use_checkpoint"]
        self._generator_losses = []
        self._discriminator_losses = []

        self.gifs_evidence_dir = config["generated_evidences_dir"]
        self.checkpoint_dir = f'{config["checkpoint"]["checkpoint_folder"]}/{self.harmonic_function.name}'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        # quality metric
        self._live_quality_polling_enabled = config["training"]["enable_live_quality_measures"]
        self._quality_model = FIDQualityModel()
        self._epochs_for_quality_measure = config["training"]["epochs_for_quality_measure"]
        self._max_samples_for_quality_measure = config["training"]["max_samples_for_quality_measure"]
        self._quality_target_value = config["training"]["quality_measure_target_value"]
        self._quality_measures = []

        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

        # overwrite arguments
        self.checkpoint_dir = f'{config["checkpoint"]["checkpoint_folder"]}/{self._model_name}/{self.harmonic_function.name}'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator.optimizer,
                                              discriminator_optimizer=self.discriminator.optimizer,
                                              generator=self.generator.model,
                                              discriminator=self.discriminator.model)

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def train(self, should_generate_gif: bool, prefix: str, num_epochs: int):
        pass

    def clear(self):
        tf.keras.backend.clear_session()

    def evaluate_with_discriminator(self, data):
        return self.discriminator.model(data)

    def generate_melody_matrix(self, num_melodies: int, new_seed: bool):
        predictions = self.generate_prediction(new_seed, size=num_melodies)
        np_config.enable_numpy_behavior()
        melody = tf.Variable(predictions)
        for i in range(melody.shape[0]):
            melody[i, :, :, 0].assign(np.where(melody[i, :, :, 0] > 0.9, 1, -1))
        return melody.numpy()

    @abstractmethod
    def load_from_checkpoint(self):
        pass

    def set_verbose(self, value: bool):
        self._verbose = value

    def should_verbose(self):
        return self._verbose

    def is_trained(self):
        return self._trained

    def generate_and_save_images(self, epoch, current_time_min, new_seed=False, harmonic_function=0):
        predictions = self.generate_prediction(new_seed)
        num_bars = predictions.shape[0]
        concat_data = np.ndarray((SCALED_NOTES_NUMBER, TIME_DIVISION * num_bars))
        a = 0
        b = TIME_DIVISION
        for i in range(num_bars):
            concat_data[:, a:b] = predictions[i, :, :, 0]
            a = b
            b = b + TIME_DIVISION

        if self._live_quality_polling_enabled:
            fig, axs = plt.subplots(3)
        else:
            fig, axs = plt.subplots(2)
        fig.suptitle(f'Training progress for epoch {epoch} hm({harmonic_function}) ({round(current_time_min, 2)} min)')

        # midi plot
        axs[0].imshow(concat_data, cmap='Blues')
        axs[0].axis([0, num_bars * TIME_DIVISION, SCALED_NOTES_RANGE[0],
                     SCALED_NOTES_RANGE[1]])  # necessary for inverting y axis
        axs[0].set(xlabel='Time Division', ylabel='MIDI Notes')

        # losses plot
        max_epochs = 50
        begin = max(0, epoch - max_epochs)
        end = epoch
        axs[1].plot(range(begin, end), self._generator_losses[begin:end])
        axs[1].plot(range(begin, end), self._discriminator_losses[begin:end])
        axs[1].legend(["Generator.py Loss: {:03f}".format(self._generator_losses[-1]),
                       "Discriminator Loss {:03f}".format(self._discriminator_losses[-1])])
        axs[1].set(xlabel='Epochs', ylabel='Loss')

        # quality measure plot
        if self._live_quality_polling_enabled:
            num_measures = len(self._quality_measures)
            epoch_array = range(0, num_measures * self._epochs_for_quality_measure, self._epochs_for_quality_measure)
            axs[2].plot(epoch_array, self._quality_measures)
            if len(self._quality_measures):
                axs[2].legend(["FID indicator: {:03f}".format(self._quality_measures[-1])])
            axs[2].set(xlabel='Epochs', ylabel='FID')

        plt.savefig('.temp/image_at_epoch_{:01f}_{:04d}.png'.format(harmonic_function, epoch))
        plt.close()

    def generate_gif(self, filename_prefix=""):
        today = date.today()
        anim_file = f'{self.gifs_evidence_dir}/{filename_prefix}{self.harmonic_function.name}_' \
                    f'{today.strftime("%Y%m%d")}_{time.time_ns()}.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('.temp/image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

    @abstractmethod
    def generate_prediction(self, new_seed: bool = False, size: int = 1):
        pass

    def get_random_train_data(self, number=1):
        train_data = self.training_data.get_as_matrix()
        index = random.sample(range(train_data.shape[0]), number)[0]
        output = np.zeros((1, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
        output[0:number, :, :, :] = train_data[index, :, :, :]
        return output

    @property
    def generator_losses(self):
        return self._generator_losses

    @property
    def discriminator_losses(self):
        return self._discriminator_losses

    def calculate_quality_measure(self):
        real_data = self.training_data.get_as_matrix()
        training_size = real_data.shape[0]
        size = min(training_size, self._max_samples_for_quality_measure)
        fake_data = self.generate_prediction(new_seed=True, size=size)

        return self._quality_model.calculate_quality(real_data[0:size, :, :, :], np.array(fake_data))

    @property
    def quality_measures(self):
        return self._quality_measures

    def harmonic_function_to_binary(self, hm=None):
        table = {
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            1: [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
            2: [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            3: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        return table[hm]

    def build_seed(self):
        noise = np.random.normal(size=(self.num_examples_to_generate, self.noise_dim))
        for i in range(self.num_examples_to_generate):
            if 0 < i < 4:
                hm = i
            else:
                hm = 1
            hm_bin = self.harmonic_function_to_binary(hm)
            noise[i, 0:len(hm_bin)] = hm_bin
        noise = tf.Variable(noise)
        self.seed = noise
