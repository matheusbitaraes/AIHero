import glob
import os
import random
import time
import traceback
from datetime import date

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config

from src.GAN.data.GANTrainingData import GANTrainingData
from src.GAN.engine.quality.FIDQualityModel import FIDQualityModel
from src.utils.AIHeroEnums import MelodicPart
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER, SCALED_NOTES_RANGE


class AIHeroGAN:
    def __init__(self, config, part=MelodicPart.X):
        # todo: fazer alguma verificação e validação para que o part seja sempre um valor do enum
        self.part_type = part.value

        # training
        self.noise_dim = config["training"]["noise_dim"]  # todo: experiment other values
        self.num_examples_to_generate = config["training"][
            "num_examples_to_generate"]  # number of melodies to be generated
        self.BATCH_PERCENTAGE = config["training"]["batch_percentage_alt"]
        self.BUFFER_PERCENTAGE = config["training"]["buffer_percentage"]
        self.BATCH_SIZE = config["training"]["max_batch_size"]
        self.num_epochs = config["training"]["num_epochs"]

        self.training_data = GANTrainingData(config, melodic_part=part)

        # Private Variables
        self._trained = False
        self._verbose = config["verbose"]
        self._should_use_checkpoint = config["checkpoint"]["use_checkpoint"]
        self._generator_losses = []
        self._discriminator_losses = []

        self.gifs_evidence_dir = config["generated_evidences_dir"]
        self.checkpoint_dir = f'{config["checkpoint"]["checkpoint_folder"]}/part_{self.part_type}'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        # quality metric
        self._live_quality_polling_enabled = config["training"]["enable_live_quality_measures"]
        self._quality_model = FIDQualityModel()
        self._epochs_for_quality_measure = config["training"]["epochs_for_quality_measure"]
        self._max_samples_for_quality_measure = config["training"]["max_samples_for_quality_measure"]
        self._quality_measures = []

        self.generator_model = self.make_generator_model()
        self.discriminator_model = self.make_discriminator_model()

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator_model,
                                              discriminator=self.discriminator_model)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

        if self._should_use_checkpoint:
            self.load_from_checkpoint()

        # if tf.config.list_physical_devices('GPU'):
        #     physical_devices = tf.config.list_physical_devices('GPU')
        #     tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
            # tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
            #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])

    def load_from_checkpoint(self):
        if self.should_verbose():
            print(f"loading checkpoint for gan of part {self.part_type}...")
        load_status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        try:
            load_status.assert_existing_objects_matched()
            there_is_a_checkpoint = True
        except:
            there_is_a_checkpoint = False

        if self.should_verbose():
            if there_is_a_checkpoint:
                print("Checkpoint loaded!")
            else:
                print("no checkpoint found")

    def set_verbose(self, value):
        self._verbose = value

    def should_verbose(self):
        return self._verbose

    def is_trained(self):
        return self._trained

    def make_generator_model(self):
        layer_1_finger = max(int(SCALED_NOTES_NUMBER / 2), 1)
        layer_1_fuse = max(int(TIME_DIVISION / 2), 1)
        layer_2_finger = max(int(SCALED_NOTES_NUMBER / 4), 1)
        layer_2_fuse = max(int(TIME_DIVISION / 4), 1)
        model = tf.keras.Sequential()

        # https://analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/#:~:text=In%20any%20neural%20network%2C%20a,in%20artificial%20neural%20network%20networks.
        # Dense layer: Neurons of the layer that is deeply connected with its preceding layer which means the neurons
        # of the layer are connected to every neuron of its preceding layer.
        model.add(layers.Dense(layer_2_finger * layer_2_fuse * 256, use_bias=False, input_shape=(self.noise_dim,)))

        ## Normalization layer
        model.add(layers.BatchNormalization())

        # Leaky Rectified Linear Unit (ReLU) layer: A leaky ReLU layer performs a threshold operation, where any
        # input value less than zero is multiplied by a fixed scalar.
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((layer_2_finger, layer_2_fuse, 256)))
        assert model.output_shape == (None, layer_2_finger, layer_2_fuse, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, layer_2_finger, layer_2_fuse, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, layer_1_finger, layer_1_fuse, 64)

        # model.add(layers.Dense(64))
        model.add(layers.LeakyReLU())
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, SCALED_NOTES_NUMBER, TIME_DIVISION, 1)

        if self.should_verbose():
            model.summary()

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[SCALED_NOTES_NUMBER, TIME_DIVISION, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        if self.should_verbose():
            model.summary()

        return model

    def discriminator_loss(self, real_output, fake_output):
        # Este método quantifica o quão bem o discriminador é capaz de distinguir
        # imagens reais de falsificações. Ele compara as previsões do discriminador
        # em imagens reais a uma matriz de 1s e as previsões do discriminador em
        # imagens falsas (geradas) a uma matriz de 0s.
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        # computes cross entropy between fake output and best output (which is everything = 1)
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator_model(noise, training=True)

            real_output = self.discriminator_model(images, training=True)
            fake_output = self.discriminator_model(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator_model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator_model.trainable_variables))

        return {
            "discriminator_loss": disc_loss,
            "generator_loss": gen_loss
        }

    def train(self, should_generate_gif=False, prefix=""):
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        try:
            dataset = self.training_data.get_as_matrix()

            BUFFER_SIZE = np.int(dataset.shape[0] * self.BUFFER_PERCENTAGE)

            # O IDEAL É O ABAIXO - APAGANDO TEMPORARIAMENTE PARA TESTE
            # self.BATCH_SIZE = min(self.BATCH_SIZE, np.int(np.int(np.ceil(dataset.shape[0] * self.BATCH_PERCENTAGE))))

            train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(self.BATCH_SIZE)

            current_time_min = 0
            for epoch in range(self.num_epochs):
                start = time.time()
                results = None
                for melody_batch in train_dataset:
                    results = self.train_step(melody_batch)  # [ BATCH_SIZE, NUM_FINGERS, TIME_DIVISION, 1]
                    if self.should_verbose():
                        print(f'Loss G: {results["generator_loss"]},'
                              f' Loss D: {results["discriminator_loss"]}', end='\r')
                if self.should_verbose():
                    print(f'Loss G: {results["generator_loss"]},'
                          f' Loss D: {results["discriminator_loss"]}')

                # Save the model every 15 epochs
                if self._should_use_checkpoint and (epoch + 1) % 15 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

                # measure GAN quality: DISABLE THIS FOR BETTER PERFORMANCE
                if (epoch + 1) % self._epochs_for_quality_measure == 0:
                    self._quality_measures.append(self.calculate_quality_measure())

                self._generator_losses.append(float(results["generator_loss"]))
                self._discriminator_losses.append(float(results["discriminator_loss"]))

                current_time_min = current_time_min + (time.time() - start) / 60

                if should_generate_gif:
                    self.generate_and_save_images(epoch, current_time_min)

                if self.should_verbose():
                    print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

            # get last quality measure
            self._quality_measures.append(self.calculate_quality_measure())

            # Generate after the final epoch
            if should_generate_gif:
                display.clear_output(wait=True)
                self.generate_and_save_images(self.num_epochs, current_time_min)
                self.generate_gif(filename_prefix=prefix)

                # erase temporary images
                for f in glob.glob('.temp/*.png'):
                    os.remove(f)

        except Exception as e:
            print(f"Failed training gan of type {self.part_type}: {e}")
            print(traceback.format_exc())

    def generate_and_save_images(self, epoch, current_time_min, new_seed=False):
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
        fig.suptitle(f'Training progress for epoch {epoch} ({round(current_time_min, 2)} min)')

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
        axs[1].legend(["Generator Loss: {:03f}".format(self._generator_losses[-1]),
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

        plt.savefig('.temp/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def generate_gif(self, filename_prefix=""):
        today = date.today()
        anim_file = f'{self.gifs_evidence_dir}/{filename_prefix}{self.part_type}_{today.strftime("%Y%m%d")}_{time.time_ns()}.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('.temp/image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

    def generate_melody_matrix(self, num_melodies=1, new_seed=False):
        predictions = self.generate_prediction(new_seed, size=num_melodies)
        np_config.enable_numpy_behavior()
        melody = tf.Variable(predictions)
        for i in range(melody.shape[0]):
            melody[i, :, :, 0].assign(round(melody[i, :, :, 0]))
        return normalize_melody(melody.numpy())

    def generate_prediction(self, new_seed=False, size=1):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        if new_seed:
            seed = tf.random.normal([size, self.noise_dim])
            predictions = self.generator_model(seed, training=False)
        else:
            predictions = self.generator_model(self.seed, training=False)
        return predictions

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


def normalize_melody(melody):
    melody[melody != 1] = -1
    return melody
