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
from src.utils.AIHeroEnums import MelodicPart
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER, TRAIN_DATA_REPLICAS, SCALED_NOTES_RANGE


class AIHeroGAN:
    def __init__(self, part=MelodicPart.X, checkpoint_folder='GAN/data/training_checkpoints', verbose=False):
        # todo: fazer alguma verificação e validação para que o part seja sempre um valor do enum
        self.part_type = part.value

        # training
        self.noise_dim = 100  # todo: experiment other values
        self.num_examples_to_generate = 1  # number of melodies to be generated
        self.BATCH_PERCENTAGE = 0.1
        self.BATCH_SIZE = 25  # this will be subscribed
        self.BUFFER_PERCENTAGE = 0.2

        self.training_data = GANTrainingData(melodic_part=part)

        # Private Variables
        self._trained = False
        self._verbose = verbose

        self.evidence_dir = 'data/evidences/gifs'
        self.checkpoint_dir = f'{checkpoint_folder}/part_{self.part_type}'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        self.generator_model = self.make_generator_model()
        self.discriminator_model = self.make_discriminator_model()

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator_model,
                                              discriminator=self.discriminator_model)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        self.load_from_checkpoint()  # todo: pensar onde essa lógica do load checkpoint vai ficar

    def load_from_checkpoint(self):
        print(f"loading checkpoint for gan of part {self.part_type}...")
        load_status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        try:
            load_status.assert_existing_objects_matched()
            there_is_a_checkpoint = True
        except:
            there_is_a_checkpoint = False

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
        # TODO: entender melhor estes parametros

        # rnn com lstm
        # fazer one hot encode
        # ver proprio site do keras
        # https://colah.github.io/posts/2015-08-Understanding-LSTMs/

        # # Add an Embedding layer expecting input vocab of size 1000, and
        # # output embedding dimension of size 64.
        # model.add(layers.Embedding(input_length=32, input_dim=SCALED_NOTES_NUMBER + 1, output_dim=64))
        #
        # # Add a LSTM layer with 128 internal units.
        # model.add(layers.LSTM(128))
        #
        # # Add a Dense layer with 10 units.
        # model.add(layers.Dense(10))

        layer_1_finger = max(int(SCALED_NOTES_NUMBER / 2), 1)
        layer_1_fuse = max(int(TIME_DIVISION / 2), 1)
        layer_2_finger = max(int(SCALED_NOTES_NUMBER / 4), 1)
        layer_2_fuse = max(int(TIME_DIVISION / 4), 1)
        model = tf.keras.Sequential()
        # adicionar one hot encoder?
        model.add(layers.Dense(layer_2_finger * layer_2_fuse * 256, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((layer_2_finger, layer_2_fuse, 256)))
        assert model.output_shape == (None, layer_2_finger, layer_2_fuse, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, layer_2_finger, layer_2_fuse, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, layer_1_finger, layer_1_fuse, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, SCALED_NOTES_NUMBER, TIME_DIVISION, 1)

        if self.should_verbose():
            model.summary()

        return model

    def make_discriminator_model(self):
        # TODO: entender melhor!
        # model.add(layers.Embedding(input_length=32, input_dim=SCALED_NOTES_NUMBER + 1, output_dim=64))
        # model.add(layers.LSTM(128))
        # # Add a Dense layer with 10 units.
        # model.add(layers.Dense(10))

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

    def train(self, num_seeds=1, epochs=50, should_generate_gif=False):
        self.num_examples_to_generate = num_seeds
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        try:
            dataset = self.training_data.get_as_matrix()
            dataset = np.repeat(dataset, TRAIN_DATA_REPLICAS, axis=0)  # DATASET AUGMENTATION

            BUFFER_SIZE = np.int(dataset.shape[0] * self.BUFFER_PERCENTAGE)
            self.BATCH_SIZE = np.int(dataset.shape[0] * self.BATCH_PERCENTAGE)
            train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(self.BATCH_SIZE)

            for epoch in range(epochs):
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
                if (epoch + 1) % 15 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

                if should_generate_gif:
                    self.generate_and_save_images(epoch)

                if self.should_verbose():
                    print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

            # Generate after the final epoch
            if should_generate_gif:
                display.clear_output(wait=True)
                self.generate_and_save_images(epochs)
                self.generate_gif()

                # erase temporary images
                for f in glob.glob('.temp/image*.png'):
                    os.remove(f)

        except Exception as e:
            print(f"Failed training gan of type {self.part_type}: {e}")
            print(traceback.format_exc())

    def generate_and_save_images(self, epoch, new_seed=False):
        predictions = self.generate_prediction(new_seed)
        # fig = plt.figure(figsize=(4, 4))
        #
        # for i in range(predictions.shape[0]):
        #     plt.imshow(predictions[i, :, :, 0], cmap='gray')
        num_bars = predictions.shape[0]
        concat_data = np.ndarray((SCALED_NOTES_NUMBER, TIME_DIVISION * num_bars))
        a = 0
        b = TIME_DIVISION
        for i in range(num_bars):
            concat_data[:, a:b] = predictions[i, :, :, 0]
            a = b
            b = b + TIME_DIVISION
        plt.imshow(concat_data, cmap='Blues')
        plt.axis([0, num_bars * TIME_DIVISION, SCALED_NOTES_RANGE[0],
                  SCALED_NOTES_RANGE[1]])  # necessary for inverting y axis
        plt.ylabel("MIDI Notes")
        plt.xlabel("Time Division")
        # plt.text(1, 90, f'Loss G: {results["generator_loss"]}, Loss D: {results["discriminator_loss"]}')
        plt.savefig('.temp/image_at_epoch_{:04d}.png'.format(epoch))

    def generate_gif(self):
        # def display_image(epoch_no):
        #     return PIL.Image.open('.temp/image_at_epoch_{:04d}.png'.format(epoch_no))
        #
        # display_image(epochs)
        today = date.today()
        anim_file = f'{self.evidence_dir}/{self.part_type}_{today.strftime("%Y%m%d")}_{time.time_ns()}.gif'

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
        return melody.numpy()

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