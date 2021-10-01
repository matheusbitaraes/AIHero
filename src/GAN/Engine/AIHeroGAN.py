import glob
import os
import time
from datetime import date

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config

from src.GAN.Data.GANTrainingData import GANTrainingData
from src.utils.AIHeroEnums import MelodicPart

MIDI_NOTES_NUMBER = 108


class AIHeroGAN:
    def __init__(self, part=MelodicPart.X, checkpoint_folder='GAN/Data/training_checkpoints'):
        self.NUM_FUSES = 32
        self.part_type = part.value  # todo: fazer alguma verificação e validação para que o part seja sempre um valor do enum

        # training
        self.noise_dim = 100
        self.num_examples_to_generate = 1  # numero de melodias que vai gerar
        self.BATCH_SIZE = 25
        self.BUFFER_SIZE = 25

        self.training_data = GANTrainingData(melodic_part=part)

        # Private Variables
        self.__trained = False
        self.__verbose = False

        self.evidence_dir = 'Data/evidences/gifs'
        self.checkpoint_dir = f'{checkpoint_folder}/part_{self.part_type}'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        self.generator_model = self.make_generator_model()
        self.discriminator_model = self.make_discriminator_model()

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

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
        self.__verbose = value

    def should_verbose(self):
        return self.__verbose

    def is_trained(self):
        return self.__trained

    def make_generator_model(self):
        # TODO: entender melhor estes parametros

        # rnn com lstm
        # fazer one hot encode
        # ver proprio site do keras
        # https://colah.github.io/posts/2015-08-Understanding-LSTMs/

        # # Add an Embedding layer expecting input vocab of size 1000, and
        # # output embedding dimension of size 64.
        # model.add(layers.Embedding(input_length=32, input_dim=MIDI_NOTES_NUMBER + 1, output_dim=64))
        #
        # # Add a LSTM layer with 128 internal units.
        # model.add(layers.LSTM(128))
        #
        # # Add a Dense layer with 10 units.
        # model.add(layers.Dense(10))

        layer_1_finger = max(int(MIDI_NOTES_NUMBER / 2), 1)
        layer_1_fuse = max(int(self.NUM_FUSES / 2), 1)
        layer_2_finger = max(int(MIDI_NOTES_NUMBER / 4), 1)
        layer_2_fuse = max(int(self.NUM_FUSES / 4), 1)
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
        assert model.output_shape == (None, MIDI_NOTES_NUMBER, self.NUM_FUSES, 1)

        return model

    def make_discriminator_model(self):
        # TODO: entender melhor!
        # model.add(layers.Embedding(input_length=32, input_dim=MIDI_NOTES_NUMBER + 1, output_dim=64))
        # model.add(layers.LSTM(128))
        # # Add a Dense layer with 10 units.
        # model.add(layers.Dense(10))

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[MIDI_NOTES_NUMBER, self.NUM_FUSES, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

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
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

        # Notice the use of `tf.function`
        # This annotation causes the function to be "compiled".

    @tf.function
    def train_step(self, images):
        # noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim], mean=MIDI_NOTES_NUMBER/2, stddev=10)
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator_model(noise, training=True)

            real_output = self.discriminator_model(images, training=True)
            fake_output = self.discriminator_model(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            # if self.should_verbose():
            #     print("\n\n")
            #     print(f'real_out_sum: {sum(abs(real_output))}')
            #     print(f'fake_ou_sum: {sum(abs(fake_output))}')
            #     print(f'real_out - fake_out: {sum(abs(real_output-fake_output))}')
            #     print(f'Gen Loss: {gen_loss}')
            #     print(f'Discr loss: {gen_loss}')

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator_model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator_model.trainable_variables))

        return None

    def train(self, epochs=50, verbose=False, should_generate_gif=False):
        # The dataset is a list of melodies encoded.
        try:
            dataset = self.training_data.get()
            dataset = np.repeat(dataset, 20, axis=0)  # todo: remover isso. paret provisória

            # add +1 do eliminate -1 notation (and intervals will be represented by 0 instead of -1)
            # dataset = dataset + 1

            # normalize dataset  to [-1, 1]
            # dataset = (dataset - MIDI_NOTES_NUMBER/2) / (MIDI_NOTES_NUMBER/2)

            # transform dataset into dim [total_size, numfingers, numfuses, 1]
            train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

            for epoch in range(epochs):
                start = time.time()
                for melody_batch in train_dataset:
                    self.train_step(melody_batch)  # [ BATCH_SIZE, NUM_FINGERS, NUM_FUSES, 1]

                # Save the model every 15 epochs
                if (epoch + 1) % 15 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

                if should_generate_gif:
                    self.generate_and_save_images(epoch)

                if verbose:
                    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

            # Generate after the final epoch
            if should_generate_gif:
                display.clear_output(wait=True)
                self.generate_and_save_images(epochs)
                self.generate_gif()
                self.erase_temp_images()
        except Exception as e:
            print(f"Failed training gan of type {self.part_type}: {e}")

    def generate_and_save_images(self, epoch, new_seed=False):
        predictions = self.generate_prediction(new_seed)
        # fig = plt.figure(figsize=(4, 4))
        #
        # for i in range(predictions.shape[0]):
        #     plt.imshow(predictions[i, :, :, 0], cmap='gray')
        num_bars = predictions.shape[0]
        concat_data = np.ndarray((MIDI_NOTES_NUMBER, 32 * num_bars))
        a = 0
        b = 32
        for i in range(num_bars):
            concat_data[:, a:b] = predictions[i, :, :, 0]
            a = b
            b = b + 32
        plt.imshow(concat_data, cmap='Blues')
        plt.axis([0, num_bars * 32, 0, 100])  # necessary for inverting y axis
        plt.ylabel("MIDI Notes")
        plt.xlabel("Fuses")
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

    def generate_melody_matrix(self, new_seed=False):
        predictions = self.generate_prediction(new_seed)
        np_config.enable_numpy_behavior()
        melody = tf.Variable(predictions)
        for i in range(melody.shape[0]):
            melody[i, :, :, 0].assign(round(melody[i, :, :, 0]))
        return melody.numpy()

    def generate_prediction(self, new_seed=False):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        if new_seed:
            new_seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
            predictions = self.generator_model(new_seed, training=False)
        else:
            predictions = self.generator_model(self.seed, training=False)
        return predictions

    def erase_temp_images(self):
        for f in glob.glob('.temp/image*.png'):
            os.remove(f)
