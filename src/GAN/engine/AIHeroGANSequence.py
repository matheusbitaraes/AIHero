import glob
import os
import time
import traceback

import matplotlib
from keras import layers

from src.GAN.engine.AIHeroGAN import AIHeroGAN
from src.GAN.exceptions.GANExceptions import GanTrainingException

matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from IPython import display

from src.utils.AIHeroHelper import HarmonicFunction
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER


# gerador:
# RNN + SOFTMAX + (LSTM) na update function g
# discriminador:
#     CNN mesmo

# treinamento:
# Policy Gradient
# https://github.com/Shaofanl/SeqGAN-Tensorflow
# https://www.tensorflow.org/text/tutorials/nmt_with_attention
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://github.com/tensorflow/nmt

class AIHeroGANSequence(AIHeroGAN):
    def __init__(self, config: dict, harmonic_function: HarmonicFunction = HarmonicFunction.TONIC):
        super().__init__(config, harmonic_function)
        self._gan_name = 'sequential_gan'

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)

        # overwrite a rguments
        self.checkpoint_dir = f'{config["checkpoint"]["checkpoint_folder"]}/{self._gan_name}/{self.harmonic_function.name}'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator_model,
                                              discriminator=self.discriminator_model)

    def get_name(self):
        return self._gan_name

    def make_generator_model(self):
        model = tf.keras.Sequential()

        model.add(layers.LSTM(DIM, input_shape=(), return_sequences=True))
        model.add(layers.Dropout(0.3))

        model.add(layers.LSTM(DIM2, input_shape=(), return_sequences=True))
        model.add(layers.Dense(DIM2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(n_vocab))

        model.add(layers.Activation('softmax'))



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

    def train(self, should_generate_gif: bool = False, prefix: str = "", num_epochs: int = None):
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        try:
            dataset = self.training_data.get_as_matrix()

            self.BUFFER_SIZE = min(self.BUFFER_SIZE, np.int(np.int(np.ceil(dataset.shape[0] * self.BUFFER_PERCENTAGE))))
            self.BATCH_SIZE = min(self.BATCH_SIZE, np.int(np.int(np.ceil(dataset.shape[0] * self.BATCH_PERCENTAGE))))

            train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

            current_time_min = 0
            total_epochs = num_epochs if num_epochs is not None else self.num_epochs
            epoch = 0
            while epoch < total_epochs:
                start = time.time()
                results = None
                for melody_batch in train_dataset:
                    results = self.train_step(melody_batch)  # [ BATCH_SIZE, NUM_FINGERS, TIME_DIVISION, 1]
                if self.should_verbose():
                    print(f'Loss G: {results["generator_loss"]},'
                          f' Loss D: {results["discriminator_loss"]}')

                # Save the model every 15 epochs
                if self._should_use_checkpoint and (epoch + 1) % 15 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)

                # measure GAN quality: DISABLE THIS FOR BETTER PERFORMANCE
                if self._live_quality_polling_enabled and (epoch + 1) % self._epochs_for_quality_measure == 0:
                    self._quality_measures.append(self.calculate_quality_measure())

                self._generator_losses.append(float(results["generator_loss"]))
                self._discriminator_losses.append(float(results["discriminator_loss"]))

                current_time_min = current_time_min + (time.time() - start) / 60

                if should_generate_gif:
                    self.generate_and_save_images(epoch, current_time_min)

                if self.should_verbose():
                    print(f'Time for epoch {epoch + 1} is {time.time() - start:.2f} sec')

                if self.stop_criterion_achieved():
                    print(f'Stop Criterion Achieved in epoch {epoch + 1}!!')
                    break

                epoch += 1

            # get last quality measure
            if self._live_quality_polling_enabled:
                self._quality_measures.append(self.calculate_quality_measure())

            # Generate after the final epoch
            if should_generate_gif:
                display.clear_output(wait=True)
                self.generate_and_save_images(epoch, current_time_min)
                self.generate_gif(filename_prefix=prefix)

                # erase temporary images
                for f in glob.glob('.temp/*.png'):
                    os.remove(f)

        except Exception as e:
            print(f"Failed training gan of type {self.harmonic_function.name}: {e}")
            print(traceback.format_exc())
            raise GanTrainingException()

    def stop_criterion_achieved(self):
        last_scores = self._quality_measures[-5:]
        if len(last_scores) < 5:
            return False

        for score in last_scores:
            if score > self._quality_target_value:
                return False
        return True
