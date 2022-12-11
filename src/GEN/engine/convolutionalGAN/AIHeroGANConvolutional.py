import glob
import os
import time
import traceback

import matplotlib

from src.GEN.engine.AIHeroGEN import AIHeroGEN
from src.GEN.engine.convolutionalGAN.Discriminator import Discriminator
from src.GEN.engine.convolutionalGAN.Generator import Generator
from src.GEN.exceptions.GENExceptions import GENTrainingException

matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from IPython import display

from src.utils.AIHeroHelper import HarmonicFunction


class AIHeroGANConvolutional(AIHeroGEN):
    def __init__(self, config: dict, harmonic_function: HarmonicFunction = HarmonicFunction.TONIC):
        super().__init__(config, harmonic_function)
        self._gan_name = 'convolutional_gan'
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

        # overwrite arguments
        self.checkpoint_dir = f'{config["checkpoint"]["checkpoint_folder"]}/{self._gan_name}/{self.harmonic_function.name}'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator.optimizer,
                                              discriminator_optimizer=self.discriminator.optimizer,
                                              generator=self.generator.model,
                                              discriminator=self.discriminator.model)

        if self._should_use_checkpoint:
            self.load_from_checkpoint()

    def get_name(self):
        return self._gan_name

    def load_from_checkpoint(self):
        if self.should_verbose():
            print(f"loading checkpoint for gan of part {self.harmonic_function.name}...")
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

    def generate_prediction(self, new_seed: bool = False, size: int = 1):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        if new_seed:
            seed = tf.random.normal([size, self.noise_dim])
            predictions = self.generator.model(seed, training=False)
        else:
            predictions = self.generator.model(self.seed, training=False)
        return predictions

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator.model(noise, training=True)

            real_output = self.discriminator.model(images, training=True)
            fake_output = self.discriminator.model(generated_images, training=True)
            gen_loss = self.generator.loss(fake_output)
            disc_loss = self.discriminator.loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)

        self.generator.optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.model.trainable_variables))
        self.discriminator.optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.model.trainable_variables))

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

                # measure GEN quality: DISABLE THIS FOR BETTER PERFORMANCE
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
            raise GENTrainingException()

    def stop_criterion_achieved(self):
        last_scores = self._quality_measures[-5:]
        if len(last_scores) < 5:
            return False

        for score in last_scores:
            if score > self._quality_target_value:
                return False
        return True
