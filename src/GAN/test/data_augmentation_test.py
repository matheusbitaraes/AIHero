# script for testing the effect of using data augmentation strategies for enhancing test
import time
from glob import glob
from json import load as jload

import numpy as np

from src.GAN.data.GANTrainingData import GANTrainingData
from src.GAN.engine.AIHeroGAN import AIHeroGAN
from src.GAN.engine.augmentation.AugmentationStrategies import TimeChangeStrategy, OctaveChangeStrategy, \
    NoteJoinStrategy
from src.data.AIHeroData import AIHeroData

NUM_SAMPLES = 50
NUM_EPOCHS = 1
TRAIN_TEST_RATIO = 0.7
AUGMENTATION_SIZE = 1  # how many times each strategy should augment

with open('test_config.json') as config_file:
    config = jload(config_file)

avg_grade_a = np.zeros(NUM_SAMPLES)
avg_grade_b = np.zeros(NUM_SAMPLES)
avg_grade_c = np.zeros(NUM_SAMPLES)

avg_time_a = np.zeros(NUM_SAMPLES)
avg_time_b = np.zeros(NUM_SAMPLES)
avg_time_c = np.zeros(NUM_SAMPLES)
data = AIHeroData()

strategies = [OctaveChangeStrategy(), TimeChangeStrategy(), NoteJoinStrategy()]
for i in range(0, NUM_SAMPLES):
    data.load_from_midi_files(glob(f"{config['train_data_folder']}/part*"))
    train_data, test_data = data.split_into_train_test(TRAIN_TEST_RATIO)

    # Create augmented data
    train_augmented_data = AIHeroData()
    train_augmented_data.set_mingus_compositions(train_data.get_mingus_compositions(), chord_list=train_data.chord_list)
    train_augmented_data.augment(strategies, AUGMENTATION_SIZE)

    # Create replica data
    d = train_augmented_data.get_spr_as_matrix()
    augmented_data_size = d.shape[0]
    train_replicated_data = AIHeroData()
    train_replicated_data.set_mingus_compositions(train_data.get_mingus_compositions(),
                                                  chord_list=train_data.chord_list)
    train_replicated_data.replicate(final_size=augmented_data_size)

    # generate and train gans
    print("Trainnig gan with test data...")
    gan_a = AIHeroGAN(config)
    gan_a.training_data = GANTrainingData(config, data=train_data)
    t_start = time.time()
    gan_a.train(num_seeds=1, epochs=NUM_EPOCHS)
    avg_time_a[i] = time.time() - t_start

    print("Trainnig gan with augmented data...")
    gan_b = AIHeroGAN(config)
    gan_b.training_data = GANTrainingData(config, data=train_augmented_data)
    t_start = time.time()
    gan_b.train(num_seeds=1, epochs=NUM_EPOCHS)
    avg_time_b[i] = time.time() - t_start

    print("Trainnig gan with replicated data...")
    gan_c = AIHeroGAN(config)
    gan_c.training_data = GANTrainingData(config, data=train_replicated_data)
    t_start = time.time()
    gan_c.train(num_seeds=1, epochs=NUM_EPOCHS)
    avg_time_c[i] = time.time() - t_start

    test_data_matrix = test_data.get_spr_as_matrix()
    size = test_data_matrix.shape[0]
    discriminator_response_a = gan_a.discriminator_model(test_data_matrix, training=False)
    discriminator_response_b = gan_b.discriminator_model(test_data_matrix, training=False)
    discriminator_response_c = gan_c.discriminator_model(test_data_matrix, training=False)
    avg_grade_a[i] = np.mean(discriminator_response_a)
    avg_grade_b[i] = np.mean(discriminator_response_b)
    avg_grade_c[i] = np.mean(discriminator_response_c)
    print(
        f"\nThe average discriminator response for iteration {i + 1} is \n test_data: {avg_grade_a[i]} augmented_data:{avg_grade_b[i]}, replicated_data: {avg_grade_c[i]} \n")

print(f"****** RESULTS ******* :\n"
      f"Original Dataset: {np.mean(avg_grade_a)} +- {np.std(avg_grade_a)}\n"
      f"Augmented Dataset: {np.mean(avg_grade_b)} +- {np.std(avg_grade_b)}\n"
      f"Replicated Dataset: {np.mean(avg_grade_c)} +- {np.std(avg_grade_c)}")

a_acc = avg_grade_a.tolist()
a_acc.insert(0, "original_acc")

b_acc = avg_grade_b.tolist()
b_acc.insert(0, "augmented_acc")

c_acc = avg_grade_c.tolist()
c_acc.insert(0, "replicated_acc")

a_time = avg_time_a.tolist()
a_time.insert(0, "original_time")

b_time = avg_time_b.tolist()
b_time.insert(0, "augmented_time")

c_time = avg_time_c.tolist()
c_time.insert(0, "replicated_time")

np.savetxt('scores.csv', [p for p in zip(a_acc, a_time, b_acc, b_time, c_acc, c_time)], delimiter=';', fmt='%s')
