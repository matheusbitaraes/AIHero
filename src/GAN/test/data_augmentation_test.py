# script for testing the effect of using data augmentation strategies for enhancing test
import json
import time
from datetime import date
from glob import glob
from json import load as jload

import numpy as np

from src.GAN.data.GANTrainingData import GANTrainingData
from src.GAN.engine.AIHeroGAN import AIHeroGAN
from data.AIHeroData import AIHeroData
from utils.AIHeroEnums import MelodicPart

NUM_SAMPLES = 20
TRAIN_TEST_RATIO = 1  # leave-one-out
EPOCH_LOSSES_RANGE = 5
part = MelodicPart.X
# INITIAL_REPLICATION = 1

should_generate_gif = False

with open('test_config.json') as config_file:
    config = jload(config_file)

average_gen_loss_a = []
average_gen_loss_b = []
average_gen_loss_c = []

average_disc_loss_a = []
average_disc_loss_b = []
average_disc_loss_c = []

diff_a = []
diff_b = []
diff_c = []

sum_a = []
sum_b = []
sum_c = []

avg_time_a = []
avg_time_b = []
avg_time_c = []

fid_a = []
fid_b = []
fid_c = []

data = AIHeroData()


def transform_specs_into_name(config, TRAIN_TEST_RATIO):
    da_pipeline = ""
    for d in config['data_augmentation']['data_augmentation_strategy_pipeline']: da_pipeline += json.dumps(d)
    da_pipeline = da_pipeline \
        .replace("{", "_") \
        .replace("}", "") \
        .replace("\"", "") \
        .replace(",", "") \
        .replace(" ", "_") \
        .replace(":", "_")
    return f"num_epochs_{config['training']['num_epochs']}_" \
           f"da_pipeline_{da_pipeline}"


specs = transform_specs_into_name(config, TRAIN_TEST_RATIO)
for i in range(0, NUM_SAMPLES):
    data.load_from_midi_files(glob(f"{config['training']['train_data_folder']}/part_{part.name}*"))
    train_data, test_data = data.split_into_train_test(TRAIN_TEST_RATIO)

    # train_data.replicate(INITIAL_REPLICATION)

    train_data_a = AIHeroData()
    train_data_b = AIHeroData()
    train_data_c = AIHeroData()
    train_data_a.set_mingus_compositions(train_data.get_mingus_compositions(), train_data.chord_list)
    train_data_b.set_mingus_compositions(train_data.get_mingus_compositions(), train_data.chord_list)
    train_data_c.set_mingus_compositions(train_data.get_mingus_compositions(), train_data.chord_list)

    # generate and train gans
    print(f"Train data size: {train_data.get_spr_as_matrix().shape[0]}")
    print("Training gan with train data...")
    config["data_augmentation"]["enabled"] = False  # disable for this test
    gan_a = AIHeroGAN(config)
    gan_a.training_data = GANTrainingData(config, data=train_data_a)
    print(f"Training data shape: {gan_a.training_data.get_as_matrix().shape}")

    # get a non trained random noise
    # test_data_matrix = test_data.get_spr_as_matrix()
    # size = test_data_matrix.shape[0]
    # random_noise = gan_a.generate_prediction(True, size)

    t_start = time.time()
    gan_a.train(should_generate_gif=should_generate_gif, prefix="original_")
    avg_time_a.append(time.time() - t_start)
    average_gen_loss_a.append(np.mean(gan_a.generator_losses[-EPOCH_LOSSES_RANGE:]))
    average_disc_loss_a.append(np.mean(gan_a.discriminator_losses[-EPOCH_LOSSES_RANGE:]))
    fid_a.append(gan_a.quality_measures[-1])

    print("Training gan with augmented data...")
    gan_b = AIHeroGAN(config)
    config["data_augmentation"]["enabled"] = True
    gan_b.training_data = GANTrainingData(config, data=train_data_b)
    print(f"Augmented data shape: {gan_b.training_data.get_as_matrix().shape}")
    augmented_data_size = gan_b.training_data.get_as_matrix().shape[0]
    t_start = time.time()
    gan_b.train(should_generate_gif=should_generate_gif, prefix="augmented_")
    avg_time_b.append(time.time() - t_start)
    average_gen_loss_b.append(np.mean(gan_b.generator_losses[-EPOCH_LOSSES_RANGE:]))
    average_disc_loss_b.append(np.mean(gan_b.discriminator_losses[-EPOCH_LOSSES_RANGE:]))
    fid_b.append(gan_b.quality_measures[-1])

    print("Training gan with replicated data...")
    config["data_augmentation"]["enabled"] = False  # disable for this test
    gan_c = AIHeroGAN(config)
    gan_c.training_data = GANTrainingData(config, data=train_data_c)
    gan_c.training_data.replicate(final_size=augmented_data_size)
    print(f"Replicated data shape: {gan_c.training_data.get_as_matrix().shape}")
    t_start = time.time()
    gan_c.train(should_generate_gif=should_generate_gif, prefix="replicated_")
    avg_time_c.append(time.time() - t_start)
    average_gen_loss_c.append(np.mean(gan_c.generator_losses[-EPOCH_LOSSES_RANGE:]))
    average_disc_loss_c.append(np.mean(gan_c.discriminator_losses[-EPOCH_LOSSES_RANGE:]))
    fid_c.append(gan_c.quality_measures[-1])

    diff_a.append(abs(average_gen_loss_a[i] - average_disc_loss_a[i]))
    diff_b.append(abs(average_gen_loss_b[i] - average_disc_loss_b[i]))
    diff_c.append(abs(average_gen_loss_c[i] - average_disc_loss_c[i]))

    sum_a.append(average_gen_loss_a[i] + average_disc_loss_a[i])
    sum_b.append(average_gen_loss_b[i] + average_disc_loss_b[i])
    sum_c.append(average_gen_loss_c[i] + average_disc_loss_c[i])

    print(f"****** SAMPLE {i + 1} ******* :\n"
          f"Original Gen loss: {np.mean(average_gen_loss_a)} +- {np.std(average_gen_loss_a)}\n"
          f"Augmented Gen loss: {np.mean(average_gen_loss_b)} +- {np.std(average_gen_loss_b)}\n"
          f"Replicated Gen loss: {np.mean(average_gen_loss_c)} +- {np.std(average_gen_loss_c)}\n\n"
          f"Original Discr loss: {np.mean(average_disc_loss_a)} +- {np.std(average_disc_loss_a)}\n"
          f"Augmented Discr loss: {np.mean(average_disc_loss_b)} +- {np.std(average_disc_loss_b)}\n"
          f"Replicated Discr loss: {np.mean(average_disc_loss_c)} +- {np.std(average_disc_loss_c)}\n\n"
          f"Original Dataset Diff: {np.mean(diff_a)} +- {np.std(diff_a)}\n"
          f"Augmented Dataset Diff: {np.mean(diff_b)} +- {np.std(diff_b)}\n"
          f"Replicated Dataset Diff: {np.mean(diff_c)} +- {np.std(diff_c)}\n\n"
          f"Original Dataset sum: {np.mean(sum_a)} +- {np.std(sum_a)}\n"
          f"Augmented Dataset sum: {np.mean(sum_b)} +- {np.std(sum_b)}\n"
          f"Replicated Dataset sum: {np.mean(sum_c)} +- {np.std(sum_c)}\n\n"
          f"Original Dataset FID: {np.mean(fid_a)} +- {np.std(fid_a)}\n"
          f"Augmented Dataset FID: {np.mean(fid_b)} +- {np.std(fid_b)}\n"
          f"Replicated Dataset FID: {np.mean(fid_c)} +- {np.std(fid_c)}")

    today = date.today()
    filename = f'result_{specs}_{today.strftime("%Y%m%d")}'
    np.savetxt(f'{filename}.csv',
               [p for p in zip(average_gen_loss_a,
                               average_disc_loss_a,
                               sum_a,
                               diff_a,
                               avg_time_a,
                               average_gen_loss_b,
                               average_disc_loss_b,
                               sum_b,
                               diff_b,
                               avg_time_b,
                               average_gen_loss_c,
                               average_disc_loss_c,
                               sum_c,
                               diff_c,
                               avg_time_c,
                               fid_a,
                               fid_b,
                               fid_c
                               )],

               delimiter=';', fmt='%s')
