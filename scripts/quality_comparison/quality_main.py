from glob import glob

import tensorflow as tf

from scripts.quality_comparison.HelperMethods import HelperMethods

CLEAN_OLD_MELODIES = True
GENERATE_FRESH_DATA = True
GENERATE_FRESH_QUALITY_MEASUREMENTS = True

WORK_DIR = 'scripts/quality_comparison'
GAN_DIRECTORY = f'{WORK_DIR}/gan_generated_midi'
EVO_DIRECTORY = f'{WORK_DIR}/ganevo_generated_midi'
EVO_ALT_DIRECTORY = f'{WORK_DIR}/ganevoalt_generated_midi'
LSTM_DIRECTORY = f'{WORK_DIR}/lstm_generated_midi'
TESTDATA_DIRECTORY = f'{WORK_DIR}/testdata_generated_midi'

# Enable memory growth for the first GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

config = {
    "WORK_DIR": WORK_DIR,
    "QUALITY_DATA_NAME": "quality_metrics_checkpoint",
    "MODELS": [
        {
            "NAME": "EVO_ALT",
            "TYPE": "GENERATED",
            "DIRECTORY": EVO_ALT_DIRECTORY,
            "DIRECTORY_PATH": glob(f"{EVO_ALT_DIRECTORY}/*"),
            "CONFIG_FILE_NAME": "config_evo_alt.json",
        },
        {
            "NAME": "GAN",
            "TYPE": "GENERATED",
            "DIRECTORY": GAN_DIRECTORY,
            "DIRECTORY_PATH": glob(f"{GAN_DIRECTORY}/*"),
            "CONFIG_FILE_NAME": "config.json",
        },
        {
            "NAME": "EVO",
            "TYPE": "GENERATED",
            "DIRECTORY": EVO_DIRECTORY,
            "DIRECTORY_PATH": glob(f"{EVO_DIRECTORY}/*"),
            "CONFIG_FILE_NAME": "config.json",
        },
        {
            "NAME": "LSTM",
            "TYPE": "GENERATED",
            "DIRECTORY": LSTM_DIRECTORY,
            "DIRECTORY_PATH": glob(f"{LSTM_DIRECTORY}/*"),
            "CONFIG_FILE_NAME": "config_lstm.json",

        },
        {
            "NAME": "TESTDATA",
            "TYPE": "TEST",
            "DIRECTORY": TESTDATA_DIRECTORY,
            "DIRECTORY_PATH": glob(f"{TESTDATA_DIRECTORY}/*"),
            "CONFIG_FILE_NAME": "config.json",
        }
    ]
}

helper = HelperMethods(config)

if CLEAN_OLD_MELODIES:
    print('Cleaning old data...')

    # iterate over all config models and delete all files
    for model in config['MODELS']:
        helper.delete_files_with_pattern(model['DIRECTORY'], '*.mid')

if GENERATE_FRESH_DATA:
    print('Generating data...')
    helper.generate_and_save_data(12 * 40, random_evo_weights=False)

if GENERATE_FRESH_QUALITY_MEASUREMENTS:
    print('Getting quality measurements...')
    helper.get_quality_measures_and_save()

# Load the dictionary from the file
print('Setting up plots and other data...')
helper.generate_quality_plots()
