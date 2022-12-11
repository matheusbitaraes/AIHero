from json import load as jload

import numpy as np
from keras import layers
from tensorflow import keras

from src.GEN.data.GANTrainingData import GANTrainingData
from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroGlobals import SCALED_NOTES_NUMBER, SCALED_NOTES_CLASSES
from src.utils.AIHeroHelper import HarmonicFunction

with open('src/config.json') as config_file:
    config = jload(config_file)

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=0.002)

# load the dataset
training_data = GANTrainingData(config, harmonic_function=HarmonicFunction(1))
inputs, targets = training_data.get_as_LSTM_training_sequences()  # num_samples, midi_notes, time_step, ?

# print(f'shape: {data.shape}')

model = keras.models.Sequential()

# seq_length, input_size. If shape is (None, SCALED_NOTES_NUMBER), we can use many time divisions
model.add(keras.Input(shape=(None, SCALED_NOTES_CLASSES)))

# N, 128 (if return_sequences=true, will be N, length, 128. good for stacking multiple rnns)
model.add(layers.LSTM(300))
model.add(layers.Dropout(0.3))  # ver se precisa
model.add(layers.Dense(SCALED_NOTES_CLASSES, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)

model.summary()

model.fit(inputs, targets, epochs=50, batch_size=100)  # inputs: sequencia, targets: sequencia + 1

# create melody
total_melody_size = 1000
seed = [48, 48, 48, 48, 48, 48, 48, 48]
steps = total_melody_size - len(seed)
melody = seed


def _sample_with_temperature(probabilities, temperature=1):
    predictions = np.log(probabilities) / temperature
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

    choices = range(len(probabilities))
    index = np.random.choice(choices, p=probabilities)

    return index

for _ in range(steps):
    # seed = seed[-max_sequence_length:]
    onehot_seed = keras.utils.to_categorical(seed, num_classes=SCALED_NOTES_CLASSES)
    onehot_seed = onehot_seed[np.newaxis, ...]

    # prediction
    probabilities = model.predict(onehot_seed)[0]

    output_int = _sample_with_temperature(probabilities, 1)
    seed.append(output_int)

print(seed)

new_melody = np.ones([1, SCALED_NOTES_NUMBER, total_melody_size, 1]) * -1
for i in range(len(seed)-1):
    if seed[i] != 48:
        new_melody[0, seed[i], i, 0] = 1

ai_hero_data = AIHeroData()
ai_hero_data.add_data(new_melody, chord_array=np.zeros([1, total_melody_size]))

ai_hero_data.export_as_midi('tetest')
