from src.AIGAN import AIGAN
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

from src.AIHeroData import AIHeroData
from src.AISynth import AISynth

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_files = glob("resources/train/Blues_Licks_-_BB_King_Scale_And_Examples.mid")

# regras iniciais para train sets (limitações do modelo)
# 1) apenas uma nota por vez
# 2) Apenas um track por vez (será considerado apenas o track da melodia - improvisacao)
# 3) as velocities (intensidades sonoras) serão iguais inicialmente
# 4) a duração das notas serão iguais (talvez criar uma função de definição aleatória de duraçao baseada em uma distribuição normal?    )
# 5) inicialmente só 4/4

train_set = AIHeroData()
train_set.load_from_midi_files(train_files)

# todo AGORA: finalizar codificação e decodificação das melodias!!


# tocar as melodias decodificando do dataset, para testar se está decodificando corretamente.
synth = AISynth()
# synth.play_compositions(train_set.get_composition())

# colocar codigo abaixo como um teste automatizado
train_set.set_data(train_set.get_data())
synth.play_compositions(train_set.get_composition())  # todo implement

# encode dataset
gan = AIGAN()
gan.train(train_set.get_data(), 1, True)  # todo implement correctly

# train dataset

# Generate melodies
