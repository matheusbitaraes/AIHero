import os
import traceback
import warnings
from glob import glob

import numpy as np
import pretty_midi as pyd
from mingus.midi import midi_file_in

from src.data.AIHeroData import AIHeroData

midi_root = "/home/matheus/Documentos/POP909-Dataset/POP909/"
ckpt_save_dir = "src/GEN/data/train/"

data = AIHeroData()
data.load_from_pop909_dataset(midi_root)
data.save_data(ckpt_save_dir)
print("finished!")
