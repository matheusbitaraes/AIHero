import os
import traceback
import warnings
from glob import glob

import numpy as np
import pretty_midi as pyd
from mingus.midi import midi_file_in

from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroHelper import HarmonicFunction

midi_root = "/home/matheus/Documentos/AIHero/src/GAN/data/train/"
ckpt_save_dir = "src/GAN/data/train/manual"

for function in HarmonicFunction:
    print(f"Getting data for melodic part: {function.name}")
    data = AIHeroData()
    path = glob(f"{midi_root}/part_{function.name}*")
    data.load_from_midi_files(path)
    data.save_data(ckpt_save_dir, prefix=function.name)
print("finished!")
