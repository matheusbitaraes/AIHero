import random
import traceback
from queue import Queue

import numpy as np

from src.EVO.service.EVOService import EVOService
from src.GAN.data.GANTrainingData import GANTrainingData
from src.GAN.service.GANService import GANService
from src.data.AIHeroData import AIHeroData
from src.service.AIHeroService import AIHeroService
from src.utils.AIHeroEnums import MelodicPart


class QueueService:
    def __init__(self):
        self.queue = Queue()

    def add_to_queue(self, melody_specs_list):
        self.queue.  append(melody_specs_list)

    def get_next_element(self, melody_specs_list):
        return self.queue.append(melody_specs_list)
