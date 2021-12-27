from glob import glob
from json import load as jload

from mingus.containers import Bar, Composition, Track, Note

from src.GAN.engine.AIHeroGAN import AIHeroGAN
from src.data.AIHeroData import AIHeroData
from src.utils.AIHeroEnums import MelodicPart

with open('config.json') as config_file:
    config = jload(config_file)

part = MelodicPart.Y
gan = AIHeroGAN(config, part=part)


def get_midi_file_from_part(config, part):
    file_directory = config["training"]["train_data_folder"]
    return glob(f"{file_directory}/part_{part.name}_*")

# c = Composition()
# t = Track()
# b1 = Bar()
# b1.place_notes(Note("C"), 64)
# b1.place_notes(Note("D"), 64)
# b1.place_notes(Note("E"), 64)
# b2 = Bar()
# b2.place_notes(Note("C"), 64)
# b2.place_notes(Note("D"), 64)
# b2.place_notes(Note("E"), 64)
# t.add_bar(b1)
# t.add_bar(b2)
# c.add_track(t)
# cs = []
# cs.append([c, 120])
# d = AIHeroData()
# d.set_mingus_compositions(cs)
# d.export_as_midi(file_name="dasdasda")
midi_file = get_midi_file_from_part(config, part)
spr = gan.training_data._ai_hero_data.get_spr()
data = AIHeroData()
data.chord_list = gan.training_data._ai_hero_data.chord_list
list = []
# list.append(spr)
data.set_spr(spr)
# data.append_base_track(midi_file[0])
data.export_as_midi(file_name="testeasaa")
