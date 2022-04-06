from threading import Thread

from mingus.midi import fluidsynth

AUDIO_DRIVER = 'pulseaudio'


class AISynth:
    def __init__(self, config):
        path = config['fluidsynth_path']
        fluidsynth.init(r'../fluidsynth/FluidR3_GM.sf2', AUDIO_DRIVER)

    # def play_composition(self, mingus_composition):
    #
    #
    #     # # _thread.start_new_thread(fluidsynth.play_Track, (composition.tracks[1],)) # não esta sincronizado
    #     # # _thread.start_new_thread(fluidsynth.play_Track, (composition.tracks[2],)) # não esta sincronizado
    #     # trds = []  # threads
    #     # for track in composition.tracks:
    #     #     # _thread.start_new_thread(fluidsynth.play_Track, (track,)) # não esta sincronizado
    #     #     trds.append(Thread(target=fluidsynth.play_Track, args=(track,)))
    #     #
    #     # for tr in trds:  # play all tracks
    #     #     tr.start()
    #     #
    #     # for tr in trds:  # wait all tracks
    #     #     tr.join()
    #     # return None

    def play_compositions(self, compositions):
        for composition in compositions:
            fluidsynth.play_Composition(composition[0])
        pass
