import fluidsynth
import numpy as np
import time
from mingus.midi import fluidsynth
from threading import Thread

from src.resources import *


class AISynth:
    def __init__(self, path=r'../fluidsynth/sf2/FluidR3_GM.sf2'):
        fluidsynth.init(path)
        # self.fs.start(driver='coreaudio')  # use DirectSound driver
        # sfid = self.fs.sfload(path)  # replace path as needed
        # self.fs.program_select(0, sfid, 0, 2)

        fluidsynth.init(path)
        # self.fs_metronome.start(driver='coreaudio')  # use DirectSound driver
        # sfid_metronome = self.fs_metronome.sfload(path)  # replace path as needed
        # self.fs_metronome.program_select(0, sfid_metronome, 0, 115) #115 é o metronomo e 116 é bumbo

        # self.fs_chord = fluidsynth.Synth()
        # self.fs_chord.start(driver='coreaudio')  # use DirectSound driver
        # sfid_chord = self.fs_chord.sfload(path)  # replace path as needed
        # self.fs_chord.program_select(0, sfid_chord, 0, 1)

        # self.seq = fluidsynth.Sequencer()
        # self.synthID = self.seq.register_fluidsynth(self.fs)
        # self.synthIDMetronome = self.seq.register_fluidsynth(self.fs_metronome)
        # self.synthIDChord = self.seq.register_fluidsynth(self.fs_chord)


    def play_composition(self, composition):
        # fluidsynth.play_Composition(composition)

        # _thread.start_new_thread(fluidsynth.play_Track, (composition.tracks[1],)) # não esta sincronizado
        # _thread.start_new_thread(fluidsynth.play_Track, (composition.tracks[2],)) # não esta sincronizado
        trds = []  # threads
        for track in composition.tracks:
            # _thread.start_new_thread(fluidsynth.play_Track, (track,)) # não esta sincronizado
            trds.append(Thread(target=fluidsynth.play_Track, args=(track,)))

        for tr in trds:  # play all tracks
            tr.start()

        for tr in trds:  # wait all tracks
            tr.join()



        #     for bar in track.bars:
        #         fluidsynth.play_Bar(bar, 1, 150)
        return None
    # def schedule_metronome(self, initial_time, duration):
    #     for t in duration*np.arange(4)/4:
    #         self.seq.note_on(time=round(initial_time+1000*t), channel=0, key=48, velocity=60, dest=self.synthIDMetronome)

    # def schedule_chord(self, central_note, chord, initial_time):
    #     for chord_note in chords[chord]:
    #         self.seq.note_on(time=initial_time, channel=0, key=central_note + chord_note,
    #                          velocity=70, dest=self.synthIDChord)

    # def play_composition(self, composition):
    #     for track in composition.tracks:
    #         for bar in track.bars:
    #             self.play_bar(bar, initial_delay=0)
    #     return None
    #
    # def play_bar(self, bar, initial_delay):
    #
    #     for note_container in bar:
    #         initial_time = int(note_container[0] * 1000)
    #         interval = int(note_container[1])  # interval of the note. Ex: if 32 the duration is 1/32 of tempo
    #         print(initial_time)
    #         print(interval)
    #         print("---")
    #         if note_container[2]:
    #             note = note_container[2].notes[0]
    #             self.seq.note_on(time=initial_time, channel=0, key=int(note), velocity=80, dest=self.synthID, absolute=False)
    #             # self.seq.note_off(time=interval, channel=0, key=int(note), dest=self.synthID)
    #         # current_time += int(1000 * fuse * round(32/interval))
    #     return None

    # def schedule_melody(self, initial_time, fuse, chord, central_note, melody, metronome=True):
    #     current_time = initial_time
    #     duration = 32 * fuse
    #
    #     if metronome:
    #         self.schedule_metronome(initial_time, duration)
    #     if chord:
    #         self.schedule_chord(central_note, chord, initial_time)
    #     for bar in melody.bars:
    #         for note_container in bar:
    #             interval = note_container[1]  # interval of the note. Ex: if 32 the duration is 1/32 of tempo
    #             if note_container[2]:
    #                 note = note_container[2].notes[0]
    #                 self.seq.note_on(time=current_time, channel=0, key=int(note), velocity=note.velocity,
    #                                  dest=self.synthID)
    #             current_time += int(1000 * fuse * round(32/interval))
    #
    #     return initial_time + round(1000 * duration)
    def play_compositions(self, compositions):
        for composition in compositions:
            self.play_composition(composition[0])
        pass
