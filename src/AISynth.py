import time

import fluidsynth
import numpy as np

from src.resources import *


class AISynth:
    def __init__(self, path=r'./fluidsynth/sf2/FluidR3_GM.sf2'):
        self.fs = fluidsynth.Synth()
        self.fs.start(driver='coreaudio')  # use DirectSound driver
        sfid = self.fs.sfload(path)  # replace path as needed
        self.fs.program_select(0, sfid, 0, 0)
        self.seq = fluidsynth.Sequencer()
        self.synthID = self.seq.register_fluidsynth(self.fs)

    def execute_melody_and_harmony(self, total_duration, fuse, chord, central_note,
                                   melody, metronome=True):
        notes_window = np.arange(0, total_duration, fuse).tolist()
        time.sleep(8 * fuse)
        for i in range(0, len(notes_window)):
            if melody[i] and melody[i].note and melody[i].note != -1:
                self.fs.noteon(0, melody[i].note, melody[i].velocity)

            # keep track of the number of tempos
            tempo = i / 8

            # check for chord play
            if tempo == 0:
                # chord_name = chord_sequence[chord_transition_time.index(tempo)]
                for chord_note in chords[chord]:
                    self.fs.noteon(0, central_note + chord_note, 80)

            # checa linha de baixo
            # if bass_line[ib][1] == i:
            #     ib = ib + 1
            #     self.fs.noteon(0, self.central_note + chords[chord_name][0] + bass_line[ib][0] - 24, 70)

            if i % 8 == 0 and metronome is True:
                self.fs.noteon(0, central_note - 24 + chords[chord][0], 60)
            time.sleep(fuse)

    def schedule_melody(self, initial_time, fuse, chord, central_note,
                        melody, metronome=True):
        current_time = initial_time
        i = 0
        for note in melody:
            if note and int(note) and int(note) != -1:
                self.seq.note_on(time=current_time, channel=0, key=int(note), velocity=note.velocity, dest=self.synthID)

            # keep track of the number of tempos
            tempo = i / 8

            # check for chord play
            if tempo == 0:
                # chord_name = chord_sequence[chord_transition_time.index(tempo)]
                for chord_note in chords[chord]:
                    self.seq.note_on(time=current_time, channel=0, key=central_note + chord_note,
                                     velocity=70, dest=self.synthID)

            if i % 8 == 0 and metronome is True:
                self.seq.note_on(time=current_time, channel=0, key=central_note - 24 + chords[chord][0],
                                 velocity=60, dest=self.synthID)

            current_time = current_time + int(1000 * fuse)
            i += 1

        return current_time
