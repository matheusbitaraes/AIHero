import time

from src.AISynth import AISynth
from src.resources import *
from mingus.containers.note import Note
from mingus.containers import Bar, NoteContainer
import mingus.extra.lilypond as LilyPond

bpm = 80
central_note = 60
scale = 'minor_blues_scale'
pulses_on_compass = 4
licks = known_licks[scale]
synth = AISynth(r'../fluidsynth/sf2/FluidR3_GM.sf2')
initial_time = 500  # delay antes de começar (ms)


def updateMusicSheet(notes):
    fuse = 60 / (bpm * 8)  # duration of 'fuse' note in seconds
    music_bar = Bar()
    for note in notes:
        music_bar.place_notes(note, 8)
    lili_bar = LilyPond.from_Bar(music_bar)
    LilyPond.to_png(lili_bar, 'melody_sheet')


for lick in licks:
    total_duration = 60 / bpm * pulses_on_compass  # time of a beat(ms) * number of beats
    fuse = 60 / (bpm * 8)  # duration of 'fuse' note in seconds

    # transforma em notas
    notes = []
    for j in range(0, len(lick)):
        # p = p + 1
        if lick[j] != None and lick[j] > -1:
            note = int(central_note + lick[j])
            notes.append(Note().from_int(note))
        else:
            notes.append(Note().empty())

    print('executing phrase {}'.format(lick))
    initial_time = synth.schedule_melody(
        initial_time, fuse, '1', central_note, notes)

    updateMusicSheet(notes)

    time.sleep(total_duration)
time.sleep(16 * fuse)


