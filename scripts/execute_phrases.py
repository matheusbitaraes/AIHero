import time

from src.AISynthDEPRECATED import AISynth
from src.resources import *
from mingus.containers.note import Note
from mingus.containers import Bar, NoteContainer, Track
import mingus.extra.lilypond as LilyPond

bpm = 100
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

    # transform in notes
    note_array = lick
    rest_count = 0
    note_bar = Bar()
    note_track = Track()
    for j in range(0, len(note_array)):
        if note_array[j] > -1:
            if rest_count > 0:  # add rest before note
                note_bar.place_rest(32/rest_count)
            rest_count = 0
            note = int(central_note + note_array[j])
            note_bar.place_notes(Note().from_int(note), 32)  # todo: future improvements is to set notes other than fuse
        else:
            rest_count += 1
        if note_bar.is_full():
            note_track + note_bar
            note_bar = Bar()  # create a new bar
    note_track + note_bar

    initial_time = synth.schedule_melody(
        initial_time, fuse, None, central_note, note_track)

time.sleep(initial_time/1000)


