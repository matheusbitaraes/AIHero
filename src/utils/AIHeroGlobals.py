MIDI_NOTES_NUMBER = 128
SCALED_NOTES_RANGE = [0, 48]  # MIN, MAX
SCALED_NOTES_NUMBER = SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]
SCALED_NOTES_CLASSES = SCALED_NOTES_NUMBER + 1 # +1 because of the "no note" class
CENTRAL_NOTE_NUMBER = 60
TIME_DIVISION = 64
# TRAIN_DATA_REPLICAS = 40
NOTES_IN_OCTAVE = 12
AVAILABLE_SOURCES = {"evo", "train", "gan"}
DEFAULT_MELODY_REQUEST = {
    "harmony_specs": [
        {
            "melodic_part": "RELAXATION",
            "chord": "C:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "TENSION",
            "chord": "F:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C:7maj" ,
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C:7maj" ,
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "F:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "TENSION",
            "chord": "F:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "TENSION",
            "chord": "G:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "TENSION",
            "chord": "F:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C:7maj",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RETAKE",
            "chord": "G:7maj",
            "key": "C",
            "tempo": 120,
        }
    ],
    "evolutionary_specs": [
        {
            "key": "notes_on_same_chord_key",
            "name": "Notes on Same Chord",
            "description": "notes_on_same_chord_key",
            "value": 0,
        },
        {
            "key": "notes_on_beat_rate",
            "name": "Notes on Beat",
            "description": "notes_on_beat_rate",
            "value": 0,
        },
        {
            "key": "note_on_density",
            "name": "Note Density",
            "description": "note_on_density",
            "value": 0,
        },
        {
            "key": "note_variety_rate",
            "name": "Note Variety",
            "description": "note_variety_rate",
            "value": 0,
        },
        {
            "key": "single_notes_rate",
            "name": "Single Notes Rate",
            "description": "single_notes_rate",
            "value": 0,
        },
        {
            "key": "notes_out_of_scale_rate",
            "name": "Notes out of Scale",
            "description": "notes_out_of_scale_rate",
            "value": -1,
        }
    ]
}
AWS_BUCKET_NAME = "aihero"
AWS_DIRECTORY_NAME = "generated_melodies"
AWS_S3_URL = "https://aihero.s3.sa-east-1.amazonaws.com"

# I - TONICA
# II - SUBDOMINANTE
# III - TONICA
# IV - SUBDOMINANTE
# V - DOMINANTE
# VI - TONICA
# VII - DOMINANTE
# 0 - fora do campo harmonico, 1 - tonica, 2 - subdominante, 3 - dominante
FACTOR_TO_HARMONIC_FUNCTION = {
        -11: 3,  # B
        -10: 0,  # A#
        -9: 1,  # A
        -8: 0,  # G#
        -7: 3,  # G
        -6: 0,  # F#
        -5: 2,  # F
        -4: 1,  # E
        -3: 0,  # D#
        -2: 2,  # D
        -1: 0,  # C#
        0: 1,  # C
        1: 0,  # C#
        2: 2,  # D
        3: 0,  # D#
        4: 1,  # E
        5: 2,  # F
        6: 0,  # F#
        7: 3,  # G
        8: 0,  # G#
        9: 1,  # A
        10: 0,  # A#
        11: 3,  # B
    }

CHORD_STRING_TO_FACTOR = {
    "C": 0,
    "C#": -1,
    "Db": -1,
    "D": -2,
    "D#": -3,
    "Eb": -3,
    "E": -4,
    "F": -5,
    "F#": -6,
    "Gb": -6,
    "G": -7,
    "G#": -8,
    "Ab": -8,
    "A": -9,
    "A#": -10,
    "Bb": -10,
    "B": -11
}
