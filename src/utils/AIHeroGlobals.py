MIDI_NOTES_NUMBER = 128
SCALED_NOTES_RANGE = [0, 48]  # MIN, MAX
SCALED_NOTES_NUMBER = SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]
CENTRAL_NOTE_NUMBER = 60
TIME_DIVISION = 64
# TRAIN_DATA_REPLICAS = 40
NOTES_IN_OCTAVE = 12
AVAILABLE_SOURCES = {"evo", "train", "gan"}
DEFAULT_MELODY_REQUEST = {
    "harmony_specs": [
        {
            "melodic_part": "RELAXATION",
            "chord": "C",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "TENSION",
            "chord": "F",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "F",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "TENSION",
            "chord": "F",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "TENSION",
            "chord": "G",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "TENSION",
            "chord": "F",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RELAXATION",
            "chord": "C",
            "key": "C",
            "tempo": 120,
        },
        {
            "melodic_part": "RETAKE",
            "chord": "G",
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
