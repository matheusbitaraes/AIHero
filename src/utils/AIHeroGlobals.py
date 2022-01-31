MIDI_NOTES_NUMBER = 128
SCALED_NOTES_RANGE = [0, 48]  # MIN, MAX
SCALED_NOTES_NUMBER = SCALED_NOTES_RANGE[1] - SCALED_NOTES_RANGE[0]
CENTRAL_NOTE_NUMBER = 60
TIME_DIVISION = 64
# TRAIN_DATA_REPLICAS = 40
NOTES_IN_OCTAVE = 12
AVAILABLE_SOURCES = {"evo", "train", "gan"}
DEFAULT_MELODY_REQUEST = [
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
]
AWS_BUCKET_NAME = "aihero"
AWS_DIRECTORY_NAME = "generated_melodies"
AWS_S3_URL = "https://aihero.s3.sa-east-1.amazonaws.com"