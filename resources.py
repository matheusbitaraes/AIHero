# possible scales
scales = {
    'major_scale': [0, 2, 4, 5, 7, 9, 11, 12],
    'minor_scale': [0, 2, 3, 4, 7, 9, 11, 12],
    'dorian_scale': [0, 2, 3, 5, 7, 9, 10, 12],
    'phrygian_scale': [0, 1, 3, 5, 7, 8, 10, 12],
    'lydian_scale': [0, 2, 4, 6, 7, 9, 11, 12],
    'mixolydian_scale': [0, 2, 4, 5, 7, 9, 10, 12],
    'aeolian_scale': [0, 2, 3, 5, 7, 8, 10, 12],
    'locrian_scale': [0, 1, 3, 5, 6, 8, 10, 12],
    'lydian_domiant_scale': [0, 2, 4, 6, 7, 9, 10, 12],
    'super_locrian_scale': [0, 1, 3, 4, 6, 8, 10, 12],
    'minor_pentatonic_scale': [0, 3, 5, 7, 10, 12],
    'major_pentatonic_scale': [0, 2, 4, 7, 9, 12],
    'minor_blues_scale': [0, 3, 5, 6, 7, 10, 12],
    'major_blues_scale': [0, 2, 3, 4, 7, 9, 12]
}
# https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies for midi keys

chords = {
    '1': [0, 4, 7],
    '2': [2],
    '3': [4],
    '4': [5, 9, 12],
    '5': [7, 11, 14],
    '6': [],
    '7': [],
    '8': [],
    '1m': [],
    '2m': [],
    '3m': [],
    '4m': [],
    '5m': [],
    '6m': [],
    '7m': [],
    '1-7': [0, 4, 7, 10],
    '2-7': [2],
    '3-7': [4],
    '4-7': [5, 9, 12, 15],
    '5-7': [7, 11, 14, 16],
    '6-7': [],
    '7-7': [],
    '8-7': [],
}

rhythmic_patterns = {  # define alguns padroes de onde as notas serão tocadas e onde haverão silencios
    '4': [  # 8 fusas entre tempos
         [-1, -1, -1, -1, -1, -1, -1, -1],
         [1, -1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, 1, -1, -1],
         [-1, -1, -1, -1, 1, -1, -1, -1],
         [1, -1, -1, 1, -1, -1, 1, -1],
         [1, -1, -1, 1, -1, 1, -1, -1],
         [1, -1, 1, -1, -1, -1, -1, -1],
         [-1, -1, -1, 1, -1, -1, -1, -1]
    ]
}# padroes ritmicos a cada intervalo de tempo


bass_lines = {
    'blues': [[0, 0], [4, 8], [7, 16], [9, 24], [7, 30]] #[[nota relativa, instante que sera tocada]...]
    # 'blues': [[0, 0], [0, 8], [0, 16], [0, 24], [0, 30]] #[[nota relativa, instante que sera tocada]...]
}