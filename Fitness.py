class Fitness:
    def __init__(self, w1=50, w2=50, w3=100, w4=-25, w5=12, w6=50, w7=25, w8=25):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8

    def eval(self, note_sequence, chord_notes):
        f1 = self.notesOnSameChordKey(note_sequence, chord_notes)  # porcentagem de notas no mesmo key do acorde (0-1)
        f2 = self.startsWithChordNote(note_sequence, chord_notes)  # Melodia começa numa nota do acorde (0,1)
        f3 = self.intervalsEvaluation(
            note_sequence)  # Porcentagem de intervalos. quanto maior, mais intervalos. 100 (0-1)
        f4 = self.evalNoteRepetitions(note_sequence)  # numero de repetições maior que 2 de uma nota (0,1)
        f5 = 100 * self.evalPitch(note_sequence, self.w5 + 12)  # função para escolher solos mais graves ou agudos (0-100)
        f6 = self.noteVarietyEvaluation(note_sequence)  # variedade de notas (0-1)
        f7 = self.notesInSequenceEvaluation(
            note_sequence)  # porcentagem de notas em sequencia (ascendente ou descendente, o que for maior) (0-1)
        f8 = self.notesOnTime(note_sequence)  # porcentagem de notas nos tempos ou contra tempos (0-1)
        fitness = self.w1 * f1 + self.w2 * f2 + self.w3 * f3 + self.w4 * f4 + \
                  f5 + self.w6 * f6 + self.w7 * f7 + self.w8 * f8
        return fitness

    def notesOnSameChordKey(self, note_sequence, chord_notes):
        number_notes_in_chord = 0
        total_notes = len(note_sequence[note_sequence != -1])
        for i in range(0, len(note_sequence)):
            if note_sequence[i] != -1 and note_sequence[i] in chord_notes:
                number_notes_in_chord = number_notes_in_chord + 1
        if total_notes != 0:
            return number_notes_in_chord / total_notes
        else:
            return 0

    def startsWithChordNote(self, note_sequence, chord_notes):
        if note_sequence[0] in chord_notes:
            return 1
        else:
            return 0

    def intervalsEvaluation(self, note_sequence):
        intervals = len(note_sequence[note_sequence == -1])
        perc = intervals / len(note_sequence)
        return perc
        # if 0.1 < perc < 0.5:
        #     return 1
        # else:
        #     return 0

    def evalNoteRepetitions(self, note_sequence):
        notes = note_sequence[note_sequence != -1]
        for i in range(2, len(notes)):
            if notes[i] == notes[i - 1] == notes[i - 2]:
                return 1
        return 0

    def evalPitch(self, note_sequence, target):
        notes = note_sequence[note_sequence != -1]
        notes_close_to_pitch = 0
        for i in range(1, len(notes)):
            if abs(notes[i] - target) < 12:
                notes_close_to_pitch += 1
        if len(notes) > 0:
            return notes_close_to_pitch / len(notes)
        else:
            return 0

    def noteVarietyEvaluation(self, note_sequence):
        notes = note_sequence[note_sequence != -1]
        unique = len(set(notes))
        total = len(notes)
        if total > 0:
            return unique / len(notes)
        else:
            return 0

    def notesInSequenceEvaluation(self, note_sequence):
        notes = note_sequence[note_sequence != -1]
        upper_progression = 0
        lower_progression = 0
        for i in range(1, len(notes)):
            if notes[i] > notes[i - 1]:
                upper_progression += 1
            elif notes[i] < notes[i - 1]:
                lower_progression += 1

        if len(notes) > 0:
            return max([upper_progression / len(notes), lower_progression / len(notes)])
        else:
            return 0

    def notesOnTime(self, note_sequence):
        notes = note_sequence[note_sequence != -1]
        n = 0
        for i in range(0,len(note_sequence)):
            if i % 4 == 0:
                if note_sequence[i] > -1:
                    n += 1

        if len(notes) > 0:
            return n/len(notes)
        else:
            return 0
