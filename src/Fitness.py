import numpy as np

from src.resources import *


class Fitness:
    def __init__(self, w1=50, w2=25, w3=100, w4=-25, w5=12, w6=50, w7=25, w8=220, scale='minor_blues_scale'):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8
        self.scale = scale

    def eval(self, note_sequence, chord_notes):
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
        f5 = 0
        f6 = 0
        f7 = 0
        f8 = 0
        if self.w1 != 0:  # percentage of notes in the same key of chord (0-1)
            f1 = self.notesOnSameChordKey(note_sequence,
                                          chord_notes)
        if self.w2 != 0:  # percetage of notes in beat of off beat (0-1)
            f2 = self.notesOnTempo(note_sequence)

        if self.w3 != 0:  # percentage of intervals (the more intervals, the higher is the grade) (0-1)
            f3 = self.intervalsEvaluation(
                note_sequence)

        if self.w4 != 0:  # number of repetitions is higher than 2 (0,1)
            f4 = self.evalNoteRepetitions(note_sequence)

        if self.w5 != 0:  # check if pitch is according to a reference note
            f5 = 100 * self.evalPitch(note_sequence,
                                      self.w5 + 12)
        if self.w6 != 0:  # check note variety (0-1)
            f6 = self.noteVarietyEvaluation(note_sequence)  # variedade de notas (0-1)

        if self.w7 != 0:  # percentage of notes in sequency (asc or desc) (0-1)
            f7 = self.notesInSequenceEvaluation(
                note_sequence)

        if self.w8 != 0:  # measure the proximity of the sequence in relation to known musical licks. 1: note
            # seguence is totally equal musical licks. 0: note sequence has nothing to do with musical licks(0-1)
            f8 = self.knownLicksProximity(
                note_sequence)

        fitness = self.w1 * f1 + self.w2 * f2 + self.w3 * f3 + self.w4 * f4 + f5 + self.w6 * f6 + self.w7 * f7 + self.w8 * f8

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

    def notesOnTempo(self, note_sequence):
        notes = note_sequence[note_sequence != -1]
        n = 0
        for i in range(0, len(note_sequence)):
            if i % 4 == 0:
                if note_sequence[i] > -1:
                    n += 1

        if len(notes) > 0:
            return n / 8  # 8 is the highest possible number of notes on beat/off beat
        else:
            return 0

    # TODO: implement this function to evaluate a proximity to known licks, in order to create a "muscular memory" into the model
    def knownLicksProximity(self, note_sequence):
        known_licks_of_scale = known_licks[self.scale]
        notes = note_sequence[note_sequence != -1]
        if len(notes) == 0:
            return
        global_similarity = 0
        for lick in known_licks_of_scale:
            lick = np.array(lick)
            flat_lick = lick[lick != -1]  # get only the notes, not the interval
            lick_dynamic = flat_lick - np.append(flat_lick[0], flat_lick[0:len(
                flat_lick) - 1])  # get the dynamic (from one note to next)
            notes_dynamic = notes - np.append(notes[0], notes[0:len(notes) - 1])
            if self.is_slice_in_list(lick_dynamic, notes_dynamic):
                global_similarity += 0.5

        # transform known lick to array
        return global_similarity

    def is_slice_in_list(self, s, l):  # check if s is slice of l
        # ref: https://stackoverflow.com/questions/20789412/check-if-all-elements-of-one-array-is-in-another-array
        len_s = len(s)
        return any((s == l[i:len_s + i]).all() for i in range(len(l) - len_s + 1))

    # backup
    # def getSequenceSimilarity(self, arr, seq):
    #     """ Find sequence in an array using cv2.
    #      """
    #     # source code: https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    #
    #     # Run a template match with input sequence as the template across
    #     # the entire length of the input array and get scores.
    #     S = cv2m(np.array(arr).astype('uint8'), np.array(seq).astype('uint8'), cv2.TM_CCOEFF_NORMED)
    #     return np.max(S)
