from random import random, randrange

import numpy as np
from mingus.containers import Bar, Track

from src.EVO.Fitness import Fitness
from mingus.containers.note import Note
from src.EVO.resources import *
import matplotlib.pyplot as plt


class AIHero:
    def __init__(self, central_note, bpm, num_compass, notes_on_compass, scale, chord_sequence,
                 fitness_function=Fitness()):
        self.central_note = int(central_note)
        self.bpm = int(bpm)
        self.num_compass = int(num_compass)
        self.pulses_on_compass = int(notes_on_compass)
        self.scale = scale  # scale with 1 octave above and 1 octave below
        self.chord_sequence = chord_sequence
        self.fuse = 8
        self.max_generations = 150
        self.pop_size = 100
        self.k = 0.30  # percentage of individuals that will participate of tournament
        self.pc = 0.5  # crossover probability
        self.pm = 0.05  # child mutation probability
        self.pnm = 0.05  # probability of changing a note when a child is going on mutation
        self.chord_note_prob = 0.05  # (initial population) probability of a note to be from the chord
        self.next_note_range = 8  # (initial population) max distances (semitones) between two notes in sequence
        self.interval_prob = 0.8 # percentage of interval for population generation
        self.fitness_function = fitness_function

    def generateMelodyArray(self, compassId=None):
        # returns optimization results in a melody array, where each value corresponds to a fuse
        notes = []
        note_track = Track()
        note_bar = Bar()
        rest_count = 0

        if compassId is None:
            for i in range(0, self.num_compass):
                compass_chord = self.chord_sequence[i]  # acorde do inicio do compasso

                # Optimization result
                note_array, fitness_evolution = self.geneticAlgorithm(compass_chord)

                # transform in notes
                for j in range(0, len(note_array)):
                    if note_array[j] > -1:
                        if rest_count > 0:  # add rest before note
                            note_bar.place_rest(32/rest_count)
                        rest_count = 0
                        note = int(self.central_note - 12 + note_array[j])
                        note_bar.place_notes(Note().from_int(note), 32)  # todo: future improvements is to set notes other than fuse
                    else:
                        rest_count += 1
                    if note_bar.is_full():
                        note_track + note_bar
                        note_bar = Bar()  # create a new bar
        else:
            compass_chord = self.chord_sequence[compassId]

            # optimization result
            note_array, fitness_evolution = self.geneticAlgorithm(compass_chord)

            # transform in notes
            for j in range(0, len(note_array)):
                if note_array[j] > -1:
                    if rest_count > 0:  # add rest before note
                        note_bar.place_rest(32/rest_count)
                    rest_count = 0
                    note = int(self.central_note - 12 + note_array[j])
                    note_bar.place_notes(Note().from_int(note), 32)  # todo: future improvements is to set notes other than fuse
                else:
                    rest_count += 1
                if note_bar.is_full():
                    note_track + note_bar
                    note_bar = Bar()  # create a new bar
        note_track + note_bar

        # plot fitness convergence
        plt.plot(fitness_evolution)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.show()

        return note_track

    def geneticAlgorithm(self, compass_chord):
        total_notes = self.pulses_on_compass * self.fuse
        fitness = np.zeros(self.pop_size)
        best_individual = None
        best_fitness = []

        # initiates population
        pop = self.generatePopulation(compass_chord)

        # generation loop
        for t in range(0, self.max_generations):

            # fitness calculation
            for j in range(0, self.pop_size):
                fitness[j] = self.fitness_function.eval(pop[j], chords[compass_chord])

            # print("best fitness:", fitness[np.argsort(-fitness)[0]])
            best_fitness.append(fitness[np.argsort(-fitness)[0]])
            best_individual = pop[np.argsort(-fitness)[0]]

            idx = 0
            new_pop = pop * 0
            while idx < self.pop_size:
                # selection: tournament
                parents = self.tournament(pop, fitness, 2)

                # crossover: one point
                children = self.crossover(parents, total_notes)

                # mutation: note flip
                for idm in range(0, len(children)):
                    if random() <= self.pm:
                        children[idm] = self.mutate(children[idm])

                new_pop[idx:idx + 2] = children

                idx = idx + 2

            pop = new_pop

        return best_individual, best_fitness

    def generatePopulation(self, compass_chord):
        should_apply_rythmic_pattern = False

        total_notes = self.pulses_on_compass * self.fuse
        octave1 = [x + 12 for x in self.scale][1:]
        octave2 = [x + 12 for x in octave1][1:]
        expanded_scale = np.append(self.scale, np.append(octave1, octave2))

        pop = np.zeros([self.pop_size, total_notes])
        for j in range(0, self.pop_size):
            indexes = np.random.randint(len(expanded_scale), size=total_notes)
            notes_of_scale = np.array(expanded_scale)[indexes.astype(int)]
            octave1 = [x + 12 for x in chords[compass_chord]][1:]
            octave2 = [x + 12 for x in octave1][1:]
            # chord_notes = np.append(self.scale, np.append(octave1, octave2))

            for idn in range(0, len(notes_of_scale)):
                reduced_scale = expanded_scale[abs(expanded_scale - notes_of_scale[idn - 1]) < self.next_note_range]

                if should_apply_rythmic_pattern:
                    notes_of_scale[idn] = reduced_scale[np.random.randint(len(reduced_scale))] # insert scale aleatory note
                else:
                    if random() > self.interval_prob:
                        notes_of_scale[idn] = reduced_scale[np.random.randint(len(reduced_scale))] # insert scale aleatory note
                    else:
                        notes_of_scale[idn] = -1

            # Aplica padrões ritmicos
            if should_apply_rythmic_pattern:
                rp = rhythmic_patterns[str(self.pulses_on_compass)]
                rhythmic_pattern = notes_of_scale * 0
                for idr in range(0, self.pulses_on_compass):
                    a = idr * self.fuse
                    b = (idr + 1) * self.fuse
                    rhythmic_pattern[a:b] = rp[
                        round(randrange(len(rp)))]  # chooses an aleatory  rithmic pattern from pattern database

                notes_of_scale[rhythmic_pattern < 0] = -1

            pop[j] = notes_of_scale

        return pop

    def tournament(self, pop, fitness, num_parents):
        n_participants = round(len(pop) * self.k)

        # get first k% individuals
        participants_idx = np.random.permutation(len(pop))[0:n_participants]
        participants = pop[participants_idx]

        # get num_parents participants with higher fitness
        participants_fit = fitness[participants_idx]
        idx = np.argsort(-1 * participants_fit)[:num_parents]
        parents = participants[idx]

        return parents

    def crossover(self, parents, total_notes):
        if random() <= self.pc:
            cut_point = randrange(self.pulses_on_compass) * self.fuse  # the cutting point happens always in a pulse
            child1 = np.append(parents[0][:cut_point], parents[1][cut_point:])
            child2 = np.append(parents[1][:cut_point], parents[0][cut_point:])
            children = np.array([child1, child2])
        else:
            children = parents

        return children

    def mutate(self, child):
        for i in range(0, len(child)):
            if child[i] != -1 and random() < self.pnm:
                child[i] = child[i] - randrange(8) + 4
        return child
