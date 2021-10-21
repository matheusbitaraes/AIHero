from random import random, randrange

import numpy as np
from mingus.containers import Note

from src.EVO.engine.Fitness import Fitness
from src.EVO.resources.resources import *
import mingus.core.chords as chords
import mingus.core.notes as notes

from src.GAN.service.GANService import GANService
from src.utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER, CENTRAL_NOTE_NUMBER


class AIHeroEVO:
    def __init__(self, config):
        # self.central_note = int(central_note)
        # self.bpm = int(bpm)
        # self.num_compass = int(num_compass)
        # self.pulses_on_compass = int(notes_on_compass)
        # self.scale = scale  # scale with 1 octave above and 1 octave below
        # self.chord_sequence = chord_sequence
        # self.fuse = 8
        self.max_generations = config["evolutionary_algorithm_config"]["max_generations"]
        self.pop_size = config["evolutionary_algorithm_config"]["population_size"]
        self.k = config["evolutionary_algorithm_config"][
            "tournament_percentage"]  # percentage of individuals that will participate of tournament
        self.pc = config["evolutionary_algorithm_config"]["crossover_probability"]  # crossover probability
        self.pm = config["evolutionary_algorithm_config"]["child_mutation_probability"]  # child mutation probability
        self.pnm = config["evolutionary_algorithm_config"][
            "note_change_probability"]  # probability of changing a note when a child is going on mutation
        self.gan_service = GANService(config)
        # self.chord_note_prob = 0.05  # (initial population) probability of a note to be from the chord
        # self.next_note_range = 8  # (initial population) max distances (semitones) between two notes in sequence
        # self.interval_prob = 0.8 # percentage of interval for population generation
        self.fitness_function = Fitness()

    def generate_melody(self, melody_specs):
        genetic_algorithm_melody = self.genetic_algorithm(melody_specs)
        return genetic_algorithm_melody

    def genetic_algorithm(self, melody_specs):
        fitness = np.zeros(self.pop_size)
        best_individual = None
        best_fitness = []

        # initiates population
        pop = self.generate_population_with_gan(melody_specs)

        # generation loop
        compass_notes = get_chord_notes(melody_specs)
        for t in range(0, self.max_generations):

            # fitness calculation
            for j in range(0, self.pop_size):
                fitness[j] = self.fitness_function.eval(pop[j, :, :], compass_notes)

            # print("best fitness:", fitness[np.argsort(-fitness)[0]])
            best_fitness.append(fitness[np.argsort(-fitness)[0]])
            best_individual = pop[np.argsort(-fitness)[0]]

            idx = 0
            new_pop = pop * 0
            while idx < self.pop_size:
                # selection: tournament
                parents = self.tournament(pop, fitness, 2)

                # crossover: one point
                children = self.crossover(parents, TIME_DIVISION)

                # mutation: note flip
                for idm in range(0, len(children)):
                    if random() <= self.pm:
                        children[idm] = self.mutate(children[idm])

                new_pop[idx:idx + 2] = children

                idx = idx + 2

            pop = new_pop

        return best_individual, best_fitness

    def generate_population_with_gan(self, melody_specs):
        pop = np.zeros([self.pop_size, SCALED_NOTES_NUMBER, TIME_DIVISION])
        melodies = self.gan_service.generate_melody(specs=melody_specs, num_melodies=self.pop_size)
        pop[:, :, :] = melodies[:, :, :, 0]
        return pop

    def generate_random_population(self, melody_specs):
        should_apply_rythmic_pattern = False
        compass_chord = melody_specs['chord']

        total_notes = TIME_DIVISION
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
                    notes_of_scale[idn] = reduced_scale[
                        np.random.randint(len(reduced_scale))]  # insert scale aleatory note
                else:
                    if random() > self.interval_prob:
                        notes_of_scale[idn] = reduced_scale[
                            np.random.randint(len(reduced_scale))]  # insert scale aleatory note
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


def get_chord_notes(melody_specs):
    chord = melody_specs["chord"]
    key = melody_specs["key"]
    note_list = chords.triad(chord, key)
    note_numbers = []
    for n in note_list:
        note_int = notes.note_to_int(n)
        note_numbers.append(int(note_int + SCALED_NOTES_NUMBER/2))
    return note_numbers



