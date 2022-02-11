from random import random, randrange

import numpy as np
from terminalplot import plot

from EVO.engine.Fitness import Fitness
from GAN.service.GANService import GANService
from utils.AIHeroGlobals import TIME_DIVISION, SCALED_NOTES_NUMBER


class AIHeroEVO:
    def __init__(self, config):
        self.gan_service = GANService(config)
        self.fitness_function = Fitness(config["evolutionary_algorithm_configs"]["fitness_function_configs"])

        self._verbose = config["verbose"]
        self._max_generations = config["evolutionary_algorithm_configs"]["max_generations"]
        self._pop_size = config["evolutionary_algorithm_configs"]["population_size"]
        self._k = config["evolutionary_algorithm_configs"][
            "tournament_percentage"]  # percentage of individuals that will participate of tournament
        self._pc = config["evolutionary_algorithm_configs"]["crossover_probability"]  # crossover probability
        self._pm = config["evolutionary_algorithm_configs"]["child_mutation_probability"]  # child mutation probability
        self._pnm = config["evolutionary_algorithm_configs"][
            "note_change_probability"]  # probability of changing a note when a child is going on mutation

    def generate_melody(self, melody_specs):
        if self._verbose:
            print(f"\n\nExecuting Evolutionary Algorithm for specs: {melody_specs} ...")

        genetic_algorithm_melody, fitness_array = self.genetic_algorithm(melody_specs)

        if self._verbose:
            print(f"Melody generated in {self._max_generations} generations, with best fitness: {fitness_array[-1]}")
            print("--------------------------------------------------------------")
            print("FITNESS GRAPH")
            plot(range(len(fitness_array)), fitness_array)
            print("--------------------------------------------------------------")
        return genetic_algorithm_melody

    def genetic_algorithm(self, melody_specs):
        fitness = np.zeros(self._pop_size)
        best_individual = None
        best_fitness = []

        # initiate population
        pop = self.generate_population_with_gan(melody_specs)

        # generation loop
        for t in range(0, self._max_generations):

            # fitness calculation
            for j in range(0, self._pop_size):
                fitness[j] = self.fitness_function.eval(pop[j, :, :], melody_specs)

            best_fitness.append(fitness[np.argsort(-fitness)[0]])
            best_individual = pop[np.argsort(-fitness)[0]]

            idx = 0
            new_pop = pop * 0
            while idx < self._pop_size:
                # selection: tournament
                parents = self.tournament(pop, fitness, 2)

                # crossover: one point
                children = self.crossover(parents, TIME_DIVISION)

                # mutation: note flip
                for idm in range(children.shape[0]):
                    if random() <= self._pm:
                        melody = self.gan_service.generate_melody(specs=melody_specs, num_melodies=1)
                        children[idm, :, :] = melody[0, :, :, 0]  # todo: replace by another mutation method?

                new_pop[idx:idx + 2] = children

                idx = idx + 2

            pop = new_pop

        return best_individual, best_fitness

    def generate_population_with_gan(self, melody_specs):
        pop = np.zeros([self._pop_size, SCALED_NOTES_NUMBER, TIME_DIVISION])
        melodies = self.gan_service.generate_melody(specs=melody_specs, num_melodies=self._pop_size)
        pop[:, :, :] = melodies[:, :, :, 0]
        return pop

    # def generate_random_population(self, melody_specs):
    #     should_apply_rythmic_pattern = False
    #     compass_chord = melody_specs['chord']
    #
    #     total_notes = TIME_DIVISION
    #     octave1 = [x + 12 for x in self.scale][1:]
    #     octave2 = [x + 12 for x in octave1][1:]
    #     expanded_scale = np.append(self.scale, np.append(octave1, octave2))
    #
    #     pop = np.zeros([self._pop_size, total_notes])
    #     for j in range(0, self._pop_size):
    #         indexes = np.random.randint(len(expanded_scale), size=total_notes)
    #         notes_of_scale = np.array(expanded_scale)[indexes.astype(int)]
    #         octave1 = [x + 12 for x in chords[compass_chord]][1:]
    #         octave2 = [x + 12 for x in octave1][1:]
    #         # chord_notes = np.append(self.scale, np.append(octave1, octave2))
    #
    #         for idn in range(0, len(notes_of_scale)):
    #             reduced_scale = expanded_scale[abs(expanded_scale - notes_of_scale[idn - 1]) < self.next_note_range]
    #
    #             if should_apply_rythmic_pattern:
    #                 notes_of_scale[idn] = reduced_scale[
    #                     np.random.randint(len(reduced_scale))]  # insert scale aleatory note
    #             else:
    #                 if random() > self.interval_prob:
    #                     notes_of_scale[idn] = reduced_scale[
    #                         np.random.randint(len(reduced_scale))]  # insert scale aleatory note
    #                 else:
    #                     notes_of_scale[idn] = -1
    #
    #         # Aplica padrões ritmicos
    #         if should_apply_rythmic_pattern:
    #             rp = rhythmic_patterns[str(self.pulses_on_compass)]
    #             rhythmic_pattern = notes_of_scale * 0
    #             for idr in range(0, self.pulses_on_compass):
    #                 a = idr * self.fuse
    #                 b = (idr + 1) * self.fuse
    #                 rhythmic_pattern[a:b] = rp[
    #                     round(randrange(len(rp)))]  # chooses an aleatory  rithmic pattern from pattern database
    #
    #             notes_of_scale[rhythmic_pattern < 0] = -1
    #
    #         pop[j] = notes_of_scale
    #
    #     return pop

    def tournament(self, pop, fitness, num_parents):
        n_participants = round(len(pop) * self._k)

        # get first k% individuals
        participants_idx = np.random.permutation(len(pop))[0:n_participants]
        participants = pop[participants_idx]

        # get num_parents participants with higher fitness
        participants_fit = fitness[participants_idx]
        idx = np.argsort(-1 * participants_fit)[:num_parents]
        parents = participants[idx]

        return parents

    def crossover(self, parents, total_notes):
        if random() <= self._pc:
            on_beat_cut_point = randrange(int(TIME_DIVISION / 4), TIME_DIVISION,
                                          int(TIME_DIVISION / 4))  # the cutting point happens on a beat
            child1 = parents[0, :, :]
            child1[:, on_beat_cut_point:] = parents[1, :, on_beat_cut_point:]
            child2 = parents[1, :, :]
            child2[:, on_beat_cut_point:] = parents[0, :, on_beat_cut_point:]
            children = np.array([child1, child2])
        else:
            children = parents

        return children

    def mutate(self, child):
        for i in range(0, len(child)):
            if child[i] != -1 and random() < self._pnm:
                child[i] = child[i] - randrange(8) + 4
        return child
