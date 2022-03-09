import glob
import os
import time
from datetime import date
from random import random, randrange

import imageio
import matplotlib
import numpy as np
from EVO.engine.Fitness import Fitness
from src.GAN.service.GANService import GANService
from IPython import display

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from src.utils.AIHeroGlobals import SCALED_NOTES_RANGE, TIME_DIVISION, SCALED_NOTES_NUMBER


class AIHeroEVO:
    def __init__(self, config):
        self.gan_service = GANService(config)
        self.fitness_function = Fitness(config["evolutionary_algorithm_configs"]["fitness_function_configs"])

        self.gifs_evidence_dir = config["generated_evidences_dir"]
        self._verbose = config["verbose"]
        self._max_generations = config["evolutionary_algorithm_configs"]["max_generations"]
        self._pop_size = config["evolutionary_algorithm_configs"]["population_size"]
        self._k = config["evolutionary_algorithm_configs"][
            "tournament_percentage"]  # percentage of individuals that will participate of tournament
        self._pc = config["evolutionary_algorithm_configs"]["crossover_probability"]  # crossover probability
        self._pm = config["evolutionary_algorithm_configs"]["child_mutation_probability"]  # child mutation probability
        self._pnm = config["evolutionary_algorithm_configs"][
            "note_change_probability"]  # probability of changing a note when a child is going on mutation
        self.should_generate_gif = self._verbose and not config["enable_parallelization"]

    def generate_melody(self, melody_specs, melody_id=""):
        if self._verbose:
            print(f"\n\nExecuting Evolutionary Algorithm for specs: {melody_specs} ...")

        genetic_algorithm_melody, fitness_array = self.genetic_algorithm(melody_specs, melody_id=melody_id)
        return genetic_algorithm_melody

    def update_fitness_functions(self, fitness_function_configs):
        self.fitness_function.update_configs(fitness_function_configs)

    def genetic_algorithm(self, melody_specs, melody_id=""):
        fitness = np.zeros(self._pop_size)
        best_individual = None
        best_fitness = []
        best_fitness_per_function = []

        filename_prefix = melody_id

        # initiate population
        pop = self.generate_population_with_gan(melody_specs)

        # generation loop
        current_time_min = 0
        for t in range(0, self._max_generations):
            start = time.time()

            # fitness calculation
            for j in range(0, self._pop_size):
                fitness[j], fitness_per_function = self.fitness_function.eval(pop[j, :, :], melody_specs)

            best_fitness.append(fitness[np.argsort(-fitness)[0]])
            best_fitness_per_function.append(fitness_per_function)
            best_individual = pop[np.argsort(-fitness)[0]]

            current_time_min = current_time_min + (time.time() - start) / 60
            if self.should_generate_gif:
                self.generate_and_save_images(epoch=t,
                                              melody=best_individual,
                                              fitness=best_fitness,
                                              fitness_per_function=best_fitness_per_function,
                                              fitness_function_names=self.fitness_function.get_function_names(),
                                              current_time_min=current_time_min,
                                              filename_prefix=filename_prefix)

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
                        children[idm, :, :] = melody[0, :, :, 0]

                new_pop[idx:idx + 2] = children

                idx = idx + 2

            pop = new_pop

        if self.should_generate_gif:
            display.clear_output(wait=True)
            self.generate_gif(filename_prefix=filename_prefix)

            # erase temporary images
            for f in glob.glob(f'.temp/{filename_prefix}*.png'):
                os.remove(f)
        return best_individual, best_fitness

    def generate_population_with_gan(self, melody_specs):
        pop = np.zeros([self._pop_size, SCALED_NOTES_NUMBER, TIME_DIVISION])
        melodies = self.gan_service.generate_melody(specs=melody_specs, num_melodies=self._pop_size)
        pop[:, :, :] = melodies[:, :, :, 0]
        return pop

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
            # on_beat_cut_point = int(TIME_DIVISION / 2)
            child1 = parents[0, :, :].copy()
            child1[:, on_beat_cut_point:] = parents[1, :, on_beat_cut_point:]
            child2 = parents[1, :, :].copy()
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

    def generate_and_save_images(self, epoch, melody, fitness, fitness_per_function, fitness_function_names, current_time_min, filename_prefix):

        # fig, axs = plt.subplots(3)
        fig, axs = plt.subplots(2)
        fig.set_figheight(10)
        fig.set_figwidth(5)
        # fig.set_figwidth(10)
        fig.suptitle(f'Training progress for generation {epoch} ({round(current_time_min, 2)} min)')

        # midi plot
        axs[0].imshow(melody, cmap='Blues')
        axs[0].axis([0, TIME_DIVISION, SCALED_NOTES_RANGE[0],
                     SCALED_NOTES_RANGE[1]])  # necessary for inverting y axis
        axs[0].set(xlabel='Time Division', ylabel='MIDI Notes')

        # fitness plot
        num_measures = len(fitness)
        axs[1].plot(range(num_measures), fitness)
        axs[1].legend(["Fitness: {:03f}".format(fitness[-1])])
        axs[1].set(xlabel='Epochs', ylabel='Fitness')

        # # fitness per function plot
        # num_measures = len(fitness_per_function)
        # axs[2].plot(range(num_measures), fitness_per_function)
        # axs[2].legend(fitness_function_names)
        # axs[2].set(xlabel='Epochs', ylabel='Fitness')

        plt.savefig('.temp/{}_epoch_{:04d}.png'.format(filename_prefix, epoch))
        plt.close()

    def generate_gif(self, filename_prefix=""):
        today = date.today()
        # anim_file = f'{self.gifs_evidence_dir}/{filename_prefix}_{today.strftime("%Y%m%d")}_{time.time_ns()}.mp4'
        anim_file = f'{self.gifs_evidence_dir}/{filename_prefix}_{today.strftime("%Y%m%d")}_{time.time_ns()}.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(f'.temp/{filename_prefix}*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)
