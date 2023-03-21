import glob
import os
import time
from datetime import date
from random import random, randrange

import imageio
import matplotlib
import numpy as np
from IPython import display

from src.EVO.engine.Fitness import Fitness
from src.GEN.service.GENService import GENService

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from src.utils.AIHeroGlobals import SCALED_NOTES_RANGE, TIME_DIVISION, SCALED_NOTES_NUMBER


class AIHeroEVO:
    def __init__(self, config):
        self.gen_service = GENService(config)
        self._use_discriminator_as_fitness_function = \
            config["evolutionary_algorithm_configs"]["fitness_function_configs"]["use_gan_discriminator"]
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
        self.should_generate_gif = config["evolutionary_algorithm_configs"]["should_generate_gif"] and not \
            config["enable_parallelization"]
        self._max_fitness_value = None

    def generate_melody(self, melody_specs, melody_id=""):
        if self._verbose:
            print(f"\n\nExecuting Evolutionary Algorithm for specs: {melody_specs} ...")

        genetic_algorithm_melody, fitness_array = self.genetic_algorithm(melody_specs, melody_id=melody_id)
        return genetic_algorithm_melody

    def update_fitness_functions(self, fitness_function_configs):
        self.fitness_function.update_configs(fitness_function_configs)

    def genetic_algorithm(self, melody_specs, melody_id=""):
        discr_weight = 0.2
        fitness = np.zeros(self._pop_size)
        self._max_fitness_value = self.fitness_function.get_maximum_possible_value() * (1-discr_weight)
        best_individual = None
        best_fitness = []
        best_fitness_per_function = []

        filename_prefix = melody_id

        # initiate population
        pop = self.generate_population_with_gan(melody_specs)
        pop = apply_filters(pop)

        # generation loop
        current_time_min = 0
        for t in range(0, self._max_generations):
            start = time.time()

            # fitness calculation
            if self._use_discriminator_as_fitness_function:
                discr_fitness = self.gen_service.evaluate_melodies_with_discriminator(pop, melody_specs)
            fitness_per_function = ""
            for j in range(0, self._pop_size):
                fitness[j], fitness_per_function = self.fitness_function.eval(pop[j, :, :], melody_specs)
                if self._use_discriminator_as_fitness_function:
                    fitness[j] = fitness[j] * (1-discr_weight) + discr_fitness[j] * discr_weight
            best_fitness_per_function.append(fitness_per_function)

            best_fitness.append(fitness[np.argsort(-fitness)[0]])
            best_individual = pop[np.argsort(-fitness)[0]]

            current_time_min = current_time_min + (time.time() - start) / 60
            if self.should_generate_gif:
                function_names_list = self.fitness_function.get_function_names()
                self.generate_and_save_images(epoch=t,
                                              melody=best_individual,
                                              fitness=best_fitness,
                                              fitness_per_function=best_fitness_per_function,
                                              fitness_function_names=function_names_list,
                                              current_time_min=current_time_min,
                                              filename_prefix=filename_prefix)

            # criterio de parada: fitness é 95% do valor máximo
            if self.is_best_fitness_closest_to_max(best_fitness[-1], 0.95) or self.is_fitness_stable(best_fitness):
                break

            idx = 0
            new_pop = pop * 0
            while idx < self._pop_size:
                # selection: tournament
                parents = self.tournament(pop, fitness, 2)

                # crossover: one point
                children = self.crossover(parents)

                # mutation: note flip
                for idm in range(children.shape[0]):
                    # if random() < self._pnm:
                    #     children[idm, :, :] = self.mutate(children[idm, :, :])
                    # self.mutate(children[idm, :, :])
                    if random() <= self._pm:
                        melody = self.gen_service.generate_melody(specs=melody_specs, num_melodies=1)
                        raw_children = melody[:, :, :, 0]
                        children[idm, :, :] = apply_filters(raw_children)

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
        melodies = self.gen_service.generate_melody(specs=melody_specs, num_melodies=self._pop_size)
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

    def crossover(self, parents):
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

    def generate_and_save_images(self, epoch, melody, fitness, fitness_per_function, fitness_function_names,
                                 current_time_min, filename_prefix):

        fig, axs = plt.subplots(3)
        # fig, axs = plt.subplots(2)
        fig.set_figheight(10)
        # fig.set_figwidth(5)
        fig.set_figwidth(10)
        fig.suptitle(f'Training progress for generation {epoch} ({round(current_time_min, 2)} min)')

        # midi plot
        axs[0].imshow(melody, cmap='Blues')
        axs[0].axis([0, TIME_DIVISION, SCALED_NOTES_RANGE[0],
                     SCALED_NOTES_RANGE[1]])  # necessary for inverting y axis
        axs[0].set(xlabel='Time Division', ylabel='MIDI Notes')

        # fitness plot
        num_measures = len(fitness)
        axs[1].plot(range(num_measures), fitness)
        perc = 100 * fitness[-1]/self._max_fitness_value
        legend = "Fitness: {:.2f}. {:.0f}% of max ({:.2f})".format(fitness[-1], perc, self._max_fitness_value)
        axs[1].legend([legend])
        axs[1].set(xlabel='Epochs', ylabel='Fitness')

        # fitness per function plot
        num_measures = len(fitness_per_function)
        axs[2].plot(range(num_measures), fitness_per_function)
        axs[2].legend(fitness_function_names)
        axs[2].set(xlabel='Epochs', ylabel='Fitness')

        plt.savefig('.temp/{}_epoch_{:04d}.png'.format(filename_prefix, epoch))
        plt.close()

    def generate_gif(self, filename_prefix=""):
        today = date.today()
        anim_file = f'{self.gifs_evidence_dir}/{filename_prefix}_{today.strftime("%Y%m%d")}_{time.time_ns()}.mp4'
        # anim_file = f'{self.gifs_evidence_dir}/{filename_prefix}_{today.strftime("%Y%m%d")}_{time.time_ns()}.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(f'.temp/{filename_prefix}*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

    def is_best_fitness_closest_to_max(self, value, tol_perc):
        return value > 0 and value / self._max_fitness_value > tol_perc

    def is_fitness_stable(self, best_fitness):
        val_number = 15
        fitness_array = best_fitness[-val_number:]
        if len(fitness_array) < val_number:
            return False
        fitness_set = set(fitness_array)
        return len(fitness_set) == 1


# todo: criar uma classe pra isso?
def erode(individual, threshold_length):
    erosion_level = threshold_length + 2

    orig_shape = individual.shape

    # sub matrices of kernel size
    flat_submatrices = np.array([
        individual[i, j:(j + erosion_level)]
        for i in range(orig_shape[0]) for j in range(orig_shape[1])
    ], dtype=object)

    # condition to replace the values - if the kernel equal to submatrix then 255 else 0
    image_erode = np.array([1 if sum(i) > threshold_length - 2 else -1 for i in flat_submatrices])
    image_erode = image_erode.reshape(orig_shape)
    return image_erode


def apply_filters(pop):
    # filter little notes -> make an erosion on x axis
    for i in range(pop.shape[0]):
        pop[i, :, :] = erode(pop[i, :, :], int(TIME_DIVISION / 32))
    return pop
