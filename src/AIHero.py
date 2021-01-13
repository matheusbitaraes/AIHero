from random import random, randrange

import numpy as np

from src.Fitness import Fitness
from mingus.containers.note import Note
from src.resources import *


class AIHero:
    def __init__(self, central_note, bpm, num_compass, notes_on_compass, scale, chord_sequence,
                 fitness_function=Fitness()):
        self.central_note = int(central_note)
        self.bpm = int(bpm)
        self.num_compass = int(num_compass)
        self.pulses_on_compass = int(notes_on_compass)
        self.scale = scale  # escala com 1 oitava acima e 1 abaixo
        self.chord_sequence = chord_sequence
        self.fuse = 8
        self.max_generations = 150
        self.pop_size = 100
        self.k = 0.30  # porcentagem de individuos que participam do torneio
        self.pc = 0.5  # probabilidade de crossover
        self.pm = 0.05  # probabilidade de mutação de um filho
        self.pnm = 0.02  # probabilidade de mutação de uma nota, quando o filho por mutadolugar
        self.chord_note_prob = 0.05  # probabilidade de uma nota da melodia (população inicial) ser uma nota pertencente ao acorde
        self.next_note_range = 6  # distancia máxima em semitons entre duas notas (população inicial)
        self.fitness_function = fitness_function

    def generateMelodyArray(self, compassId=None):  # retorna um array com notas por fusa, resultado da otimização
        notes = []

        if compassId is None:
            # itera nos compassos
            for i in range(0, self.num_compass):
                compass_chord = self.chord_sequence[i]  # acorde do inicio do compasso

                # pega array resultado da otimização
                note_array = self.geneticAlgorithm(compass_chord)

                # transforma em notas
                for j in range(0, len(note_array)):
                    # p = p + 1
                    if note_array[j] > -1:
                        note = int(self.central_note - 12 + note_array[j])
                        # plota nota no gráfico
                        # plt.scatter(p, note, color='b')
                        # plot_notes.append(note)
                        # plot_interval.append(p)
                    notes.append(Note().from_int(note))
                else:
                    notes.append(Note().empty())

                # plt.plot(plot_interval, plot_notes, color='b')
                # plt.pause(0.005)
            # plt.ioff()
        else:
            compass_chord = self.chord_sequence[compassId]  # acorde do inicio do compasso
            # pega array resultado da otimização
            note_array = self.geneticAlgorithm(compass_chord)

            # transforma em notas
            for j in range(0, len(note_array)):
                # p = p + 1
                if note_array[j] > -1:
                    note = int(self.central_note - 12 + note_array[j])
                    # plota nota no gráfico
                    # plt.scatter(p, note, color='b')
                    # plot_notes.append(note)
                    # plot_interval.append(p)
                    notes.append(Note().from_int(note))
                else:
                    notes.append(Note().empty())

            # plt.plot(plot_interval, plot_notes, color='b')
            # plt.pause(0.005)

        return notes

    def geneticAlgorithm(self, compass_chord):
        total_notes = self.pulses_on_compass * self.fuse
        fitness = np.zeros(self.pop_size)

        # inicia população
        pop = self.generatePopulation(compass_chord)

        # loop geracional
        for t in range(0, self.max_generations):

            # cálculo do fitness
            for j in range(0, self.pop_size):
                fitness[j] = self.fitness_function.eval(pop[j], chords[compass_chord])

            # print("best fitness:", fitness[np.argsort(-fitness)[0]])
            best_fitness = fitness[np.argsort(-fitness)[0]]
            best_individual = pop[np.argsort(-fitness)[0]]

            idx = 0
            new_pop = pop * 0
            while idx < self.pop_size:
                # seleção: torneio
                parents = self.tournament(pop, fitness, 2)

                # cruzamento: 1 ponto de corte
                children = self.crossover(parents, total_notes)

                # mutação: mudança de nota
                for idm in range(0, len(children)):
                    if random() <= self.pm:
                        children[idm] = self.mutate(children[idm])

                new_pop[idx:idx + 2] = children

                idx = idx + 2

            pop = new_pop

        # print("best fitness:", best_fitness)
        # print("best fitness:", best_individual)
        return best_individual

    def generatePopulation(self, compass_chord):
        total_notes = self.pulses_on_compass * self.fuse
        octave1 = [x + 12 for x in self.scale][1:]
        octave2 = [x + 12 for x in octave1][1:]
        expanded_scale = np.append(self.scale, np.append(octave1,
                                                         octave2))  # adicionando -1, que seria "sem nota" e duas oitavas a mais

        pop = np.zeros([self.pop_size, total_notes])
        for j in range(0, self.pop_size):
            indexes = np.random.randint(len(expanded_scale), size=total_notes)
            notes_of_scale = np.array(expanded_scale)[indexes.astype(int)]
            octave1 = [x + 12 for x in chords[compass_chord]][1:]
            octave2 = [x + 12 for x in octave1][1:]
            # chord_notes = np.append(self.scale, np.append(octave1, octave2))

            for idn in range(0, len(notes_of_scale)):
                reduced_scale = expanded_scale[abs(expanded_scale - notes_of_scale[idn - 1]) < self.next_note_range]
                # reduced_chord_notes = chord_notes[abs(chord_notes - notes_of_scale[idn - 1]) < self.next_note_range]
                # if random() <= self.chord_note_prob and len(
                #         reduced_chord_notes) > 0:  # coloca alguma nota pertencente ao acorde para ser tocada
                #     notes_of_scale[idn] = reduced_chord_notes[np.random.randint(len(reduced_chord_notes))]
                # elif len(reduced_scale) > 0:
                #     notes_of_scale[idn] = reduced_scale[np.random.randint(len(reduced_scale))]

                # pega nota aleatória dentro da escala
                notes_of_scale[idn] = reduced_scale[np.random.randint(len(reduced_scale))]

            # Aplica padrões ritmicos
            rp = rhythmic_patterns[str(self.pulses_on_compass)]
            rhythmic_pattern = notes_of_scale * 0
            for idr in range(0, self.pulses_on_compass):
                a = idr * self.fuse
                b = (idr + 1) * self.fuse
                rhythmic_pattern[a:b] = rp[
                    round(randrange(len(rp)))]  # escolhe um padrão ritmico aleatorio da base de padrões

            notes_of_scale[rhythmic_pattern < 0] = -1
            pop[j] = notes_of_scale

        return pop

    def tournament(self, pop, fitness, num_parents):
        n_participants = round(len(pop) * self.k)
        # fitness_array = np.zeros([2, n_participants])
        # Pega os k% primeiros individuos, selecionados aleatoriamente
        participants_idx = np.random.permutation(len(pop))[0:n_participants]
        participants = pop[participants_idx]

        # pega os num_parents participantes com maior fitness
        participants_fit = fitness[participants_idx]
        idx = np.argsort(-1 * participants_fit)[:num_parents]
        parents = participants[idx]

        return parents

    def crossover(self, parents, total_notes):
        if random() <= self.pc:
            cut_point = randrange(self.pulses_on_compass) * self.fuse  # o ponto de corte é sempre em um pulso.
            child1 = np.append(parents[0][:cut_point], parents[1][cut_point:])
            child2 = np.append(parents[1][:cut_point], parents[0][cut_point:])
            children = np.array([child1, child2])
        else:
            children = parents

        return children

    def mutate(self, child):
        # mutação construtiva. Pega duas notas em sequencia e coloca uma nota intermediária entre elas
        # note1 = None
        # note2 = None
        # id1 = None
        # id2 = None
        # for i in range(0, len(child)):
        #     if child[i] != -1:
        #         if note1 is None:
        #             note1 = child[i]
        #             id1 = i
        #         else:
        #             note2 = child[i]
        #             id2 = i
        #             break
        # if id1 is not None and id2 is not None and note1 is not None and note2 is not None:
        #     id3 = id1 + randrange(id2-id1)
        #     if note1 > note2:
        #         note3 = note1 - randrange(note1 - note2)
        #         child[id3] = note3
        #     elif note1 < note2:
        #         note3 = note1 + randrange(note2 - note1)
        #         child[id3] = note3
        for i in range(0, len(child)):
            if child[i] != -1 and random() < self.pnm:
                child[i] = child[i] - randrange(8) + 4
        return child

# gerador melódico
# estratégia baseada na ordem das notas
# critério de parada: quando ocorrer takeover
# função de fitness: garayacevedo2005.pdf
# soma pesos com funções de fitness que são:
# porcentagem de notas no mesmo key do acorde (100)
# Melodia começa numa nota do acorde (50)
# Porcentagem de intervalos menor que uma terça 100 (??)
# numero de repetições maior que 2 de uma nota (-25)
