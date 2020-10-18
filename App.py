import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput

from AIHero import AIHero
from AISynth import AISynth
from Fitness import Fitness
from resources import *


class AIHeroUI(App):
    def __init__(self, **kwargs):
        super(AIHeroUI, self).__init__(**kwargs)
        self.central_note = 60
        self.bpm = 90
        self.num_compass = 2
        self.pulses_on_compass = 4
        self.scaleName = 'minor_blues_scale'
        self.chord_sequence = ['1-7', '4-7', '1-7', '1-7', '4-7', '4-7', '1-7', '1-7', '5-7', '4-7', '1-7', '5-7']
        self.chord_transition_time = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
        self.melody_notes = []
        self.plot = MeshLinePlot(color=[1, 0, 0, 1])
        self.fitness_function = Fitness(0, 0, 0, 0, 0, 0, 0, 0)
        self.stop = False
        self.graph = Graph(xlabel='tempo', ylabel='Notas', x_ticks_minor=5,
                           x_ticks_major=25, y_ticks_major=1,
                           y_grid_label=True, x_grid_label=True, padding=5,
                           x_grid=True, y_grid=True, xmin=0, xmax=self.num_compass * self.pulses_on_compass * 8,
                           ymin=self.central_note - 20, ymax=self.central_note + 20)

    def build(self):
        box = BoxLayout(orientation='vertical', spacing=100)

        # configuração de parâmetros
        box_parameter_setup = BoxLayout(orientation='horizontal', spacing=10)
        box_parameter = BoxLayout(orientation='vertical', spacing=10)
        box_parameter_fitness = BoxLayout(orientation='vertical', spacing=10)
        # title = Label(text='Definição de parametros do problema', font_size=30)
        # box_parameter_setup.add_widget(title, 10)
        box_parameter.add_widget((self.buildButtons()))
        box_parameter.add_widget(self.buildTextInput('central_note', self.central_note), 1)
        box_parameter.add_widget(self.buildTextInput('bpm', self.bpm), 1)
        box_parameter.add_widget(self.buildTextInput('num_compass', self.num_compass), 1)
        box_parameter.add_widget(self.buildTextInput('pulses_on_compass', self.pulses_on_compass), 1)
        box_parameter.add_widget(self.buildDropdownInput('scale', scales), 1)
        box_parameter.add_widget(self.buildTextInput('chord_sequence', self.chord_sequence), 1)
        box_parameter_fitness.add_widget(self.buildSliderInput('Notas nos acordes', self.fitness_function.w1))
        box_parameter_fitness.add_widget(self.buildSliderInput('Notas no tempo', self.fitness_function.w8))
        box_parameter_fitness.add_widget(self.buildSliderInput('Intervalos entre notas', self.fitness_function.w3))
        box_parameter_fitness.add_widget(self.buildSliderInput('Numero de repetições', self.fitness_function.w4))
        box_parameter_fitness.add_widget(self.buildSliderInput('Pitch', self.fitness_function.w5, -12, 12))
        box_parameter_fitness.add_widget(self.buildSliderInput('Variedade de notas', self.fitness_function.w6))
        box_parameter_fitness.add_widget(self.buildSliderInput('Notas em sequencia', self.fitness_function.w7))
        box_parameter_setup.add_widget(box_parameter)
        box_parameter_setup.add_widget(box_parameter_fitness)

        # define parte de exibição de gráficos e definiçao de objetivos
        box_output = BoxLayout(orientation='vertical')
        self.graph.add_plot(self.plot)
        Clock.schedule_interval(self.updatePlot, 1)
        box_output.add_widget(self.graph)

        box.add_widget(box_parameter_setup, 1)
        box.add_widget(box_output, 0)
        return box

    def updatePlot(self, dt):
        i = 0
        points = []
        self.graph.xmax = self.pulses_on_compass * int(self.num_compass) * 8  # len(self.melody_notes) #
        for note in self.melody_notes:
            if note.note is not None:
                points.append([i, note.note])
            i += 1
        self.plot.points = points  # [(i, self.melody_notes[i].note) for i in range(0, len(self.melody_notes))]

    def buildSliderInput(self, name, value, min=-100, max=100):
        def on_value_change(instance, v):
            if name == 'Notas nos acordes':
                self.fitness_function.w1 = v
            if name == 'Notas no tempo ou contra tempo':
                self.fitness_function.w8 = v
            if name == 'Intervalos entre notas':
                self.fitness_function.w3 = v
            if name == 'Numero de repetições':
                self.fitness_function.w4 = v
            if name == 'Pitch':
                self.fitness_function.w5 = v
            if name == 'Variedade de notas':
                self.fitness_function.w6 = v
            if name == 'Notas em sequencia':
                self.fitness_function.w7 = v
            label.text = name + " (" + str(round(v)) + ")"

        box = BoxLayout()
        box.size_hint_y = None
        box.height = 60
        label_txt = name + " (" + str(value) + ")"
        label = Label(text=label_txt, size_hint=(0.4, 1))
        variable = Slider(min=min, max=max, value=value, size_hint=(0.6, 1))
        variable.bind(value=on_value_change)
        box.add_widget(label)
        box.add_widget(variable)
        return box

    def buildButtons(self):
        start_button = Button(text="Tocar", on_release=self.startExecution, size_hint=(1, 1))
        stop_button = Button(text="Parar", on_release=self.stopExecution, size_hint=(1, 1))
        box = BoxLayout()
        box.size_hint_y = None
        box.height = 60
        box.add_widget(start_button)
        box.add_widget(stop_button)
        return box

    def buildTextInput(self, name, value):
        def on_text_change(instance, v):
            if name == 'central_note':
                self.central_note = v
            if name == 'bpm':
                self.bpm = v
            if name == 'num_compass':
                self.num_compass = v
            if name == 'pulses_on_compass':
                self.pulses_on_compass = v
            if name == 'chord_sequence':
                self.chord_sequence = v

        box = BoxLayout()
        box.size_hint_y = None
        box.height = 60
        label = Label(text=name, size_hint=(0.4, 1))
        variable = TextInput(text=str(value), multiline=False, size_hint=(0.6, 1))  # TODO aqui
        variable.bind(text=on_text_change)
        box.add_widget(label)
        box.add_widget(variable)
        return box

    def buildDropdownInput(self, name, list):

        def setValue(x):
            self.scaleName = x

        box = BoxLayout()
        box.size_hint_y = None
        box.height = 60
        label = Label(text=name, size_hint=(0.4, 1))
        dropdown = DropDown()
        for value in list:
            btn = Button(text=value, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))
            dropdown.add_widget(btn)
        mainbutton = Button(text=self.scaleName, size_hint=(0.6, 1))
        mainbutton.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: (setattr(mainbutton, 'text', x), setValue(x)))
        box.add_widget(label)
        box.add_widget(mainbutton)

        return box

    chord_name = None

    def startExecution(self, button):
        self.central_note = int(self.central_note)
        self.bpm = int(self.bpm)
        self.num_compass = int(self.num_compass)
        self.pulses_on_compass = int(self.pulses_on_compass)
        self.chord_transition_time = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
        self.melody_notes = []

        print(self.fitness_function.w1, self.fitness_function.w2, self.fitness_function.w3, self.fitness_function.w4,
              self.fitness_function.w5, self.fitness_function.w6, self.fitness_function.w7)

        synth = AISynth()
        initial_time = 2000  # delay antes de começar

        # while self.stop is False:
        #     # inicializa classe de otimização
        #     scale = scales[self.scaleName]
        #     ai_hero = AIHero(self.central_note, self.bpm, self.num_compass, self.pulses_on_compass, scale,
        #                      self.chord_sequence, self.fitness_function)
        #
        #     # retorna melodia (uma lista de notas(nota, velocidade), onde cada nota dura o tempo de uma fusa
        #     for i in range(0, self.num_compass):
        #         total_duration = 60 / self.bpm * self.pulses_on_compass  # time of a beat(ms) * number of beats
        #         fuse = 60 / (self.bpm * 8)  # duration of 'fuse' note in seconds
        #         compass_melody = ai_hero.generateMelodyArray(compassId=i)
        #         if i == 3:
        #             ai_hero.fitness_function.w7 = 100
        #         if i == 4:
        #             ai_hero.fitness_function.w7 = - 100
        #         self.melody_notes = compass_melody
        #         initial_time = synth.schedule_melody(
        #             initial_time, fuse, self.chord_sequence[i], self.central_note, compass_melody)
        #     time.sleep(1)

        # inicializa classe de otimização
        scale = scales[self.scaleName]
        ai_hero = AIHero(self.central_note, self.bpm, self.num_compass, self.pulses_on_compass, scale,
                         self.chord_sequence, self.fitness_function)

        # retorna melodia (uma lista de notas(nota, velocidade), onde cada nota dura o tempo de uma fusa
        for i in range(0, self.num_compass):
            total_duration = 60 / self.bpm * self.pulses_on_compass  # time of a beat(ms) * number of beats
            fuse = 60 / (self.bpm * 8)  # duration of 'fuse' note in seconds
            compass_melody = ai_hero.generateMelodyArray(compassId=i)
            if i == 3:
                ai_hero.fitness_function.w7 = 100
            if i == 4:
                ai_hero.fitness_function.w7 = - 100
            self.melody_notes = np.append(self.melody_notes, compass_melody)
            initial_time = synth.schedule_melody(
                initial_time, fuse, self.chord_sequence[i], self.central_note, compass_melody)

        # # linha de baixo
        # bass_line = bass_lines['blues']
        # ext_bass_line = bass_line
        # num_lines = len(bass_line)
        # # replica para o numero de compassos
        # for ic in range(1, self.num_compass):
        #     for ib in range(0, num_lines):
        #         bass_time = bass_line[ib][1] + 32 * ic
        #         ext_bass_line.append([bass_line[ib][0], bass_time])
        # ib = 0

        # Execução da musica

    def stopExecution(self, button):
        self.stop = True


AIHeroUI().run()
