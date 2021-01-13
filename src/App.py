from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from mingus.containers import Bar, Track
import mingus.extra.lilypond as LilyPond
from mingus.midi import fluidsynth
import time

from src.AIHero import AIHero
from src.AISynth import AISynth
from src.Fitness import Fitness
from src.resources import *

MENU_TITLES = ['Notes of chord', 'Notes on tempo', 'Notes interval', 'Notes repetition', 'Notes Pitch', 'Note variety',
               'Note sequency']


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
        self.melody_notes = Track()
        self.fitness_function = Fitness(0, 0, 0, 0, 0, 0, 0, 0)
        self.stop = False

    def build(self):
        box = BoxLayout(orientation='horizontal', spacing=100)

        # parameter configuration
        box_parameter_setup = BoxLayout(orientation='vertical', spacing=10)
        box_parameter = BoxLayout(orientation='vertical', spacing=10)
        box_parameter_fitness = BoxLayout(orientation='vertical', spacing=10)
        box_parameter.add_widget((self.buildButtons()))
        box_parameter.add_widget(self.buildTextInput('central_note', self.central_note), 1)
        box_parameter.add_widget(self.buildTextInput('bpm', self.bpm), 1)
        box_parameter.add_widget(self.buildTextInput('num_compass', self.num_compass), 1)
        box_parameter.add_widget(self.buildTextInput('pulses_on_compass', self.pulses_on_compass), 1)
        box_parameter.add_widget(self.buildDropdownInput('scale', scales), 1)
        box_parameter.add_widget(self.buildTextInput('chord_sequence', self.chord_sequence), 1)
        box_parameter_fitness.add_widget(self.buildSliderInput(MENU_TITLES[0], self.fitness_function.w1))
        box_parameter_fitness.add_widget(self.buildSliderInput(MENU_TITLES[1], self.fitness_function.w2))
        box_parameter_fitness.add_widget(self.buildSliderInput(MENU_TITLES[2], self.fitness_function.w3))
        box_parameter_fitness.add_widget(self.buildSliderInput(MENU_TITLES[3], self.fitness_function.w4))
        box_parameter_fitness.add_widget(self.buildSliderInput(MENU_TITLES[4], self.fitness_function.w5, -12, 12))
        box_parameter_fitness.add_widget(self.buildSliderInput(MENU_TITLES[5], self.fitness_function.w6))
        box_parameter_fitness.add_widget(self.buildSliderInput(MENU_TITLES[6], self.fitness_function.w7))
        box_parameter_setup.add_widget(box_parameter_fitness)
        box_parameter_setup.add_widget(box_parameter)

        # defines musical sheet exhibition part
        image = Image(source='melody_sheet.png', allow_stretch=True, keep_ratio=False)
        Clock.schedule_interval(lambda dt: image.reload(), 0.25)

        box.add_widget(box_parameter_setup, 1)
        box.add_widget(image, 0)
        return box

    def buildSliderInput(self, name, value, min=-100, max=100):
        def on_value_change(instance, v):
            if name == MENU_TITLES[0]:
                self.fitness_function.w1 = v
            if name == MENU_TITLES[1]:
                self.fitness_function.w2 = v
            if name == MENU_TITLES[2]:
                self.fitness_function.w3 = v
            if name == MENU_TITLES[3]:
                self.fitness_function.w4 = v
            if name == MENU_TITLES[4]:
                self.fitness_function.w5 = v
            if name == MENU_TITLES[5]:
                self.fitness_function.w6 = v
            if name == MENU_TITLES[6]:
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
        start_button = Button(text="Play", on_release=self.startExecution, size_hint=(1, 1))
        stop_button = Button(text="Stop", on_release=self.stopExecution, size_hint=(1, 1))
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

        print(self.fitness_function.w1, self.fitness_function.w2, self.fitness_function.w3, self.fitness_function.w4,
              self.fitness_function.w5, self.fitness_function.w6, self.fitness_function.w7)

        synth = AISynth()

        # inicializa classe de otimização
        scale = scales[self.scaleName]
        ai_hero = AIHero(self.central_note, self.bpm, self.num_compass, self.pulses_on_compass, scale,
                         self.chord_sequence, self.fitness_function)


        # mudar depois
        fluidsynth.init('./fluidsynth/sf2/FluidR3_GM.sf2')

        # retorna melodia (uma lista de notas(nota, velocidade), onde cada nota dura o tempo de uma fusa
        track = Track()
        first_execution = True  # indicates if it is the first time that the algorithm is being executed
        initial_time = 0
        for i in range(0, self.num_compass):
            fuse = 60 / (self.bpm * 8)  # duration of 'fuse' note in seconds
            t = time.time()
            compass_melody = ai_hero.generateMelodyArray(compassId=i)
            elapsed_time = time.time() - t
            print("Melody optimization {} took {}s".format(i, round(elapsed_time, 2)))

            if first_execution:  # check if is first execution
                initial_time = round(1000 * (elapsed_time + 1 + fuse * 64))
                print("waiting {} for execution".format(round(initial_time/1000, 2)))
                first_execution = False

            for bar in compass_melody:
                track.add_bar(bar)

            initial_time = synth.schedule_melody(
                initial_time, fuse, self.chord_sequence[i], self.central_note, compass_melody)

        # save music sheet into file
        LilyPond.to_png(LilyPond.from_Track(track), 'melody_sheet')

    def stopExecution(self, button):
        self.stop = True


AIHeroUI().run()
