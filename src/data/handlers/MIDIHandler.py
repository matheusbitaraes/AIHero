from abc import ABC, abstractmethod


class MIDIHandlerInterface(ABC):

    @abstractmethod
    def load_from_midi_file(self, file):
        pass

    @abstractmethod
    def export_as_midi(self, spr, chord_array, file_name):
        pass

    @abstractmethod
    def append_track_and_export_as_midi(self, spr, chord_array, midi_file, file_name):
        pass

