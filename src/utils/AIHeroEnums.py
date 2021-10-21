from enum import Enum


class MelodicPart(Enum):
    X = 'RELAXATION'
    Y = 'TENSION'
    Z = 'RETAKE'

    def get_from_value(self, value):
        if value == "RELAXATION":
            return MelodicPart.X
        if value == "TENSION":
            return MelodicPart.Y
        if value == "RETAKE":
            return MelodicPart.X
