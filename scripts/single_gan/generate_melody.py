from json import load as jload

from src.service.AIHeroService import AIHeroService
from src.test.utils.test_utils import build_big_request_body
from src.utils.AIHeroGlobals import TIME_DIVISION


def execute_function_on_data(self, f):
    data, chords = f(self._spr_data, self._transposition_factor)
    self._spr_data = data
    self._transposition_factor = chords


def add_chord(data, chords):
    data[:, 0, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    # data[:, 4, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    # data[:, 7, 0:TIME_DIVISION:int(TIME_DIVISION / 4), :] = 1
    return data, chords


work_dir = 'scripts/single_gan'
with open(f'{work_dir}/config.json') as config_file:
    config = jload(config_file)

# avalia para cada acorde e ve se deu certo
service = AIHeroService(config)

harmony_specs = build_big_request_body().melody_specs.harmony_specs
data = service.generate_GEN_compositions(harmony_specs, melody_id="id")

# add a chord being played on every SPR
data.execute_function_on_data(add_chord)

data.export_as_midi(f'{work_dir}/evidences/generated_melody')
