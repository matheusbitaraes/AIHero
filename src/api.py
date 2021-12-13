from json import load as jload

from flask import Flask, request, send_file
from flask_cors import CORS

from src.service.AIHeroService import AIHeroService

with open('config.json') as config_file:
    config = jload(config_file)

MIDI_FILENAME = 'temp_midi'

app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

ai_hero_service = AIHeroService(config)


@app.route('/train/melodies', methods=['POST'])
def getMelodiesFromGANTrainData():
    api_input = request.json
    ai_hero_data = ai_hero_service.generate_compositions_with_train_data(api_input)
    return prepare_and_send_response(ai_hero_data)


@app.route('/gan/melodies', methods=['POST'])
def getMelodiesFromGAN():
    api_input = request.json
    ai_hero_data = ai_hero_service.generate_GAN_compositions(api_input)
    return prepare_and_send_response(ai_hero_data)


@app.route('/melodies', methods=['POST'])
def getMelodies():
    api_input = request.json
    ai_hero_data = ai_hero_service.generate_compositions(api_input)
    return prepare_and_send_response(ai_hero_data)


def prepare_and_send_response(ai_hero_data):
    ai_hero_data.export_as_midi(file_name=MIDI_FILENAME)
    new_file = open(f'{MIDI_FILENAME}_1.mid',
                    'rb')  # todo: only first composition is returned now. This can be a problem someday
    return send_file(new_file, mimetype='audio/mid', as_attachment=True, attachment_filename=f'{MIDI_FILENAME}.mid')


app.run()
