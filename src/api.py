from json import load as jload

from flask import Flask, redirect, url_for, request

from src.service.AIHeroService import AIHeroService

with open('config.json') as config_file:
    config = jload(config_file)

app = Flask(__name__)
app.config["DEBUG"] = True

ai_hero_service = AIHeroService(config)


@app.route('/train/melodies', methods=['POST'])
def getMelodiesFromGANTrainData():
    bars = request.form
    print(bars)
    aa = generate_compositions_with_train_data(bars)
    return True


@app.route('/gan/melodies', methods=['POST'])
def getMelodiesFromGAN(bars):
    return ai_hero_service.generate_compositions(bars)


@app.route('/melodies', methods=['POST'])
def getMelodies(bars):
    return ai_hero_service.generate_compositions(bars)


app.run()
