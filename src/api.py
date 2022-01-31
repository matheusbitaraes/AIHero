from __future__ import annotations

from json import load as jload

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.service.QueueService import QueueService
from src.utils.AIHeroGlobals import AVAILABLE_SOURCES, DEFAULT_MELODY_REQUEST, AWS_S3_URL, AWS_DIRECTORY_NAME

with open('config.json') as config_file:
    config = jload(config_file)

MIDI_FILENAME = 'temp_midi'

app = FastAPI(title="AIHero", version="0.0.1")

queue_service = QueueService(config)


@app.post('/melody')
def getMelodies(melody_specs_list: list = DEFAULT_MELODY_REQUEST, source: str = "evo"):
    if source not in AVAILABLE_SOURCES:
        raise HTTPException(status_code=400, detail="source is not a valid!")
    melody_request = {
        "source": source,
        "melody_specs_list": melody_specs_list
    }
    melody_id = queue_service.add_to_queue(melody_request)
    return {
        "message": f"melody with id {melody_id}requested",
        "melody_id": melody_id,
        "melody_url": f"{AWS_S3_URL}/{AWS_DIRECTORY_NAME}/{melody_id}.mid"
    }


@app.get('/melody/{melody_id}')
def getMelodies(melody_id: str):
    melody_file = queue_service.get_melody_by_id(melody_id)
    if melody_file is None:
        raise HTTPException(status_code=404, detail="melody not found yet")
    return FileResponse(path=melody_file, filename=melody_file)


if __name__ == "__main__":
    uvicorn.run(app, port=8080)
