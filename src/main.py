from __future__ import annotations

from sys import argv

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from json import load

from model.ApiModels import (
    MelodyRequestResponse,
    MelodyRequestInput,
    ApiConfig,
    MelodySpecs,
)
from service.QueueService import QueueService
from utils.AIHeroGlobals import (
    AVAILABLE_SOURCES,
    DEFAULT_MELODY_REQUEST,
    AWS_S3_URL,
    AWS_DIRECTORY_NAME,
)

origins = [
    "http://localhost",
    "http://localhost:3000",
]


def create_app():
    if len(argv) < 2:
        filepath = "src/config.json"
    else:
        filepath = argv[1]

    with open(filepath) as config_file:
        config = load(config_file)
    queue_service = QueueService(config)

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthy")
    def getHealthy():
        return {"message": "ok"}

    @app.post("/melody", response_model=MelodyRequestResponse)
    def getMelodies(
        melody_specs: MelodySpecs = DEFAULT_MELODY_REQUEST, source: str = "evo"
    ):
        if source not in AVAILABLE_SOURCES:
            raise HTTPException(status_code=400, detail="source is not a valid!")
        melody_id = queue_service.add_to_queue(
            MelodyRequestInput(source=source, melody_specs=melody_specs)
        )
        melody_url = f"{AWS_S3_URL}/{AWS_DIRECTORY_NAME}/{melody_id}.mid"
        return MelodyRequestResponse(melody_id=melody_id, melody_url=melody_url)

    @app.get("/melody/{melody_id}")
    def getMelodies(melody_id: str):
        melody_file = queue_service.get_melody_by_id(melody_id)
        if melody_file is None:
            raise HTTPException(status_code=404, detail="melody not found yet")
        return FileResponse(path=melody_file, filename=melody_file)

    app.title = config["title"]
    app.version = config["version"]
    uvicorn.run(app, host=config["uvicorn"]["host"], port=config["uvicorn"]["port"])


if __name__ == "__main__":
    create_app()
