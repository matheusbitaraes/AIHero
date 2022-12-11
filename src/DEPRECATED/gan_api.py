from __future__ import annotations

import time
from json import load as jload

import uvicorn
from fastapi import FastAPI, HTTPException

from src.GEN.service.GENService import GENService
from src.utils.AIHeroHelper import MelodicPart

with open('config.json') as config_file:
    config = jload(config_file)

app = FastAPI(title="AIHeroGAN", version="0.0.1")

gan_service = GENService(config)


@app.get('/train/{melodic_type_value}/{num_epochs}', status_code=200)
def trainGan(melodic_type_value: str, num_epochs: int, should_generate_gif: bool = False):
    try:
        part = MelodicPart(melodic_type_value)
        t = time.time()
        gan_service.train_model(part=part.value, should_generate_gif=should_generate_gif, num_epochs=num_epochs)
        return {"message": f"gan type {melodic_type_value} trained! Took {time.time() - t:.2f}s for {num_epochs} epochs"}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{melodic_type_value} is not a valid type")


if __name__ == "__main__":
    uvicorn.run(app, port=8079)
