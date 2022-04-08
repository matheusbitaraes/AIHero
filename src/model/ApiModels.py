from typing import List
from uuid import UUID

from pydantic import BaseModel


class UvicornConfig(BaseModel):
    host: str
    port: int


class ApiConfig(BaseModel):
    title = "AI Hero API"
    version: str
    uvicorn: UvicornConfig


class MelodyRequestResponse(BaseModel):
    melody_id: UUID
    melody_url: str


class HarmonySpecs(BaseModel):
    melodic_part: str
    chord: int
    key: str
    tempo: int


class FitnessFunction(BaseModel):
    key: str
    name: str
    description: str
    value: float = None
    weight: float = 0


class MelodySpecs(BaseModel):
    harmony_specs: List[HarmonySpecs]
    evolutionary_specs: List[FitnessFunction]


class MelodyRequestInput(BaseModel):
    source: str
    melody_specs: MelodySpecs


class MelodyRequest(BaseModel):
    id: UUID
    source: str
    melody_specs: MelodySpecs
