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


class MelodySpecs(BaseModel):
    melodic_part: str
    chord: str
    key: str
    tempo: int


class MelodyRequestInput(BaseModel):
    source: str
    melody_specs_list: List[MelodySpecs]


class MelodyRequest(BaseModel):
    id: UUID
    source: str
    melody_specs_list: List[MelodySpecs]
