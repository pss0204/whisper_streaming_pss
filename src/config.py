
from dataclasses import dataclass, field
from typing import Literal



@dataclass
class Config:
    beam_size: int = 1
    min_chunk_size: float = 10.0