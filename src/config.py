
from dataclasses import dataclass, field
from typing import Literal



@dataclass
class Config:
    beam_size: int = 5
    min_chunk_size: float = 0.1