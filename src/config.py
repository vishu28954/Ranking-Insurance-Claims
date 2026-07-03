from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    outputs_dir: Path
    seed: int = 42
    test_size: float = 0.25
    stage1_threshold: float = 0.30
    capacity_hours: float = 8.0
