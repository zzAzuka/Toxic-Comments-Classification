from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    features: int
    output_len: int

@dataclass(frozen=True)
class ModelTrainConfig:
    root_dir: Path
    trained_model_path: Path
    features: int
    loss: str
    optimizer: str
    epochs: int