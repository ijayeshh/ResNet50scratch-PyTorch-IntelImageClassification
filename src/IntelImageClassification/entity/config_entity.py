from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareResnetModelConfig:
    root_dir: Path
    resnet_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_classes: int