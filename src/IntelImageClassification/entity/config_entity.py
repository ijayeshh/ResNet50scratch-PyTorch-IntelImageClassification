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

@dataclass(frozen=False)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    resnet_model_path: Path
    training_data: Path
    test_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list