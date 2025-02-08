from IntelImageClassification.constants import *
from IntelImageClassification.utils.common import read_yaml, create_directories
from IntelImageClassification.entity.config_entity import (DataIngestionConfig, PrepareResnetModelConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_prepare_resnet_model_config(self) -> PrepareResnetModelConfig:
        config = self.config.prepare_resnet_model
        
        create_directories([config.root_dir])

        prepare_resnet_model_config = PrepareResnetModelConfig(
            root_dir=Path(config.root_dir),
            resnet_model_path=Path(config.resnet_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES
        )

        return prepare_resnet_model_config