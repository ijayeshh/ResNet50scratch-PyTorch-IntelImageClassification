from IntelImageClassification.constants import *
from IntelImageClassification.utils.common import read_yaml, create_directories
from IntelImageClassification.entity.config_entity import (DataIngestionConfig, PrepareResnetModelConfig,TrainingConfig)
import os

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

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_resnet_model = self.config.prepare_resnet_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "seg_train","seg_train")
        print(training_data)
        test_data=os.path.join(self.config.data_ingestion.unzip_dir, "seg_test","seg_test")
        print(test_data)
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            resnet_model_path=Path(prepare_resnet_model.resnet_model_path),
            training_data=Path(training_data),
            test_data=Path(test_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config