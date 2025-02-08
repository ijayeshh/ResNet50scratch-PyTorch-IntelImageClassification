from IntelImageClassification.config.configuration import ConfigurationManager
from IntelImageClassification import logger
from IntelImageClassification.components.prepare_resnet_model import block,ResNet
import torch

STAGE_NAME = "Prepare base model"

class PrepareResnetModelTrainingPipeline:
    def __init__(self):
        pass
    def ResNet50(self,img_channel, num_classes):
        return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)

    def main(self):
        net= self.ResNet50(img_channel=3, num_classes=6)
        config = ConfigurationManager()
        prepare_resnet_model_config=config.get_prepare_resnet_model_config()
        ppath=prepare_resnet_model_config.resnet_model_path
        torch.save(net,ppath)

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareResnetModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e