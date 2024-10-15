from CNN_Classifier.config.configuration import ConfigurationManager
from CNN_Classifier.components.model_train import ModelTrain
from CNN_Classifier import logger

STAGE_NAME = "Prepare Model and Train"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self,train,val):
        config = ConfigurationManager()
        model_train_config = config.get_model_train_config()
        model_trainer = ModelTrain(config=model_train_config)
        model_trainer.get_model_train()
        model_trainer.train_model(train,val)
        model_trainer.save_model()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
