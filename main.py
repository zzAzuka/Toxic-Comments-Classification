from CNN_Classifier import logger
from CNN_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
import numpy

STAGE_NAME = "Data Ingestion Stage"

try:
   logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   train, val, test = data_ingestion.main()
   logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e