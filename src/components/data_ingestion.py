import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig

from src.components.model_trainer import ModelTraining
from src.components.model_trainer import ModelTrainingconfig

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifact',"train.csv")
    test_data_path :str = os.path.join('artifact',"test.csv")
    raw_data_path:str = os.path.join('artifact',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config =DataIngestionConfig()
    
    def initiate_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('Data\housing_price_dataset.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path))
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train Test Split Initiated")
            trainset,testset = train_test_split(df,train_size=0.75,random_state=None)
            trainset.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            testset.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion complete")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise  CustomException(e,sys)
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,path = data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer = ModelTraining()
    print(model_trainer.initiate_training(train_array=train_arr,test_array=test_arr))
    