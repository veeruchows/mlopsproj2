import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelConfig
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','dataa.csv')
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
    def intiate_data_ingestion(self):
        logging.info('enterd the data ingestion component')
        try:
            df=pd.read_csv(r'data\credit_risk.csv')
            logging.info('enter the dataset as df')
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info('ingestion completed')
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.intiate_data_ingestion()
    data_transformation=DataTransformation()
    train_array,test_array,_=data_transformation.intiate_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_training(train_array,test_array))

