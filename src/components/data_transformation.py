import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import convert_target_variable

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    data_transformation_config=os.path.join('artifacts','processor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config_obj= DataTransformationConfig()
    def get_transformed_data(self):
        try:
            num_features=["Age", "Income", "Amount", "Status", "Percent_income", "Cred_length"]
            impute_median_features = ["Emp_length"]
            impute_mean_features = ["Rate"]
            cat_features = ["Home", "Intent"]
            num_pipeline=Pipeline(
                steps=[
                    ('sc',StandardScaler())

                ]
            )
            med_pipeline=Pipeline(
                steps=[
                    ('median',SimpleImputer(strategy='median')),
                    

                    ('sc',StandardScaler())
                ]
            )
            mean_pipeline=Pipeline(
                steps=[
                    ('mean',SimpleImputer(strategy='mean')),
                    ('sc',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('onehot',OneHotEncoder()),
                    ('sc',StandardScaler(with_mean=False))

                ]
            )
            preprocessor=ColumnTransformer(
                transformers=[
                    ('num',num_pipeline,num_features),
                    ('med',med_pipeline,impute_median_features),
                    ('me',mean_pipeline,impute_mean_features),
                    ('cat',cat_pipeline,cat_features)


                ]
            )
            logging.info('encoding completed')
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def intiate_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('train test data read sucessfully')
            preprocess_obj= self.get_transformed_data()
            train_df['Default']=convert_target_variable(train_df['Default'])
            test_df['Default']=convert_target_variable(test_df['Default'])
            logging.info('converted target variable to binary format')
            target_variable=['Default']
            not_imp_variables=['Id']
            input_features_train_df = train_df.drop(columns= target_variable + not_imp_variables, axis=1)
            target_train_df = train_df[target_variable]

            input_features_test_df = test_df.drop(columns= target_variable + not_imp_variables, axis=1)
            target_test_df = test_df[target_variable]
            input_features_train_df_array = preprocess_obj.fit_transform(input_features_train_df)
            input_features_test_df_array = preprocess_obj.transform(input_features_test_df)
            logging.info('applying preprocessing on train and test')
            train_array=np.c_[input_features_train_df_array,np.array(target_train_df)]
            test_array=np.c_[input_features_test_df_array,np.array(target_test_df)]
            save_object(
                file_path=self.data_transformation_config_obj.data_transformation_config,obj=preprocess_obj
            )
            logging.info('save preprocessing object')
            return(
                train_array,
                test_array,
                self.data_transformation_config_obj.data_transformation_config
            )
        except Exception as e:
            raise CustomException(e,sys)


