import sys 
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preproccesor_path = os.path.join('artifact',"preproccesor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
    def get_transformer_obj(self):
        try:
            categoric_columns = ["Neighborhood"]
            numeric_columns = ["SquareFeet","Bedrooms","Bathrooms","YearBuilt"]
            numeric = Pipeline([("Input",SimpleImputer(missing_values=np.nan,strategy="mean")),
                    ("Scaling",StandardScaler())])
            categoric = Pipeline([("Input", SimpleImputer(fill_value=np.nan,strategy="most_frequent")),
            ("Encoding",OneHotEncoder(sparse=False)),("Scaling",StandardScaler())])
            logging.info("Categoric and Numeric Pipeines created")
            preprocessor = ColumnTransformer([("Categoric",categoric,categoric_columns),
                                  ("Numeric",numeric,numeric_columns)])
            logging.info("Preprocessor done")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)
            logging.info("Reading train and test data")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_transformer_obj()
            target_column_name = "Price"
            train_df_input = train_df.drop(columns=[target_column_name],axis=1)
            train_df_output = train_df[target_column_name]
            test_df_input = test_df.drop(columns=[target_column_name],axis=1)
            test_df_output = test_df[target_column_name]
            logging.info("Applying preprocessing")
            train_ar_input=preprocessing_obj.fit_transform(train_df_input)
            test_ar_input=preprocessing_obj.transform(test_df_input)  
            train_arr = np.c_[train_ar_input,np.array(train_df_output)]
            test_arr = np.c_[test_ar_input,np.array(test_df_output)]
            logging.info("Saving preprocessing object")
            save_object(file_path = self.data_transformation_config.preproccesor_path ,obj= preprocessing_obj )
            return (train_arr,test_arr,self.data_transformation_config.preproccesor_path)
        except Exception as e:
            raise CustomException(e,sys)
