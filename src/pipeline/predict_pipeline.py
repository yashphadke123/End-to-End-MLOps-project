import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 

class Predict_pipeline:
    def __init__(self):
        pass

    def prediction(self,features):
        try :
            model_path = "artifact/model.pkl"
            preprocessor_path = "artifact/preproccesor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,SquareFeet:int,Bedrooms:int,Bathrooms:int,Neighborhood:str,YearBuilt:int):
        self.square_feet = SquareFeet
        self.bedrooms = Bedrooms
        self.bathrooms = Bathrooms
        self.neighborhood = Neighborhood
        self.yearbuilt = YearBuilt

    def get_data(self):
        try:
            custom_data = {
                "SquareFeet" : [self.square_feet],
                "Bedrooms": [self.bedrooms],
                "Bathrooms": [self.bathrooms],
                "Neighborhood":[self.neighborhood],
                "YearBuilt":[self.yearbuilt]
            }
            return pd.DataFrame(custom_data)
        except Exception as e:
            raise CustomException(e,sys)
                 