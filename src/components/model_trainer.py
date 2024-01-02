import os 
import sys 
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from src.utils import save_object,evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainingconfig:
        model_path = os.path.join('artifact',"model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainingconfig()

    def initiate_training(self,train_array,test_array):
         try:
              logging.info("splitting training and test input data")
              x_train,y_train,x_test,y_test = (
                   train_array[:,:-1],
                   train_array[:,-1],
                   test_array[:,:-1],
                   test_array[:,-1]
              )
              models = {
                   "Linear Regression":LinearRegression(),
                   "SVR": SVR(),
                   "KNeighboursREgression": KNeighborsRegressor(),
                   "DecisionTreeRegression": DecisionTreeRegressor(),
                   "XGBoost":XGBRegressor(),
              }

              model_report:dict = evaluate_models(X_train=x_train,Y_train=y_train,X_test=x_test,Y_test=y_test,models=models)
              best_model_score = max(sorted(model_report.values()))
              best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
              best_model = models[best_model_name]
              logging.info("got the best model")
              save_object(file_path=self.model_trainer_config.model_path,obj=best_model)
              predicted = best_model.predict(x_test)
              R2_score = r2_score(predicted,y_test)
              return R2_score
         except Exception as e:
              raise CustomException(e,sys)