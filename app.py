from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,Predict_pipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else: 
        data = CustomData(
            SquareFeet=request.form.get('square_feet'),
            Bedrooms= request.form.get('bedrooms'),
            Bathrooms= request.form.get('bathrooms'),
            Neighborhood=request.form.get('neighborhood'),
            YearBuilt=request.form.get('year_built')
        )
        predict_df = data.get_data()
        print(predict_df)
        predict_pipeline = Predict_pipeline()
        results =predict_pipeline.prediction(predict_df)
        return render_template('home.html',results = results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)