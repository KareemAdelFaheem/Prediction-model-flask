from flask import Flask
import joblib
import numpy as np
import pandas as pd
import math
import pickle
import xgboost
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=joblib.load(open('xgb_model.pkl','rb'))

# mice_imputer = joblib.load('mice_imputer.pkl')


with open('encoders.pkl', 'rb') as f:
    lencoders = pickle.load(f)

scaler = joblib.load('scaler.pkl')

def encode_data(data):
    for col in data.select_dtypes(include=['object']).columns:
        if col in lencoders:
            # Apply the encoder to the new data
            data[col] = lencoders[col].transform(data[col])
    return data


@app.route('/')
def predict():
    input_data = pd.DataFrame([["Albury", "14.0", "18.0", "10.2", "5.0", "1.2", "S", "37", "SE", "S", "7", "15", "95",
                                "80", "1005.2", "1003.8", "7", "8", "15.5", "17.2", "1"]],
                              columns=["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                                       "WindGustDir", "WindGustSpeed", "WindDir9am",
                                       "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
                                       "Pressure9am", "Pressure3pm", "Cloud9am",
                                       "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"])

    # Encode the data
    encoded = encode_data(input_data)
    # imputed_data = mice_imputer.transform(encoded)
    final_data= scaler.transform(encoded)


    prediction= model.predict(final_data)
    # print(final_data)

    return "prediction result %s "% prediction


if __name__ == '__main__':
    app.run(host='0.0.0.0')


