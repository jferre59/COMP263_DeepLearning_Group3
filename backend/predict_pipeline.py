import json
import os 
import numpy as np
import keras
import joblib


os.chdir(os.path.dirname(os.path.abspath(__file__))) #Set current directory to backend folder

def predPipeline(file): #Function predPipeline, takes in a string for the file name then loads the json data and runs pipeline to predict fraud or non-fraud
    with open(f"uploads/{file}", 'r') as file: 
        data = json.load(file) #Load specified json file

    values = list(data.values()) #Convert data values to a list of 30 values
    pred = np.array(values).reshape(1, 30)

    #Scaling amount and time to match training
    scaler = joblib.load("model/scaler.pkl")
    pred_scaled = pred.copy()
    pred_scaled[:, 0]  = scaler["time"].transform(pred[:, [0]])   #Time index 0
    pred_scaled[:, 29] = scaler["amount"].transform(pred[:, [29]]) #Amount is index 29

    autoencoder = keras.models.load_model("model/autoencoder.keras") #Load autoencoder model
    clf = keras.models.load_model("model/clf.keras") #Load Dense NN Classifier model

    reconstruction = autoencoder.predict(pred_scaled, verbose=0)
    r_error = np.mean(np.square(pred_scaled - reconstruction), axis=1)
    pred_aug = np.column_stack((r_error, pred_scaled))

    final_pred = clf.predict(pred_aug, verbose=0)
    return final_pred[0][0] #Retrun the value of the prediction (is float value of 0.0 or 1.0)


#predPipeline("predict.json") #Debugging line for testing

    