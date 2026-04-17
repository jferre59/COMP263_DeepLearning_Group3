import json
import os 
import numpy as np
import keras


os.chdir(os.path.dirname(os.path.abspath(__file__))) #Set current directory to backend folder

def predPipeline(file): #Function predPipeline, takes in a string for the file name then loads the json data and runs pipeline to predict fraud or non-fraud
    with open(f"uploads/{file}", 'r') as file: 
        data = json.load(file) #Load specified json file

    values = list(data.values()) #Convert data values to a list of 30 values
    pred = np.array(values) #Convert list of values to an array of values
    pred_values = pred.reshape(1, 30) #Reshape the array from (30,) to (1,30) for input to autoencoder

    autoencoder = keras.models.load_model("model/autoencoder.keras") #Load autoencoder model
    clf = keras.models.load_model("model/clf.keras") #Load Dense NN Classifier model

    prediction = autoencoder.predict(pred_values) #Run autoencoder prediction

    r_error = np.mean(np.square(pred_values - prediction), axis=1) #Calculate reconstruction error
    pred_values_aug = np.column_stack((r_error, pred_values)) #Augment data with reconstruction error

    final_pred = clf.predict(pred_values_aug) #Run deep nn classifier prediction of error augmented data

    return final_pred[0][0] #Retrun the value of the prediction (is float value of 0.0 or 1.0)


#predPipeline("predict.json") #Debugging line for testing

    