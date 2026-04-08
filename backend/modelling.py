from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import regularizers
from dataloader import Preprocessor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score,average_precision_score,precision_recall_curve
import os

os.chdir(os.path.dirname(os.path.abspath(__file__))) #Set current working directory to directory this file is located in

processor = Preprocessor('data/creditcard.csv') #Create an instacne of the Preprocessor class and pass the file location/name to it
train, val, test, y_test, shape = processor.pipeline() #Run the pipeline from the preprocessor and get the training, testing, and validation data as results along with shape of training data

input_dim = shape[1] #Input dimension is second dimension value of test data shape

input_layer = Input(shape=(input_dim,)) #Input layer, dimensions based on second dimension of training data shape
encoder = Dense(14, activation="relu")(input_layer) #Encoder layer 1, 14 neurons and relu activation
encoder = Dense(7, activation="relu")(encoder) #Encoder layer 2, 7 neurons and relu activation

decoder = Dense(7, activation="relu")(encoder) #Decoder layer 1, 7 neurons and relu activation
decoder = Dense(14, activation="relu")(decoder) #Decoder layer 2, 14 neurons and relu activation

output_layer = Dense(input_dim, activation="linear")(decoder) #Output layer, output dim same as input activation is linear

autoencoder = Model(inputs=input_layer, outputs=output_layer) #Build full autoencoder based on layers developed above

autoencoder.compile( #Compile autoencoder with adam optimizer and mean squared error as loss
    optimizer="adam",
    loss="mean_squared_error"
)

autoencoder.summary() #Print summary of autoencoder model

early_stop = EarlyStopping( #Declare early stopping for the model if there is no improvement in validation loss after 5 epochs, restore best weights
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = autoencoder.fit( #Fit the autoencoder to the data for 50 epochs with early stopping, batchs of size 256
    train, train,
    epochs=100,
    batch_size=32,
    validation_data=(val, val),
    callbacks=[early_stop],
    shuffle=True
)

#reconstruction of testing data
reconstructions = autoencoder.predict(test)

#MSE error
mse = np.mean(np.power(test - reconstructions, 2), axis=1)

pr_auc = average_precision_score(y_test, mse)
print("PR-AUC:", pr_auc)

autoencoder.save("model/autoencoder.keras")