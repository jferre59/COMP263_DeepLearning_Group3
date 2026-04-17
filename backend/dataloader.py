"""
Script to load data from data/creditcard.csv.
Usage in other scripts / notebooks: 
    from dataloader import load_data
    df = load_data()
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

<<<<<<< HEAD
os.chdir(os.path.dirname(os.path.abspath(__file__))) #Set current working directory to directory this file is located in

class Preprocessor:
    def __init__(self, file):
        self.df = pd.read_csv(file)
        self.X = None
        self.y = None
        self.X_scaled = None
        self.X_train_scaled = None
        self.X_fraud = None
        self.X_normal = None
        self.shape = None
        self.X_train = None
        self.X_val = None 

    def explore(self): #Explores the data before preprocessing
        print("-----------------------Data Exploration---------------------------------")
        print(self.df.info())
        print("Counts for each Class:")
        print(self.df['Class'].value_counts())
        print(f"Shape of Dataset: {self.df.shape}")

        self.y = self.df['Class'] #Get the targets for the dataset from the class column
        self.X = self.df.drop(['Class', 'Time'], axis=1) #Get the features excluding class and time which are dropped

        self.X_normal = self.X[self.y == 0] #Get only the non-fraud cases for training

    def scaling(self):
        print("-----------------------Data Scaling---------------------------------")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.X_train_scaled = scaler.fit_transform(self.X_normal)

    def data_split(self):
        print("-----------------------Data Splitting---------------------------------")
        #Train test split with 80% training data and 20% testing data
        self.X_train, self.X_val = train_test_split(self.X_train_scaled, test_size=0.20, random_state=88)

        #Print the new shape of the testing and validation sets
        print(f"Shape of Training Set: {self.X_train.shape}")
        print(f"Shape of Validation Set: {self.X_val.shape}")

        self.shape = self.X_train.shape #Save the shape of the training data for use in determining the input dimensions of the autoencoder

    def pipeline(self): #Pipeline for running all preprocessing steps
        self.explore() #Explore data to understand data
        self.scaling() #Scale the ammount column, all other columns scaled already via PCA
        self.data_split() #Split dataset into testing 20% and training 80%

        return self.X_train, self.X_val, self.X_scaled, self.y.to_numpy(), self.shape


'''
#Debugging code for testing Preprocessor pipeline
processor = Preprocessor('data/creditcard.csv')

train, val, test, y_test, shape = processor.pipeline()

print(type(train))
print(type(test))
print(type(val))
print(type(y_test))
print(type(shape))
'''
=======
import pandas as pd

def load_data():
    """
    Loads the credit card fraud detection dataset from data/creditcard.csv.
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    print("Loading data from data/creditcard.csv...")
    
    df = pd.read_csv("data/creditcard.csv")

    print(f"   Total rows: {len(df)}")
    print(f"   Fraud count: {df['Class'].sum():,} ({df['Class'].mean() * 100:.2f}%)")

    return df
>>>>>>> b84d8e2 (Created load_data() for import ot other modules)
