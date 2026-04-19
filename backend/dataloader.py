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

import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
