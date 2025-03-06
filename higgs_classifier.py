import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load the Higgs-Boson dataset from CSV file"""
    df = pd.read_csv("higgs.csv")
    return df
def data_cleaning():
    """Cleans the Higgs-Boson dataset by replacing missing values and converting column objects to numerical values"""
    df = df.copy()
    df.replace(-999.0, np.nan, inplace=True)
    #Remove any rows missing with values and convert object columns
    df.dropna(inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col])
    print(df.dtypes.value_counts())
    print("hi")
    return df
def data_splitting():
    pass
def train_model_XGboost():
    pass
def model_evaluation():
    pass
def shap_generator():
    pass
def main():
    pass
if __name__ == "__main__":
    main()

 
