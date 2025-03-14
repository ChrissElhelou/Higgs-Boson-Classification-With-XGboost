import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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
    """Train-test splits the dataset"""
    return train_test_split()
def train_model_XGboost():
    """Trains the XGboost model"""
    num_neg = (y_train==0).sum()
    num_pos = (y_train==1).sum()
    scale_pos_weight = num_neg / num_pos

    # Create the model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc'
    )
    
    # Basic fit without early stopping
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    else:
        model.fit(X_train, y_train)
    return model
def model_evaluation():
    pass
def shap_generator():
    pass
def main():
    data_path = "higgs.csv"
    test_size=0.2
    rand_state=69
    df=load_data(data_path)
    df=data_cleaning(df)
    
    pass
if __name__ == "__main__":
    main()

 
