import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from xgboost import XGBClassifier


def load_data(csv_path:str)-> pd.DataFrame:
    """Load the Higgs-Boson dataset from CSV file"""
    df = pd.read_csv("higgs.csv")
    return df
def data_cleaning(df:pd.DataFrame)-> pd.DataFrame:
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
    return train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, rand_state: int = 42))
def train_model_XGboost():
    """Trains the XGboost model"""
    num_neg = (y_train==0).sum()
    num_pos = (y_train==1).sum()
    scale_pos_weight = num_neg / num_pos

    # Creates the model
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
def feature_label_gen(df:pd.DataFrame, label_column:str="Label")-> tuple[pd.DataFrame, pd.Series]:
    y = df[label_column].map({'s': 1, 'b': 0}).astype(int)
    columns_to_drop = [label_column, 'EventId', 'Weight', 'KaggleSet', 'KaggleWeight']
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    # Check for any remaining object columns and report them
    if object_cols:
        print(f"Warning: The following columns still have object type and will thus be dropped: {object_cols}")
        X = X.drop(columns=object_cols)
    return X, y
    
def model_evaluation(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)
    print("Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"ROC Curve, AUC = {auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristics")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()
    
    return auc_score
def shap_generator():
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_sample)
    # shap.summary_plot(shap_values, X_sample, show=False)
    # plt.tight_layout()
    # plt.savefig("shap_summary.png")
    # plt.close()
    # return shap_values
def main():
    data_path = "higgs.csv"
    test_size=0.2
    rand_state=69
    df=load_data(data_path)
    df_clean=data_cleaning(df)
    X,y=feature_label_gen(df_clean, label_column="Label")
    X_train, X_val, y_train, y_val = data_splitting(X, y, test_size=test_size, rand_state=rand_state)
    model=train_model_XGboost(X_train, y_train, X_val, y_val)
    auc_score=model_evaluation(model, X_val, y_val)
    _ = shap_generator(model, X_val.sample(100))
    model.save_model("higgs_classifier.model")
    
if __name__ == "__main__":
    main()

 
