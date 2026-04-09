from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(filename):
    if filename is None:
        return "No File uploaded"
    if not filename.name.endswith('.csv'): 
        return "Only CSV File allowed"
    try:
        df=pd.read_csv(filename)
        if len(df)==0:
            return "File is empty"
    except Exception as e:
        return f"Error reading filename {e}"
    return df

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_tuning_data(X_train, y_train):

    if len(X_train) > 100000:
        idx = X_train.sample(50000, random_state=42).index
        return X_train.loc[idx], y_train.loc[idx]

    return X_train, y_train



