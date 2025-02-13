# utils/data_loader.py
import pandas as pd

def load_data():
    """
    Load the cleaned and processed dataset.
    """
    data_path = "/workspaces/ml_retail/data/pp_new_data.csv"
    return pd.read_csv(data_path)
