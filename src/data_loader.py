import pandas as pd
import os

def load_data(filepath):
    """
    Load data from an Excel file.
    
    Args:
        filepath (str): Path to the Excel file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    try:
        df = pd.read_excel(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e
