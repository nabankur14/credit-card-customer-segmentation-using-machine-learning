import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess the customer data: drop unnecessary columns and scale features.
    
    Args:
        df (pd.DataFrame): Raw customer data.
        
    Returns:
        pd.DataFrame: Scaled data as a DataFrame.
        pd.DataFrame: Original data (cleaned) for profiling.
    """
    # Drop irrelevant columns (ID columns)
    cols_to_drop = ['Sl_No', 'Customer Key']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Check for duplicates (though notebook said none, good practice to keep)
    # df_clean.drop_duplicates(inplace=True) 
    # For this specific dataset, we preserve rows as per analysis but in production this is a standard step.
    
    # Scaling
    scaler = StandardScaler()
    df_scaled_array = scaler.fit_transform(df_clean)
    
    # Convert back to DataFrame for convenience in further steps
    df_scaled = pd.DataFrame(df_scaled_array, columns=df_clean.columns)
    
    return df_scaled, df_clean
