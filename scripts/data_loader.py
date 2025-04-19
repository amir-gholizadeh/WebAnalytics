# scripts/data_loader.py
import pandas as pd
import os

def load_data(file_path):
    """Load dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type based on extension
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def clean_data(df):
    """Perform initial data cleaning."""
    # Drop duplicates
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    return df