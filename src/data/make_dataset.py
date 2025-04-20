import pandas as pd

def load_data(filepath):
    """Load the loan data from CSV file"""
    loan_data_backup = pd.read_csv(filepath)
    loan_data = loan_data_backup.copy()
    return loan_data

def save_data(loan_data, output_path):
    """Save processed data to file"""
    loan_data.to_csv(output_path, index=False)