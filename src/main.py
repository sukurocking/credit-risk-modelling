# ./src/main.py

import pandas as pd
from data.make_dataset import load_data, save_data
from features.build_features import preprocess_data, split_data
from visualization.visualize import plot_data_info
from models.train_model import train_model
from models.predict_model import predict_model


def main():
    # Configuration
    RAW_DATA_PATH = "../data/raw/loan_data_2007_2014.csv"
    PROCESSED_DATA_PATH = "../data/processed/processed_loan_data.csv"
    
    # 1. Load raw data
    print("Loading data...")
    loan_data = load_data(RAW_DATA_PATH)
    
    # 2. Preprocess data
    print("Preprocessing data...")
    loan_data = preprocess_data(loan_data)
    
    # 3. Save processed data
    print("Saving processed data...")
    save_data(loan_data, PROCESSED_DATA_PATH)
    
    # 4. Data visualization
    print("Generating visualizations...")
    plot_data_info(loan_data)
    
    # 5. Train-test split
    X_train, X_test, y_train, y_test = split_data(loan_data)
    
    # 6. Model training
    # model = train_model(X_train, y_train)
    
    # 7. Model prediction
    # predictions = predict_model(model, X_test)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()