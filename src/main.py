# ./src/main.py

import pandas as pd
import numpy as np
from src.data.make_dataset import load_data, save_data
from src.features.build_features import preprocess_data, split_data
from src.visualization.visualize import plot_data_info
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

    # 6. Calculating weight of evidence of the variable grade
    df_temp = pd.concat([X_train['grade'], y_train], axis=1)
    df_temp1 = df_temp.groupby("grade").agg(
        n_obs=pd.NamedAgg(column="good_bad", aggfunc="count"),
        prop_good_bad=pd.NamedAgg(column="good_bad", aggfunc="mean"),
        n_good=pd.NamedAgg(column="good_bad", aggfunc="sum")
    ).reset_index()
    df_temp1["n_bad"] = df_temp1["n_obs"] - df_temp1["n_good"]
    df_temp1["prop_good"] = df_temp1["n_good"] / sum(df_temp1["n_good"])
    df_temp1["prop_bad"] = df_temp1["n_bad"] / sum(df_temp1["n_bad"])
    # df_temp.good_bad.value_counts()

    # Calculating Weight of Evidence
    df_temp1["WoE"] = np.log(df_temp1["prop_good"] / df_temp1["prop_bad"])
    df_temp1.sort_values("WoE", ascending=True, inplace=True)

    # Calculating Information Value (IV)
    df_temp1["IV"] = (df_temp1["prop_good"] - df_temp1["prop_bad"]) * df_temp1["WoE"]
    df_temp1["IV"] = df_temp1["IV"].sum()
    # Information Value of the variable is <3, hence the predictive power of the variable is weak


    # 6. Model training
    # model = train_model(X_train, y_train)

    # 7. Model prediction
    # predictions = predict_model(model, X_test)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
