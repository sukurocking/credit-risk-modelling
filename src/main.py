# ./src/main.py
import os
os.getcwd()
os.chdir('/Users/sukumarsubudhi/Downloads/Learning/credit_risk_modelling_python')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.make_dataset import load_data, save_data
from src.build_features import preprocess_data, split_data, woe_discrete
from src.visualize import plot_data_info, plot_by_woe


def main():
    # Configuration
    RAW_DATA_PATH = "./data/raw/loan_data_2007_2014.csv"
    PROCESSED_DATA_PATH = "./data/processed/processed_loan_data.csv"

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
    # print("Generating visualizations...")
    # plot_data_info(loan_data)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = split_data(loan_data)

    # 6. Calculating weight of evidence of the variable grade
    df_temp1 = woe_discrete(df=X_train, discrete_variable_name="grade", good_bad_variable_df=y_train)
    # Information Value of the variable is <3, hence the predictive power of the variable is weak

    # Visualizing weight of evidence of variable grade
    plot_by_woe(df_temp1)
    # Keeping categories of grade variable as separate categories
    # Weight of Evidence is increasing as the grade changes from G to A

    # Calculating Weight of Evidence of the variable home_ownership
    # pd.options.display.max_columns = 20
    df_temp1 = woe_discrete(df=X_train, discrete_variable_name="home_ownership", good_bad_variable_df=y_train)
    plot_by_woe(df_temp1)


    # Based on Weight of Evidence of the categories in home_ownership, we will combine the categories OTHER, NONE, RENT and ANY

    print("Columns for HomeOwnership: \n")
    print([col for col in X_train.columns.values if col.lower().startswith("home_ownership")])
    X_train["Home_Ownership:OTHER_NONE_RENT_ANY"] = (X_train["Home_Ownership:OTHER"] +
                                                     X_train["Home_Ownership:NONE"] +
                                                     X_train["Home_Ownership:RENT"] +
                                                     X_train["Home_Ownership:ANY"])
    # Observing that the field home_ownership is showing IV as inf (due to denominator as 0)
    # Lets calculate the actual IV value of the field
    X_train["home_ownership_new"] = np.where(
        X_train["home_ownership"].isin(["OTHER", "NONE", "RENT", "ANY"]), "OTHER_NONE_RENT_ANY", X_train["home_ownership"]
    )
    df_temp1 = woe_discrete(df=X_train, 
                            discrete_variable_name="home_ownership_new",
                            good_bad_variable_df=y_train)
    # Information Value (IV) of the variable home_ownership is 0.02, which means the predictive power of the variable is weak

    # 6. Model training
    # model = train_model(X_train, y_train)

    # 7. Model prediction
    # predictions = predict_model(model, X_test)

    print("Pipeline completed successfully!")




if __name__ == "__main__":
    main()
