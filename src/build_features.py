import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(loan_data):
    """Preprocess the loan data"""
    # Employee term length processing
    loan_data['emp_length_int'] = (loan_data['emp_length']
                                   .str.replace('+ years', '')
                                   .str.replace(' years', '')
                                   .str.replace(' year', '')
                                   .str.replace('< ', '')
                                   .str.strip()
                                   .astype('Int64'))
    
    # Term processing
    loan_data['term_int'] = loan_data['term'].str.replace(' months', '').str.strip().astype('int64')
    
    # Date processing
    loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format="%b-%y")
    loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format="%b-%y")
    
    # Offset incorrect dates
    loan_data.loc[loan_data['earliest_cr_line_date'] > '2017-12-01', ['earliest_cr_line_date']] = (
        loan_data.loc[loan_data['earliest_cr_line_date'] > '2017-12-01', ['earliest_cr_line_date']] - 
        pd.DateOffset(years=100))
    
    # Calculate months since
    loan_data["mths_since_earliest_cr_line"] = round(
        pd.to_numeric((pd.Timestamp("2017-12-01") - loan_data["earliest_cr_line_date"]) / np.timedelta64(1,'M')))
    loan_data["mths_since_issue_d"] = round(
        pd.to_numeric((pd.to_datetime("2017-12-01") - loan_data["issue_d_date"]) / np.timedelta64(1,"M")))
    
    # Create dummy variables
    loan_data_dummies = [
        pd.get_dummies(loan_data["grade"], prefix="Grade", prefix_sep=":"),
        pd.get_dummies(loan_data['sub_grade'], prefix="Sub_Grade", prefix_sep=":"),
        pd.get_dummies(loan_data['home_ownership'], prefix="Home_Ownership", prefix_sep=":"),
        pd.get_dummies(loan_data['verification_status'], prefix="Verification_Status", prefix_sep=":"),
        pd.get_dummies(loan_data['loan_status'], prefix="Loan_Status", prefix_sep=":"),
        pd.get_dummies(loan_data['purpose'], prefix="Purpose", prefix_sep=":"),
        pd.get_dummies(loan_data['addr_state'], prefix="Addr_State", prefix_sep=":"),
        pd.get_dummies(loan_data['initial_list_status'], prefix="Initial_List_Status", prefix_sep=":")
    ]
    
    loan_data_dummies = pd.concat(loan_data_dummies, axis=1)
    loan_data = pd.concat([loan_data, loan_data_dummies], axis=1)
    
    # Handle missing values
    loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)
    loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)
    
    for col in ['mths_since_earliest_cr_line', 'acc_now_delinq', 'total_acc', 
                'pub_rec', 'open_acc', 'inq_last_6mths', 'delinq_2yrs', 'emp_length_int']:
        loan_data[col].fillna(0, inplace=True)
    
    # Create target variable
    loan_data['loan_status'] = loan_data['loan_status'].str.strip()
    loan_data['good_bad'] = np.where(
        loan_data['loan_status'].isin([
            'Fully Paid', 'Current', 'In Grace Period', 
            'Late (16-30 days)', 'Does not meet the credit policy. Status:Fully Paid'
        ]), 1, 0)
    
    return loan_data


# Train test Split
def split_data(loan_data, test_size=0.2, random_state=42, target_col='good_bad'):
    """
    Split data into train and test sets
    
    Parameters:
    - loan_data: DataFrame containing the processed data
    - test_size: Proportion of data for test set (default 0.2)
    - random_state: Random seed for reproducibility (default 42)
    - target_col: Name of target variable column (default 'good_bad')
    
    Returns:
    - X_train, X_test, y_train, y_test: Split features and targets
    """
    X = loan_data.drop(columns=[target_col])
    y = loan_data[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# TODO: Calculate WoE (Weight of Evidence) and IV (Information Value) of discrete variables
def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    df_temp = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)
    df_temp1 = df_temp.groupby(discrete_variable_name).agg(
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
    df_temp1 = df_temp1.sort_values("WoE", ascending=True).reset_index(drop=True)
    df_temp1["diff_WoE"] = df_temp1["WoE"].diff().abs()

    # Calculating Information Value (IV)
    df_temp1["IV"] = (df_temp1["prop_good"] - df_temp1["prop_bad"]) * df_temp1["WoE"]
    df_temp1["IV"] = df_temp1["IV"].sum()

    return df_temp1

    
# TODO: Calculate WoE (Weight of Evidence) and IV (Information Value) of continuous variables
