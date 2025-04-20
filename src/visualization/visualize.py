import pandas as pd
from .plot_settings import set_plot_style

def plot_data_info(loan_data):
    """Plot basic information about the data"""
    set_plot_style()
    
    # TODO: Implement visualization functions
    print("Dimensions of the dataframe:", loan_data.shape)
    print("Listing of the columns:\n", loan_data.columns.values)
    print("Variables with string datatypes", 
          loan_data.dtypes[loan_data.dtypes == 'object'])
    
    # TODO: Add more visualizations
    # TODO 6: Visualize WoE of the variable "grade"
    # TODO 7: Automate visualization of variables