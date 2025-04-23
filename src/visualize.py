import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.plot_settings import set_plot_style

def plot_data_info(loan_data):
    """Plot basic information about the data"""
    set_plot_style()
    
    # TODO: Implement visualization functions
    print("Dimensions of the dataframe:", loan_data.shape)
    print("Listing of the columns:\n", loan_data.columns.values)
    print("Variables with string datatypes", 
          loan_data.dtypes[loan_data.dtypes == 'object'])
    
# TODO: Add more visualizations
# TODO 6: Automate visualization of WoE of variables
def plot_by_woe(df_Woe):
    x_label = df_Woe.columns.values[0]
    y_label = "Weight of Evidence"
    plt.plot(df_Woe[x_label], df_Woe["WoE"], marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Weight of Evidence of " + x_label + " column")
    plt.show()