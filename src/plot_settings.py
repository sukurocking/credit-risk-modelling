import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

colors = cycler(color=plt.get_cmap("tab10").colors)  # ["b", "r", "g"]
# colors = cycler(color=["#282782", "r", "g"])

mpl.style.use("ggplot")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "lightgray"
mpl.rcParams["axes.prop_cycle"] = colors
mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.titlesize"] = 25
mpl.rcParams["figure.dpi"] = 100


import matplotlib.pyplot as plt

def set_plot_style():
    """Set consistent plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12