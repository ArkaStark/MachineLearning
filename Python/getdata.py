# This module contains functions to get data and create a suitable dataframe
import numpy as np
import tkinter as Tk
import pandas as pd


def gdata(loc, name=[]):
    """ This function is used to open the file containing the data and load the data."""
    f = pd.read_csv(loc, names=name)
    return f


def rem_nan(f, x=0):
    pass
    f = pd.DataFrame(f)
    return f.isna()
