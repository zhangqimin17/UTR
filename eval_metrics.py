import numpy as np
import pandas as pd


def mae(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    abs_err = abs(true - pred)
    abs_err = abs_err[~(abs_err == 'nan')]
    abs_err = abs_err[~np.isnan(abs_err)]
    return np.nanmean(abs_err)
