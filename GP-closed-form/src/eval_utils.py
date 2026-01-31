import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def eval_Rsquared(
    Y_pred,
    Y_truth,
    take_log:bool=False,
):
    if take_log:
        # was not in log scale before, so we put in log scale
        Y_pred_log = np.log(np.abs(Y_pred)) # took absolute value to prevent a few negative values from getting nan
        Y_truth_log = np.log(Y_truth)
        numerator = np.sum((Y_pred_log - Y_truth_log)**2)
        mean = np.mean(Y_truth_log)
        denominator = np.sum((Y_pred_log - mean)**2)
        return 1.0 - numerator / denominator

    else: # is in log scale prob so we don't change it
        numerator = np.sum((Y_pred - Y_truth)**2)
        mean = np.mean(Y_truth)
        denominator = np.sum((Y_pred - mean)**2)
        return 1.0 - numerator / denominator
    
def eval_rmse(
    Y_pred,
    Y_truth,
    take_log:bool=False,
): 
    if take_log:
        Y_pred_log = np.log(Y_pred)
        Y_truth_log = np.log(Y_truth)
        sq_diff = (Y_pred_log - Y_truth_log)**2
    else:
        sq_diff = (Y_pred - Y_truth)**2
    return np.sqrt(np.mean(sq_diff))
