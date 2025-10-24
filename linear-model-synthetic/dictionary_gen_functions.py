import os
import sys
import time
import numpy as np
from scipy.special import expit
from read_file import numpy_data_str_pref_data
from learning_functions import (
    joint_pref_time_learning_time_separate,
    joint_pref_time_learning_time_separate_cross_validation,
    joint_pref_time_learning_time_separate_reuse,
    joint_pref_time_learning_time_separate_y_nuisance,
    logistic_regression_preferences
)
import pickle

# Redirect output to log file



def compute_metrics(weights_dict, X_diff_test, pref_test, true_w, a_val):
    results = {}
    for name, w in weights_dict.items():
        dist = np.linalg.norm(w/a_val - true_w, ord=2)
        logits = expit(2 * np.dot(w, X_diff_test.T))
        preds = np.where(logits >= 0.5, 1, -1)
        acc = np.mean(preds == pref_test)
        results[name] = {"distance": dist, "accuracy": acc}
    return results

def compute_metrics_y_nuisance(weights_dict, X_diff_test, pref_test, true_w, a_val):
    results = {}
    for name, w in weights_dict.items():
        dist = np.linalg.norm(w*a_val - true_w, ord=2) ##y nuisance we estimate r(X)/a
        logits = expit(2 * np.dot(w, X_diff_test.T))
        preds = np.where(logits >= 0.5, 1, -1)
        acc = np.mean(preds == pref_test)
        results[name] = {"distance": dist, "accuracy": acc}
    return results


def compare_reward_time_dml_no_time(
    theta_bound=7,
    csv_index=8,
    dimension=20,
    data_points=1000,
    test_data_points=2000,
    a_val=1.0,
    k_fold=4,
    a_dist="nodist",
    a_dist_k=None,
    random_state=None
):
    X_diff, pref, time, true_w = numpy_data_str_pref_data(
        theta_bound=theta_bound,
        csv_index=csv_index,
        dimension=dimension,
        a_dist=a_dist,
        a_dist_k=a_dist_k,
        a_val=a_val
    )

    if data_points + test_data_points > X_diff.shape[0]:
        raise ValueError("Test and train split do not add up")

    X_diff_test = X_diff[-test_data_points:]
    pref_test = pref[-test_data_points:]

    X_diff = X_diff[:data_points]
    pref = pref[:data_points]
    time = time[:data_points]

    
    
    w_logistic = (
        logistic_regression_preferences(
            X_diff=X_diff,
            pref=pref,
            time=time,
            true_w=true_w,
            random_state=random_state
        )[0] / 2
    )
    w_time_sep, w_nonortho_sep = joint_pref_time_learning_time_separate( ##orefers to orthogonal loss 1 and time estimated by a 3 layered neural network
        X_diff, pref,
        theta_bound=theta_bound,
        time=time,
        true_w=true_w,
        k_fold=k_fold,
        no_weights=False,
        a_val=a_val,
        random_state=random_state
    )

    w_time_sep_cross_val, w_nonortho_sep_cross_val = joint_pref_time_learning_time_separate_cross_validation(
        X_diff, pref,
        theta_bound=theta_bound,
        time=time,
        true_w=true_w,
        k_fold=k_fold,
        no_weights=False,
        a_val=a_val,
        random_state=random_state
    )

    w_time_sep_reuse, w_nonortho_sep_reuse = joint_pref_time_learning_time_separate_reuse(
        X_diff, pref,
        theta_bound=theta_bound,
        time=time,
        true_w=true_w,
        k_fold=k_fold,
        no_weights=False,
        a_val=a_val,
        random_state=random_state
    )


    
    weights = {
        "logistic": w_logistic,
        "dml-time-separate": w_time_sep,
        "dml-non-ortho-time-separate": w_nonortho_sep,
        "dml-time-separate-cross-val": w_time_sep_cross_val,
        "dml-non-ortho-time-separate-cross-val": w_nonortho_sep_cross_val,
        "dml-time-separate-reuse": w_time_sep_reuse,
        "dml-non-ortho-time-separate-reuse": w_nonortho_sep_reuse,
    }

    return compute_metrics(weights, X_diff_test, pref_test, true_w, a_val)


def compare_reward_time_dml_no_time_y_nuisance( ##refers to orthogonal loss 2
    theta_bound=7,
    csv_index=8,
    dimension=20,
    data_points=1000,
    test_data_points=2000,
    a_val=1.0,
    k_fold=4,
    a_dist="nodist",
    a_dist_k=None,
    random_state=None
):
    X_diff, pref, time, true_w = numpy_data_str_pref_data(
        theta_bound=theta_bound,
        csv_index=csv_index,
        dimension=dimension,
        a_dist=a_dist,
        a_dist_k=a_dist_k,
        a_val=a_val
    )

    if data_points + test_data_points > X_diff.shape[0]:
        raise ValueError("Test and train split do not add up")

    X_diff_test = X_diff[-test_data_points:]
    pref_test = pref[-test_data_points:]

    X_diff = X_diff[:data_points]
    pref = pref[:data_points]
    time = time[:data_points]

    
    w_logistic = (
        logistic_regression_preferences(
            X_diff=X_diff,
            pref=pref,
            time=time,
            true_w=true_w,
            random_state=random_state
        )[0] / 2
    )
    w_time_sep, w_nonortho_sep = joint_pref_time_learning_time_separate_y_nuisance( ##refers to orthogonal loss 2
        X_diff, pref,
        theta_bound=theta_bound,
        time=time,
        true_w=true_w,
        k_fold=k_fold,
        no_weights=False,
        a_val=a_val,
        random_state=random_state
    )
    

    weights = {
        "logistic": w_logistic/(a_val**2), ###as goal to estimate r(X)/a
        "dml-time-separate": w_time_sep,
        "dml-non-ortho-time-separate": w_nonortho_sep,
        
    }

    return compute_metrics_y_nuisance(weights, X_diff_test, pref_test, true_w, a_val)



