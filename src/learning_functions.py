from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression,SGDRegressor,Ridge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,make_scorer

from sklearn.linear_model import SGDRegressor
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.special import expit as logistic
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

# def logistic_mse_loss(y,y_pre):
#   return mean_squared_error(logistic(y_pre),y)

def get_optimal_cv_strategy(y, verbose=True):
    """
    Automatically choose the best CV strategy for small datasets
    """
    n_samples = len(y)
    min_class_size = min(np.bincount(y))
    
    if verbose:
        print(f"Dataset size: {n_samples}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"Minimum class size: {min_class_size}")
    
    # Decision logic for small datasets
    if min_class_size < 3:
        if verbose:
            print("→ Using LeaveOneOut (min class size < 3)")
        return LeaveOneOut()
    elif n_samples < 15:
        if verbose:
            print("→ Using LeaveOneOut (very small dataset)")
        return LeaveOneOut()
    elif min_class_size >= 3:
        if verbose:
            print("→ Using StratifiedKFold(n_splits=3)")
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    else:
        if verbose:
            print("→ Using StratifiedKFold(n_splits=5)")
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



def safe_all(x):
    """Check if all elements are True, handling both NumPy arrays and scalars"""
    if isinstance(x, (bool, int, float)):  # If it's a scalar
        return bool(x)
    elif hasattr(x, 'all'):  # If it's a NumPy array or similar
        return x.all()
    elif hasattr(x, '__iter__'):  # If it's another iterable
        return all(x)
    else:
        return bool(x)  # Default case, convert to bool


def expected_time_reward_diff(reward_diff):

  return np.where(reward_diff != 0, np.tanh(reward_diff) / reward_diff, 1.0)


# # Define a custom loss function (e.g., mean absolute error)
# def mean_absolute_error(y_true, y_pred):
#     return np.mean(np.abs(y_true - y_pred))

# # Initialize the custom regressor
# sgd_regressor = CustomSGDRegressor(custom_loss=mean_absolute_error, alpha=0.0001, max_iter=1000, tol=1e-3)

# # Fit the model
# # X_train and y_train are your training data
# sgd_regressor.fit(X_train, y_train)

def logistic_regression_preferences(X_diff, pref, time, true_w, init_w = None, verbose = False):

    """
    Robust logistic regression that handles small datasets and single-class problems
    """
    
    # Check if we have at least 2 classes
    unique_classes = np.unique(pref)
    
    if len(unique_classes) < 2:
        print(f"Warning: Only one class found in preferences: {unique_classes}")
        
        # For preference-based bandits, create synthetic comparison using mean
        n_samples, n_features = X_diff.shape
        mean_diff = X_diff.mean(axis=0).reshape(1, -1)  # Mean of observed differences
        
        if unique_classes[0] == 1:  # Only positive preferences observed
            # Add synthetic negative preference at mean difference
            X_synth = np.vstack([X_diff, mean_diff])
            pref_synth = np.concatenate([np.ones(n_samples), np.array([0])])
        else:  # Only negative preferences (pref == -1)
            # Add synthetic positive preference at mean difference  
            X_synth = np.vstack([X_diff, mean_diff])
            pref_synth = np.concatenate([np.zeros(n_samples), np.array([1])])
        
        
            # Use heavy regularization since we're adding synthetic data
        model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', 
                                    fit_intercept=False, max_iter=1000)
        model.fit(X_synth, pref_synth)
        
        if verbose:
            print(f"Used synthetic mean approach. Original class: {unique_classes[0]}")
            print(f"Synthetic dataset size: {len(X_synth)}")
            print(f"Mean difference vector: {mean_diff.flatten()}")
        
        return model.coef_
        
    # Check if dataset is too small for GridSearchCV
    if len(X_diff) < 5:  # Need at least 5 samples for 2-fold CV
        print(f"Warning: Dataset too small for CV ({len(X_diff)} samples). Using simple LogisticRegression.")
        y_train = np.where(pref == -1, 0, pref)
        
        model = LogisticRegression(C=1, penalty='l2', solver='lbfgs', 
                                        fit_intercept=False, max_iter=1000)
        model.fit(X_train, y_train)
        
        if verbose:
            print("Weights (coefficients):", model.coef_)
            print("True weights", true_w)
            print("Used simple LogisticRegression (no CV)")
        
        return model.coef_
        
    
    pref = np.where(pref == -1, 0, pref) ###converting to 0/1 for logistic case.

    ###X_train, X_test, y_train, y_test = train_test_split(X_diff, pref, test_size=0.3, random_state=42)
    ###no test/train split
    X_train = X_diff
    y_train = pref

    if safe_all(init_w != None):

        model = LogisticRegression(max_iter=100, random_state=42,fit_intercept=False)
        model.coef_ = init_w
    else:
        model = LogisticRegression(max_iter=100, random_state=42,  fit_intercept=False)
    param_grid = {
        'C': [0.1, 1],  # Range of C values
        'penalty': ['l2'],
        'solver': ['lbfgs'], ##, '', 'liblinear', 'sag', 'saga'],
        'class_weight': [None] ##, 'balanced']
    }

    cv_strategy = get_optimal_cv_strategy(pref, verbose=verbose)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv= cv_strategy, scoring='accuracy', verbose=0, n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    if verbose == True:

      print("Weights (coefficients):", best_model.coef_)

      print ("True weights", true_w)

      # Print bias (intercept)
      print("Bias (intercept):", best_model.intercept_)

      # Observe the best parameters and score
      print("Best Parameters:", grid_search.best_params_)
      print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    results = grid_search.cv_results_

    return best_model.coef_

class BoundedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lower_bound=1e-6,upper_bound = 1, **mlp_kwargs):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.model = MLPRegressor(**mlp_kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return np.minimum(np.maximum(self.model.predict(X), self.lower_bound), self.upper_bound)


def learning_time_nn(X_diff, pref, time, true_w, verbose = False, a = 1, hidden_layer_no = 64, theta_bound = 10): ###

   
    n_samples = X_diff.shape[0]

    # Define the model
    time_model = BoundedRegressor(
        lower_bound=np.tanh(theta_bound) / theta_bound,
        upper_bound=1,
        hidden_layer_sizes=(hidden_layer_no, hidden_layer_no // 2),
        activation='relu',
        max_iter=5000,
        solver='adam',
        random_state=42
    )

    if n_samples < 5:
        # Train on full data directly
        if verbose:
            print(f"Small dataset ({n_samples} samples) — training on full data.")
        time_model.fit(X_diff, time)
        return time_model

    # Otherwise, split and train normally
    X_train, X_test, time_train, time_test = train_test_split(
        X_diff, time, test_size=0.20, shuffle=True, random_state=42
    )

    time_model.fit(X_train, time_train)

    if verbose:
        time_test_pred = time_model.predict(X_test)
        print("Mean Squared Error (MSE) on Testing Set:", mean_squared_error(time_test, time_test_pred))

    return time_model


def joint_pref_time_learning_estimates_regression(X_diff, pref, time, true_w, custom_r_predictor, estimated_time_func, initial_w = None, verbose = False, unweighted_loss = False):

    ####this notion of DML directly calls the regressor it also estimates using the predictors passed to it

    target_reward = custom_r_predictor(X =X_diff, y= pref, t = time)

    # Check if dataset is too small for GridSearchCV
    if len(X_diff) < 6:  # Need at least 3 samples per fold for meaningful CV
        print(f"Warning: Dataset too small for CV ({len(X_diff)} samples). Using simple Ridge regression.")
        
        # Use simple Ridge with default alpha
        model = Ridge(alpha=0.1, fit_intercept=False)  # Conservative regularization
        
        if unweighted_loss:
            model.fit(X_diff, target_reward)
        else:
            model.fit(X_diff, target_reward, sample_weight=estimated_time_func(X_diff).reshape(-1))
        
        if verbose:
            print("Weights (coefficients):", model.coef_)
            print("True weights", true_w)
            print("Bias (intercept):", model.intercept_)
            print("Used simple Ridge (no CV)")
        
        return model.coef_

    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    param_grid = {
        'alpha': [0.01, 0.1],  # Regularization strength
        'solver': ['auto']  # Solver for optimization
    }



    model = Ridge(fit_intercept=False)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=2,
        verbose=0,
        error_score="raise"
      )

    # Fit the model with sample weights

    ##print ("Shape", estimated_time_func(X_diff).shape)

    if unweighted_loss: 
        grid_search.fit(X_diff, target_reward) 
    else:
        grid_search.fit(X_diff, target_reward, sample_weight = estimated_time_func(X_diff).reshape(-1))

    best_model = grid_search.best_estimator_


    if verbose == True:

      print("Best Parameters:", grid_search.best_params_)
      print("Best Cross-Validation Accuracy:", grid_search.best_score_)


      print("Weights (coefficients):", best_model.coef_)

      print ("True weights", true_w)

    # Print bias (intercept)
      print("Bias (intercept):", model.intercept_)

    return best_model.coef_


def joint_pref_time_learning_regression(X_diff, pref, time, true_w, a_val = 1.0, initial_w = None, no_weights = True, verbose = False, k_fold = 4):

    ###This part splits the data and then estimates on one subset and uses this estimate on the other subset for DML. Uses a K fold method for averaging to compute final estimate.


    X_diff_split = np.array_split(X_diff, k_fold, axis=0)
    pref_split = np.array_split(pref, k_fold, axis=0)
    time_split = np.array_split(time, k_fold, axis=0)

    running_params_ortho = 0###used to average
    running_params_non_ortho = 0
    for itr in range(k_fold):

        X_diff_first = X_diff_split[itr]
        pref_first = pref_split[itr]
        time_first = time_split[itr]

        X_diff_second = np.concatenate([X_diff_split[i] for i in range(k_fold) if i!= itr])
        pref_second = np.concatenate([pref_split[i] for i in range(k_fold) if i!= itr])
        time_second = np.concatenate([time_split[i] for i in range(k_fold) if i!= itr])

        weights_logistic_pref_regression_first = logistic_regression_preferences(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w)[0]/2
        ####weights_time_regression_first = regression_time(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w) --- this will not work clearly even from theory



        def estimated_time_func(X_di): ###assumes knowledge of weights_time_regression
          ##return expected_time_reward_diff(np.dot(X_di,weights_time_regression_first))
          
            reward_diff = np.dot(X_di,weights_logistic_pref_regression_first)
            return np.where(reward_diff != 0, (a_val**2)*np.tanh(reward_diff) / reward_diff, a_val**2)
        
        # def estimated_time_func(X_di): ###assumes knowledge of weights_time_regression

        # def estimated_pref_func(X_di):
        #     return 2*logistic(np.dot(X_di, weights_logistic_pref_regression_first)) - 1



        def custom_r_predictor(X,t,y): ##for DML estimator directly fitting to weighted regression

            return (a_val**2)*y/estimated_time_func(X) - (t - estimated_time_func(X))/(estimated_time_func(X)) * np.dot(X, weights_logistic_pref_regression_first)

        weights_joint_learning_dml_regressor = joint_pref_time_learning_estimates_regression(X_diff_second, pref_second, time_second, true_w, custom_r_predictor = custom_r_predictor, estimated_time_func = estimated_time_func,unweighted_loss= no_weights)


        running_params_ortho += weights_joint_learning_dml_regressor

        ##print ("Itr", itr, " over ")


    return running_params_ortho/k_fold


def joint_pref_time_learning_time_separate(X_diff, pref, theta_bound, time, true_w = None, a_val = 1.0, no_weights = True, initial_w = None, verbose = False, k_fold = 4):

    # PATCH: Handle small datasets
    n_samples = len(X_diff)

    
    
    # If dataset is too small, use simple approach without cross-validation
    if n_samples < 3:  # Need at least 3 samples for meaningful 2-fold CV
        if verbose:
            print(f"Warning: Dataset too small ({n_samples} samples). Using simple estimation without CV.")
        
        return np.zeros(X_diff.shape[1]), np.zeros(X_diff.shape[1])  # Return zeros for both estimates
    
    # PATCH: Adjust k_fold for small datasets
    max_k_fold = min(k_fold, n_samples // 4)  # Ensure each fold has at least 2-3 samples
    if max_k_fold < 2:
        max_k_fold = 2
    
    if max_k_fold != k_fold and verbose:
        print(f"Adjusting k_fold from {k_fold} to {max_k_fold} due to small dataset size.")
    
    k_fold = max_k_fold

    print (" number of samples", n_samples, "k_fold", k_fold)

    ###This part splits the data and then estimates on one subset and uses this estimate on the other subset for DML. Uses a K fold method for averaging to compute final estimate.

    ###This part splits the data and then estimates on one subset and uses this estimate on the other subset for DML. Uses a K fold method for averaging to compute final estimate.


    X_diff_split = np.array_split(X_diff, k_fold, axis=0)
    pref_split = np.array_split(pref, k_fold, axis=0)
    time_split = np.array_split(time, k_fold, axis=0)

    running_params_ortho = 0###used to average
    running_params_non_ortho = 0
    sucess_count = 0
    for itr in range(k_fold):

        X_diff_first = X_diff_split[itr]
        pref_first = pref_split[itr]
        time_first = time_split[itr]

        X_diff_second = np.concatenate([X_diff_split[i] for i in range(k_fold) if i!= itr])
        pref_second = np.concatenate([pref_split[i] for i in range(k_fold) if i!= itr])
        time_second = np.concatenate([time_split[i] for i in range(k_fold) if i!= itr])

        try:
            weights_logistic_pref_regression_first = logistic_regression_preferences(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w)[0]/2 ### note we always estimate a*r(X)
        ####weights_time_regression_first = regression_time(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w) --- this will not work clearly even from theory
            sucess_count += 1
        except:
            continue

        time_model = learning_time_nn(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w, theta_bound = theta_bound, verbose= False)


        def estimated_time_func(X_di): ###assumes knowledge of weights_time_regression

          # time_model.eval()
          # with torch.no_grad():

          #   return np.minimum(np.maximum(time_model(torch.from_numpy(X_di.astype(np.float32))).detach().numpy(), np.tanh(theta_bound)/theta_bound),1)
          return time_model.predict(X_di)

        # def estimated_pref_func(X_di):
        #     return 2*logistic(np.dot(X_di, weights_logistic_pref_regression_first)) - 1


        

        def custom_r_predictor(X,t,y): ##for DML estimator directly fitting to weighted regression

            # if no_weights: ###this is becuase the nuisance function is different for two cases
            return (a_val**2)*y/estimated_time_func(X) - (t - estimated_time_func(X))/(estimated_time_func(X)) * np.dot(X, weights_logistic_pref_regression_first)
            # else:
            #     return y/estimated_time_func(X) - (t - estimated_time_func(X))/(estimated_time_func(X)**2) * estimated_pref_func(X)

        def custom_r_non_ortho_predictor(X,t,y): ##for DML estimator without orthogonalization directly fitting to weighted regression
            return (a_val**2)*y/estimated_time_func(X)

        weights_joint_learning_dml_regressor = joint_pref_time_learning_estimates_regression(X_diff_second, pref_second, time_second, true_w, custom_r_predictor = custom_r_predictor, estimated_time_func = estimated_time_func, unweighted_loss= no_weights)
        weights_joint_learning_dml_non_ortho_regressor = joint_pref_time_learning_estimates_regression(X_diff_second, pref_second, time_second, true_w, custom_r_predictor = custom_r_non_ortho_predictor, estimated_time_func = estimated_time_func, unweighted_loss= no_weights)

        running_params_ortho += weights_joint_learning_dml_regressor
        running_params_non_ortho += weights_joint_learning_dml_non_ortho_regressor

        ##print ("Itr", itr, " over ")
    if sucess_count > 0:
        return running_params_ortho/sucess_count, running_params_non_ortho/sucess_count

    else:
        print("Warning: No successful iterations in joint_pref_time_learning_time_separate. computing with zero weights from first stage")
        time_model = learning_time_nn(X_diff=X_diff,pref=pref,time=time, true_w = true_w, theta_bound = theta_bound, verbose= False)

        def estimated_time_func(X_di): ###assumes knowledge of weights_time_regression

            return time_model.predict(X_di)
        def custom_r_non_ortho_predictor(X,t,y): ##for DML estimator without orthogonalization directly fitting to weighted regression
            return (a_val**2)*y/estimated_time_func(X)
        
        weights_joint_learning_dml_non_ortho_regressor = joint_pref_time_learning_estimates_regression(X_diff, pref, time, true_w, custom_r_predictor = custom_r_non_ortho_predictor, estimated_time_func = estimated_time_func, unweighted_loss= no_weights)

        return weights_joint_learning_dml_non_ortho_regressor, weights_joint_learning_dml_non_ortho_regressor
    

def joint_pref_time_learning_time_separate_reuse(X_diff, pref, theta_bound, time, true_w = None, a_val = 1.0, initial_w = None, no_weights = True):

    try:
        weights_logistic_pref_regression_first = logistic_regression_preferences(X_diff=X_diff,pref=pref,time=time, true_w = true_w)[0]/2 ### note we always estimate a*r(X)
        ####weights_time_regression_first = regression_time(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w) --- this will not work clearly even from theory
    except:
        weights_logistic_pref_regression_first = np.zeros(X_diff.shape[1])  # Fallback if logistic regression fails

    time_model = learning_time_nn(X_diff=X_diff,pref=pref,time=time, true_w = true_w, theta_bound = theta_bound, verbose= False)


    def estimated_time_func(X_di): ###assumes knowledge of weights_time_regression

        return time_model.predict(X_di)

    def custom_r_predictor(X,t,y): ##for DML estimator directly fitting to weighted regression

            # if no_weights: ###this is becuase the nuisance function is different for two cases
        return (a_val**2)*y/estimated_time_func(X) - (t - estimated_time_func(X))/(estimated_time_func(X)) * np.dot(X, weights_logistic_pref_regression_first)
            # else:
            #     return y/estimated_time_func(X) - (t - estimated_time_func(X))/(estimated_time_func(X)**2) * estimated_pref_func(X)

    def custom_r_non_ortho_predictor(X,t,y): ##for DML estimator without orthogonalization directly fitting to weighted regression
        return (a_val**2)*y/estimated_time_func(X)

    weights_joint_learning_dml_regressor = joint_pref_time_learning_estimates_regression(X_diff, pref, time, true_w, custom_r_predictor = custom_r_predictor, estimated_time_func = estimated_time_func, unweighted_loss= no_weights)
    weights_joint_learning_dml_non_ortho_regressor = joint_pref_time_learning_estimates_regression(X_diff, pref, time, true_w, custom_r_predictor = custom_r_non_ortho_predictor, estimated_time_func = estimated_time_func, unweighted_loss= no_weights)


    return weights_joint_learning_dml_regressor, weights_joint_learning_dml_non_ortho_regressor



def joint_pref_time_learning_regression_y_nuisance(X_diff, pref, time, true_w, a_val = 1.0, initial_w = None, no_weights = True, verbose = False, k_fold = 4):

    ###This part splits the data and then estimates on one subset and uses this estimate on the other subset for DML. Uses a K fold method for averaging to compute final estimate.


    X_diff_split = np.array_split(X_diff, k_fold, axis=0)
    pref_split = np.array_split(pref, k_fold, axis=0)
    time_split = np.array_split(time, k_fold, axis=0)

    running_params_ortho = 0###used to average
    running_params_non_ortho = 0
    for itr in range(k_fold):

        X_diff_first = X_diff_split[itr]
        pref_first = pref_split[itr]
        time_first = time_split[itr]

        X_diff_second = np.concatenate([X_diff_split[i] for i in range(k_fold) if i!= itr])
        pref_second = np.concatenate([pref_split[i] for i in range(k_fold) if i!= itr])
        time_second = np.concatenate([time_split[i] for i in range(k_fold) if i!= itr])

        weights_logistic_pref_regression_first = logistic_regression_preferences(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w)[0]/2 ###learns a*r(X)
        ####weights_time_regression_first = regression_time(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w) --- this will not work clearly even from theory



        def estimated_time_func(X_di): ###assumes knowledge of weights_time_regression
          ##return expected_time_reward_diff(np.dot(X_di,weights_time_regression_first))
          
            reward_diff = np.dot(X_di,weights_logistic_pref_regression_first)
            return np.where(reward_diff != 0, (a_val**2)*np.tanh(reward_diff) / reward_diff, a_val**2)
        
        # def estimated_time_func(X_di): ###assumes knowledge of weights_time_regression

        # def estimated_pref_func(X_di):
        #     return 2*logistic(np.dot(X_di, weights_logistic_pref_regression_first)) - 1



        def custom_r_predictor(X,t,y): ##for DML estimator directly fitting to weighted regression

            return y/estimated_time_func(X) - (t - estimated_time_func(X))/((estimated_time_func(X))**2) * (2*logistic(2*np.dot(X, weights_logistic_pref_regression_first))-1)##now estimates r(X)/a

        weights_joint_learning_dml_regressor = joint_pref_time_learning_estimates_regression(X_diff_second, pref_second, time_second, true_w, custom_r_predictor = custom_r_predictor, estimated_time_func = estimated_time_func,unweighted_loss= no_weights)


        running_params_ortho += weights_joint_learning_dml_regressor

        ##print ("Itr", itr, " over ")


    return running_params_ortho/k_fold


def joint_pref_time_learning_time_separate_y_nuisance(X_diff, pref, theta_bound, time, true_w, a_val = 1.0, no_weights = True, initial_w = None, verbose = False, k_fold = 4):

    ###This part splits the data and then estimates on one subset and uses this estimate on the other subset for DML. Uses a K fold method for averaging to compute final estimate.


    X_diff_split = np.array_split(X_diff, k_fold, axis=0)
    pref_split = np.array_split(pref, k_fold, axis=0)
    time_split = np.array_split(time, k_fold, axis=0)

    running_params_ortho = 0###used to average
    running_params_non_ortho = 0
    for itr in range(k_fold):

        X_diff_first = X_diff_split[itr]
        pref_first = pref_split[itr]
        time_first = time_split[itr]

        X_diff_second = np.concatenate([X_diff_split[i] for i in range(k_fold) if i!= itr])
        pref_second = np.concatenate([pref_split[i] for i in range(k_fold) if i!= itr])
        time_second = np.concatenate([time_split[i] for i in range(k_fold) if i!= itr])

        weights_logistic_pref_regression_first = logistic_regression_preferences(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w)[0]/2 ### note we always estimate a*r(X)
        ####weights_time_regression_first = regression_time(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w) --- this will not work clearly even from theory

        time_model = learning_time_nn(X_diff=X_diff_first,pref=pref_first,time=time_first, true_w = true_w, theta_bound = theta_bound, verbose= False)


        def estimated_time_func(X_di): ###assumes knowledge of weights_time_regression

          # time_model.eval()
          # with torch.no_grad():

          #   return np.minimum(np.maximum(time_model(torch.from_numpy(X_di.astype(np.float32))).detach().numpy(), np.tanh(theta_bound)/theta_bound),1)
          return time_model.predict(X_di)

        # def estimated_pref_func(X_di):
        #     return 2*logistic(np.dot(X_di, weights_logistic_pref_regression_first)) - 1


        

        def custom_r_predictor(X,t,y): ##for DML estimator directly fitting to weighted regression

            # if no_weights: ###this is becuase the nuisance function is different for two cases
            return y/estimated_time_func(X) - (t - estimated_time_func(X))/((estimated_time_func(X))**2) * (2*logistic(2*np.dot(X, weights_logistic_pref_regression_first))-1) ###now estimates r(X)/a
            # else:
            #     return y/estimated_time_func(X) - (t - estimated_time_func(X))/(estimated_time_func(X)**2) * estimated_pref_func(X)

        def custom_r_non_ortho_predictor(X,t,y): ##for DML estimator without orthogonalization directly fitting to weighted regression
            return y/estimated_time_func(X)

        weights_joint_learning_dml_regressor = joint_pref_time_learning_estimates_regression(X_diff_second, pref_second, time_second, true_w, custom_r_predictor = custom_r_predictor, estimated_time_func = estimated_time_func, unweighted_loss= no_weights)
        weights_joint_learning_dml_non_ortho_regressor = joint_pref_time_learning_estimates_regression(X_diff_second, pref_second, time_second, true_w, custom_r_predictor = custom_r_non_ortho_predictor, estimated_time_func = estimated_time_func, unweighted_loss= no_weights)

        running_params_ortho += weights_joint_learning_dml_regressor
        running_params_non_ortho += weights_joint_learning_dml_non_ortho_regressor

        ##print ("Itr", itr, " over ")


    return running_params_ortho/k_fold, running_params_non_ortho/k_fold






