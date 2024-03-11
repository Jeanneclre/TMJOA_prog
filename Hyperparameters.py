import numpy as np

max_depth = [None, 3,5,7,10,30]
max_iter = [10000,20000,50000]


#Glmnet = elasticNet
param_grid_glmnet_origin = {
    'alpha': [10,100,10000],
    'l1_ratio': [0.01,0.1,1],
}
param_grid_glmnet = {
    'alpha': np.linspace(0.1, 10, 5),  # More finely spaced
    'l1_ratio': np.linspace(0.01, 1, 5),
}


#LogisticRegression --> glmnet # and AUC feature selection
param_grid_lr_origin = {
    'penalty': ['l1'],         # Penalty type
    'C': [1.0, 10.0],     # Regularisation parameter
    'solver': ['liblinear'],  # resolution algorithm
    'max_iter': max_iter,  # Maximum number of iterations
    'tol': [0.00001,0.0001],  # Tolerance for stopping criteria
}

param_grid_lr = {
    'penalty': ['l1', 'l2'],  # Consider both L1 and L2 regularization
    'C': np.logspace(-4, 4, 5),  # Broader and more granular range
    'solver': ['liblinear'],  # Suitable for small datasets
    'max_iter': [10000,20000,50000],
    'tol': [1e-5, 1e-4],
}

#SVM
param_grid_svm_origin = {
    'C': [ 1.0, 10.0],
    'kernel': ['linear'],
    'degree': [1,3],
    'gamma': ['scale', 'auto'],
    'max_iter': max_iter,  # -1 means no limit.
}

param_grid_svm = {
    'C': np.logspace(-2, 2, 5),  # More granular values
    'kernel': ['linear', 'rbf'],  # Including RBF for non-linearity
    'degree': [1, 3],  # Relevant only for 'poly' kernel
    'gamma': ['scale', 'auto'],
    'max_iter': [1000, 5000],  # Reduced to prevent excessive computation
}

# Random forest
param_grid_rf_origin= {
    'n_estimators': [100, 200],  # Reduced the number of trees to try
    'max_features': ['sqrt', 'log2'],
    'max_depth': max_depth,  # Reduced max depth and included None to grow full trees
    'min_samples_split': [2,4],  # Increased the minimum number of samples to split
    'min_samples_leaf': [1, 5],   # Increased the minimum number of samples per leaf
    'bootstrap': [True, False]
}
param_grid_rf = {
    'n_estimators': [50, 100, 150],  # More options, avoiding too large models
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None,1,2, 3, 5, 7],  # More granular
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2, 4],  # More options to control leaf size
    'bootstrap': [True, False],
}


# XGBtree - XGBClassifier()
param_grid_xgb_origin = {
    'n_estimators': [100,300],
    'learning_rate': [0.01,0.1,1.0],
    'max_depth': max_depth,
    'subsample': [0.5,0.9],
}

param_grid_xgb = {
    'n_estimators': [50, 100, 150],  # Adjusted for dataset size
    'learning_rate': np.linspace(0.01, 1, 5),
    'max_depth': [3, 5, 7, 10],  # More granular
    'subsample': [0.5, 0.75, 1],
}



# LDA
param_grid_lda_origin = {
    'solver': ['svd','lsqr'],
    'shrinkage': [None],
    'n_components': [1],
    'tol': [0.00001,0.0001,0.1],
}
param_grid_lda = [
    {'solver': ['svd', 'lsqr'],
    'shrinkage': [None],  # Added 'auto' option for lsqr solver
    'n_components': [1],
    'tol': [1e-5, 1e-4, 1e-3],
},
{
    'solver': ['lsqr'],
    'shrinkage': [None, 'auto'],  # Added 'auto' option for lsqr solver
    'n_components': [1],
    'tol': [1e-5, 1e-4, 1e-3],
}
]



# Neural Network
param_grid_nnet_origin= {
    'hidden_layer_sizes': [(10, 10), (40, 40), (50, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [ 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': max_iter,
}

param_grid_nnet = {
    'hidden_layer_sizes': [(10,), (20, 20), (30, 30)],  # Simpler networks
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['adam'],  # Focusing on 'adam' for its adaptive properties
    'alpha': np.logspace(-4, 0, 5),  # More granular
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 1000, 2000],  # Adjusted range
}


# Glmboost

param_grid_glmboost_origin = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.01,0.1],
    'n_estimators': [100,200],
    'subsample': [0.5,0.9],
    'max_depth': max_depth,
    'min_samples_split': [2, 4],  # Increased the minimum number of samples to split
    'max_features' : ['sqrt', 'log2'],
}

param_grid_glmboost = {
    'loss': ['log_loss', 'exponential'],  # Consider 'deviance' for logistic regression
    'learning_rate': np.linspace(0.01, 0.1, 5),
    'n_estimators': [50, 100, 150],
    'subsample': [0.5, 0.75, 1],
    'max_depth': [1,2,3, 5, 7],
    'min_samples_split': [2, 4],
    'max_features': ['sqrt', 'log2'],
}


# HDDA (high-dimensional discriminant analysis ) - GaussianMixture

param_grid_hdda_origin = {
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'max_iter': max_iter,
    'tol': [0.00001],
    'reg_covar': [0.01],
    'init_params': ['kmeans', 'random'],
    }

param_grid_hdda = {
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'max_iter': [100, 1000, 5000],  # Adjusted range
    'tol': [1e-5],
    'reg_covar': [1e-4, 1e-2],  # More options
    'init_params': ['kmeans', 'random'],
}

