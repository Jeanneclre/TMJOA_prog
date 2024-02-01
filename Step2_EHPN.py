'''
Author: Jeanne Claret

This file is used as a step2 for the project TMJOA_PROGNOSIS.
The goal is to combine the bests 3 models from step1 and use their prediction to train a new model.

Input: the files in out/ (prediction probabilities of the outer loop)
and out_valid/ (predictions probabilities of the inner loop)

'''
from sklearn.model_selection import KFold

from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier


from glob import glob
import argparse
import pandas as pd
import numpy as np

import shap

from Step1 import hyperparam_tuning
import Hyperparameters as hp
import modelFunctions as mf

def main(args):
    input = args.input
    output = args.output

    # Parameters
    nb_column = 10

    # Set seed for reproducibility
    seed =2024


    model =HistGradientBoostingClassifier(random_state=seed)
    param_grid =  {
    'max_iter': [100, 200, 300],  # Number of boosting iterations.
    'learning_rate': [0.01, 0.1, 0.2],  # Step size for updating the weights.
    'max_depth': [3, 5, 7],  # Maximum depth of each tree.
    'min_samples_leaf': [20, 40, 60],  # Minimum number of samples per leaf.
    'l2_regularization': [0.0, 0.1, 1.0],  # L2 regularization term on weights.
    'max_bins': [255, 200, 100]  # Maximum number of bins used for splitting.
    }


    A = pd.read_csv("TMJOAI_Long_040422_Norm.csv")
    y = A.iloc[40:, 0].values
    print('y shape:',y.shape)
    X = A.iloc[40:, 1:]
    Nfold = 10
    N = 10

    # Initialize empty matrix for features
    fea = np.nan * np.empty((X.shape[1], Nfold * N))
    fea = pd.DataFrame(fea, index=X.columns)

    seed0 = 2022
    np.random.seed(seed0)

    # Creating folds for cross-validation
    kf = KFold(n_splits=Nfold, shuffle=True, random_state=seed0)
    foldsCVT = [(train_index, test_index) for train_index, test_index in kf.split(X)]

    # Preparing file paths for reading predictions
    file0 = glob(f'{input}out_valid/*_lda2*.csv') + glob(f'{input}out_valid/*_glmboost*.csv') + glob(f'{input}out_valid/*_hdda*.csv')
    L00 = len(file0)

    pred00 = np.empty((len(y), nb_column, L00))
    pred00[:] = np.nan
    pred01 = np.empty((len(y), L00))
    pred01[:] = np.nan

    # Reading prediction files
    for ii, file_name in enumerate(file0):
        data00 = pd.read_csv(f'{file_name}', na_values='NA',usecols=range(1,nb_column+1)) #start range at0 if first column is not index
        if data00.shape[1]== nb_column:
            pred00[:, :, ii] = data00.iloc[40:].fillna(np.nan).values  # Replace 'NA' with numpy.nan
        else:
            print(f'File {file_name} has {data00.shape[1]} columns instead of {nb_column} columns')

        #prediction of the outer loop in out/
        file_name = file_name.replace('out_valid', 'out')
        data01 = pd.read_csv(f'{file_name}', na_values='NA')
        pred01[:, ii] = data01['Predicted proba'].iloc[40:].fillna(np.nan).values

    predY_list = []*len(y)
    scores_list = []*len(y)
    for subfold in range(Nfold):
        print('back count:',Nfold-subfold)
        # Get the index of the testing set
        train_idx, test_idx = foldsCVT[subfold]
        y0 = y[train_idx]
        X0= pred00[train_idx, :, subfold]
        X1 = pred01[test_idx, subfold]

        if X1.ndim == 1:
            X1_reshaped = X1.reshape(-1, 1)
        else:
            X1_reshaped = X1


        best_estimator= hyperparam_tuning(model, param_grid, X0, y0,kf)
        best_estimator.fit(X0,y0)
        predY = best_estimator.predict(X1_reshaped)
        scores = np.array(best_estimator.predict_proba(X1_reshaped)).astype(float)

        # add the prediction at the right index in the lsit
        for i in range(len(test_idx)):
            predY_list[test_idx[i]] = predY[i]
            scores_list[test_idx[i]] = scores[i]

    column_name,list_eval = mf.evaluation(y,predY_list,scores_list)[1:]
    mf.write_files(output,column_name,list_eval)

    #shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Plot
    shap.summary_plot(shap_values, X)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='predicted_proba/', help='input folder')
    parser.add_argument('--output', type=str, default='predicted_proba/', help='output folder with the evaluation of the model',required=False)
    args = parser.parse_args()

    main(args)