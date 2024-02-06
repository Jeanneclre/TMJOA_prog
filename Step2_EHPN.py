'''
Author: Jeanne Claret

This file is used as a step2 for the project TMJOA_PROGNOSIS.
The goal is to combine the bests 3 models from step1 and use their prediction to train a new model.

Input: the files in out/ (prediction probabilities of the outer loop)
and out_valid/ (predictions probabilities of the inner loop)

'''
from sklearn.model_selection import StratifiedKFold, cross_val_score
import sklearn.metrics as mt
from sklearn.ensemble import HistGradientBoostingClassifier

import xgboost as xgb
import lightgbm as lgb


from glob import glob
import argparse
import pandas as pd
import numpy as np
import os

import shap

from Step1 import hyperparam_tuning
import Hyperparameters as hp
import modelFunctions as mf

import time

def split_argList(arg):
    return arg.split(',')

def choose_CombModel(method,seed):

    if method == 0:
        model =HistGradientBoostingClassifier(n_iter_no_change=10,random_state=seed)
        param_grid =  {
        'max_iter': [100, 200, 300],  # Number of boosting iterations.
        'learning_rate': [0.01, 0.1, 0.2],  # Step size for updating the weights.
        'max_depth': [3, 5, 7],  # Maximum depth of each tree.
        'min_samples_leaf': [20, 40, 60],  # Minimum number of samples per leaf.
        'l2_regularization': [0.0, 0.1, 1.0],  # L2 regularization term on weights.
        'max_bins': [255, 200, 100]  # Maximum number of bins used for splitting.
        }
    if method == 1:
        model = xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',random_state=seed)
        param_grid = {
        'n_estimators': [200,300],  # Number of boosting iterations.
        'learning_rate': [0.1],  # Step size shrinkage used in update to prevents overfitting.
        'max_depth': [3],  # Maximum depth of the tree.
        'min_child_weight': [3],  # Minimum sum of instance weight (hessian) needed in a child.
        'gamma': [0,0.01,0.1],  # Minimum loss reduction required to make a further partition on a leaf node of the tree.
        'subsample': [0.9],  # Subsample ratio of the training instances.
        'colsample_bytree': [None,0.7],  # Subsample ratio of columns when constructing each tree.
        'reg_alpha': [0.01,0.1,0.2],  # L1 regularization term on weights.
        'reg_lambda': [None,0.01,0.1,0.2],  # L2 regularization term on weights.
        'scale_pos_weight': [None,1]  # Control the balance of positive and negative weights, useful for unbalanced classes.
        }

    # if method == 2:
    #     model = lgb.LGBMClassifier(random_state=seed)
    #     param_grid = {
    #     'n_estimators': [100, 200, 300],  # Number of boosting iterations.
    #     'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage used in update to prevents overfitting.
    #     'max_depth': [3, 5, 7],  # Maximum depth of the tree.
    #     'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight (hessian) needed in a child.
    #     'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a further partition on a leaf node of the tree.
    #     'subsample': [0.5, 0.7, 0.9],  # Subsample ratio of the training instances.
    #     }

    return model, param_grid


def main(args):
    input = args.input
    output = args.output

    time_start = time.time()
    # Parameters
    nb_column = 10

    # Set seed for reproducibility
    seed =2024
    np.random.seed(seed)


    A = pd.read_csv("TMJOAI_Long_040422_Norm.csv")
    y = A.iloc[40:, 0].values
    print('y shape:',y.shape)
    X = A.iloc[40:, 1:]
    Nfold = 10
    N = 10

    # Choose the model to combine
    model, param_grid = choose_CombModel(1,seed)
    # # Initialize empty matrix for features
    # fea = np.nan * np.empty((X.shape[1], Nfold * N))
    # fea = pd.DataFrame(fea, index=X.columns)


    # Creating folds for cross-validation
    out_kf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=seed)
    in_kf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=seed)
    foldsCVT = [(train_index, test_index) for train_index, test_index in out_kf.split(X,y)]

    # Preparing file paths for reading predictions
    file0 = []
    for model_combine in args.lst_models:
        file0 += glob(f'{input}out_valid/*_{model_combine}*.csv')
    # file0 = glob(f'{input}out_valid/nnet_glmnet.csv') + glob(f'{input}out_valid/nnet_svmLinear.csv') + glob(f'{input}out_valid/nnet_lda2.csv') + glob(f'{input}out_valid/glmnet_lda2.csv') + glob(f'{input}out_valid/glmboost_nnet.csv')
    # file0 = glob(f'{input}out_valid/*_glmnet*.csv') + glob(f'{input}out_valid/*_hdda*.csv')
    L00 = len(file0)
    # print('file0:',file0)
    print('L00:',L00)

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

    predY_list = ['Na']*len(y)
    scores_list = ['Na']*len(y)
    f1_old =0
    for subfold in range(Nfold):
        print(f'=====Loop number {Nfold-subfold}=====')
        # Get the index of the testing set
        train_idx, test_idx = foldsCVT[subfold]
        y0 = y[train_idx]
        X0= pred00[train_idx, subfold, :]
        X1 = pred01[test_idx, :]

        f1_oldIn = 0
        predY_in_list = ['Na']*len(y0)
        scores_in_list = ['Na']*len(y0)
        true_in_label = ['Na']*len(y0)
        for subfoldIn,(trainIn_idx, testIn_idx) in enumerate(in_kf.split(X0,y0)):

            X0_in = X0[trainIn_idx]
            y0_in = y0[trainIn_idx]
            X1_in = X0[testIn_idx]

            best_estimator,method_ht= hyperparam_tuning(model, param_grid, X0_in, y0_in,in_kf)
            best_estimator.fit(X0_in,y0_in)
            predY_in = best_estimator.predict(X1_in)
            scores_in = np.array(best_estimator.predict_proba(X1_in)).astype(float)
            cv_scores = cross_val_score(best_estimator, X0_in, y0_in, cv=in_kf, scoring='roc_auc')
            # print(f'cv_scores: {cv_scores}')

            f1_scoreIn = round(mt.f1_score(y0[testIn_idx], predY_in,average='macro'),3)

            for j in range(len(testIn_idx)):
                true_in_label[testIn_idx[j]] = y0[testIn_idx[j]]
                predY_in_list[testIn_idx[j]] = predY_in[j]
                scores_in_list[testIn_idx[j]] = scores_in[j]

            if f1_scoreIn > f1_oldIn:
                f1_old = f1_scoreIn
                best_trained_estimator = best_estimator
                cv_best_scores = cross_val_score(best_trained_estimator, X0_in, y0_in, cv=in_kf, scoring='roc_auc')

                print(f'----Inner loop {subfoldIn}----')
                print(f'cv_best_scores: {cv_best_scores}')

        # Evaluation Inner loop
        mf.evaluation(true_in_label,predY_in_list,scores_in_list)
        print('best inner model evaluation:',best_trained_estimator.get_params())

        best_trained_estimator.fit(X0,y0)
        predY = best_trained_estimator.predict(X1)
        scores = np.array(best_trained_estimator.predict_proba(X1)).astype(float)
        # add the prediction at the right index in the lsit
        for i in range(len(test_idx)):
            predY_list[test_idx[i]] = predY[i]
            scores_list[test_idx[i]] = scores[i]

        f1_score = round(mt.f1_score(y[test_idx], predY,average='macro'),3)
        if f1_score > f1_old:
            best_outerModel = best_trained_estimator
            best_pred = predY
            best_scores = scores
            best_trueLabel = y[test_idx]
            best_subfold = subfold

    print('----Outer loop----')
    print(f'best outer model evaluation from {best_subfold}:')
    mf.evaluation(best_trueLabel,best_pred,best_scores)

    print('----Final evaluation/ Mean ----')
    print(f'y shape: {len(y)}, predY shape: {len(predY_list)}, scores shape: {len(scores_list)}')
    column_name,list_eval = mf.evaluation(y,predY_list,scores_list)[1:]
    column_name.insert(0,'Models combined')
    list_eval.insert(0,args.lst_models)
    column_name.insert(1,'Method HT')
    list_eval.insert(1,method_ht)
    column_name.insert(len(column_name),'Best estimator param')
    list_eval.insert(len(column_name),best_outerModel.get_params())

    # column_name.insert(0,'Total')
    # list_eval.insert(0,len(predY_list))
    mf.write_files(output,column_name,list_eval)

    # #shap
    # explainer = shap.TreeExplainer(best_outerModel)
    # dtest = xgb.DMatrix(X1)

    # shap_values = explainer.shap_values(dtest)

    # # # Plot
    # shap.summary_plot(shap_values, X1)

    time_end = time.time()
    print(f'Done in {round(time_end - time_start,3)} seconds.')


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='predicted_proba/', help='input folder')
    parser.add_argument('--output', type=str, default='predicted_proba/', help='output folder with the evaluation of the model')
    parser.add_argument('--lst_models', type=split_argList, default=['glmnet', 'hdda'], help='list of the models to combine',required=False)
    args = parser.parse_args()


    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(args)