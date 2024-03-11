'''
Author: Jeanne Claret

This file is used as a step2 for the project TMJOA_PROGNOSIS.
The goal is to combine the bests 3 models from step1 and use their prediction to train a new model.

Input: the files in out/ (prediction probabilities of the outer loop)
and out_valid/ (predictions probabilities of the inner loop)

'''
from sklearn.model_selection import StratifiedKFold, cross_val_score
import sklearn.metrics as mt
from sklearn.ensemble import HistGradientBoostingClassifier,GradientBoostingClassifier

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
import shap

device ='cuda'

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
        model = xgb.XGBClassifier(objective='binary:logistic',eval_metric=mt.f1_score,random_state=seed,)
        # param_grid = {
        # 'n_estimators': [200,300,500],  # Number of boosting iterations.
        # 'learning_rate': [0.01],  # Step size shrinkage used in update to prevents overfitting.
        # 'max_depth': [2,3],  # Maximum depth of the tree.
        # 'min_child_weight': [3],  # Minimum sum of instance weight (hessian) needed in a child.
        # 'gamma': [0,0.01,0.1],  # Minimum loss reduction required to make a further partition on a leaf node of the tree.
        # 'subsample': [0.9],  # Subsample ratio of the training instances.
        # 'colsample_bytree': [None,0.7],  # Subsample ratio of columns when constructing each tree.
        # 'reg_alpha': [0.01,0.1,0.2],  # L1 regularization term on weights.
        # 'reg_lambda': [None,0.01,0.1,0.2],  # L2 regularization term on weights.
        # 'scale_pos_weight': [None,1]  # Control the balance of positive and negative weights, useful for unbalanced classes.
        # }
        param_grid = {
        'n_estimators': [100, 150],  # Fewer boosting rounds to prevent overfitting.
        'learning_rate': [0.05, 0.1],  # Slightly higher to compensate for fewer rounds but still cautious.
        'max_depth': [2, 3],  # Lower maximum depth to simplify models.
        'min_child_weight': [1, 3],  # Including a lower bound to allow for some flexibility.
        'gamma': [0.1, 0.2],  # Increasing the minimum loss reduction for further partitioning.
        'subsample': [0.8, 0.9],  # Subsampling less than 1 to prevent overfitting.
        'colsample_bytree': [0.6, 0.7],  # Using a smaller subset of features for each tree.
        'reg_alpha': [0.1, 0.5],  # Increasing L1 regularization to control model complexity.
        'reg_lambda': [0.1, 0.5],  # Increasing L2 regularization for the same reason.
        'scale_pos_weight': [1]  # Assuming balanced classes; adjust as needed for your dataset.
        }


    if method == 2:
        model = GradientBoostingClassifier(n_iter_no_change=10,random_state=seed)
        # Param grid for 17 best combined model from validation set- 3 seeds
        param_grid= {
        'loss': ['log_loss','exponential'],
        'n_estimators': [80,100,250,300],
        'learning_rate': [0.15,0.3,0.45],
        'max_depth': [1,2,3],
        'subsample': [0.5,0.9],
        'min_samples_split': [2,4],  # Increased the minimum number of samples to split
        'max_features' : [0.45,0.5],
        'tol': [0.0001,0.001],
        'ccp_alpha': [0.0001],
        }

    return model, param_grid


def main(args,seed):
    input = args.input
    output = args.output

    time_start = time.time()
    # Parameters
    nb_column = 10

    # Set seed for reproducibility
    # seed =2024
    np.random.seed(seed)
    print('seed:',seed)

    A = pd.read_csv("TMJOAI_Long_040422_Norm.csv")
    y = A.iloc[40:, 0].values
    print('y shape:',y.shape)
    X = A.iloc[40:, 1:]
    Nfold = 10
    N = 10

    # Choose the model to combine
    method=2
    model, param_grid = choose_CombModel(method,seed)

    # Creating folds for cross-validation
    out_kf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=seed)
    in_kf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=seed)
    foldsCVT = [(train_index, test_index) for train_index, test_index in out_kf.split(X,y)]

    # Preparing file paths for reading predictions
    file0 = []
    seeds = [2023,2024,2025]
    for seed in seeds:
        for model_combine in args.lst_models:
            input_path_seed = input + f'_{seed}/'
            print('input_path_seed:',input_path_seed)
            file0 += glob(f'{input_path_seed}out_valid/{model_combine}.csv')
    # file0 = glob(f'{input}out_valid/rf_nnet.csv')+ glob(f'{input}out_valid/rf_glmnet.csv') + glob(f'{input}out_valid/rf_svmLinear.csv') + glob(f'{input}out_valid/xgbTree_lda2.csv') + glob(f'{input}out_valid/xgbTree_nnet.csv') + glob(f'{input}out_valid/rf_lda2.csv')
    # file0 = file0 + glob(f'{input}out_valid/glmnet_nnet.csv') + glob(f'{input}out_valid/AUC_nnet.csv')+glob(f'{input}out_valid/AUC_lda2.csv')+glob(f'{input}out_valid/xgbTree_glmnet.csv')+glob(f'{input}out_valid/xgbTree_svmLinear.csv')+glob(f'{input}out_valid/glmnet_lda2.csv')
    # file0= file0+glob(f'{input}out_valid/nnet_hdda.csv')+glob(f'{input}out_valid/glmnet_glmnet.csv')+glob(f'{input}out_valid/AUC_glmnet.csv')+glob(f'{input}out_valid/glmnet_svmLinear.csv')+glob(f'{input}out_valid/nnet_svmLinear.csv')+glob(f'{input}out_valid/AUC_svmLinear.csv')
    # file0 = glob(f'{input}out_valid/*_glmnet*.csv') + glob(f'{input}out_valid/*_hdda*.csv')
    L00 = len(file0)

    print('L00:',L00)

    pred00 = np.empty((len(y), nb_column, L00))
    pred00[:] = np.nan
    pred01 = np.empty((len(y), L00))
    pred01[:] = np.nan


    if method==2:
        na_replacement = 0.5
    else:
        na_replacement = np.nan
    # Reading prediction files
    for ii, file_name in enumerate(file0):
        data00 = pd.read_csv(f'{file_name}', na_values='NA',usecols=range(1,nb_column+1)) #start range at0 if first column is not index
        if data00.shape[1]== nb_column:
            pred00[:, :, ii] = data00.iloc[40:].fillna(na_replacement).values  # Replace 'NA' with numpy.nan
        else:
            print(f'File {file_name} has {data00.shape[1]} columns instead of {nb_column} columns')

        #prediction of the outer loop in out/
        file_name = file_name.replace('out_valid', 'out')
        data01 = pd.read_csv(f'{file_name}', na_values='NA')
        pred01[:, ii] = data01['Predicted proba'].iloc[40:].fillna(na_replacement).values

    y_trueList = ['Na']*len(y)
    predY_list = ['Na']*len(y)
    scores_list = ['Na']*len(y)
    f1_old =0
    for fold in range(Nfold):
        print(f'=====Loop number {Nfold-fold}=====')
        # Get the index of the testing set
        train_idx, test_idx = foldsCVT[fold]
        y0 = y[train_idx]
        X0= pred00[train_idx, fold, :]
        X1 = pred01[test_idx, :]

        # if fold == 0:
        #     X_unseen = X0
        #     y_unseen = y0

        #     X1_unseen = X1
        #     test_idx_unseen = test_idx
        #     continue

        for subfold_in, (traind_in_idx, valid_idx) in enumerate(in_kf.split(X0,y0)):
            X0_in = X0[traind_in_idx]
            y0_in = y0[traind_in_idx]
            # X1_in = X0[valid_idx]
            # y1_in = y0[valid_idx]

            estimator_in,method_ht= hyperparam_tuning(model, param_grid, X0_in, y0_in,in_kf,seed)
            print('method fine tuning:',method_ht)

            # # estimator_in.fit(X0_in,y0_in)
            # f1_score_in = round(mt.f1_score(y_unseen, estimator_in.predict(X_unseen),average='macro'),3)
            # print(f'f1_score_in: {f1_score_in}')

            # if f1_score_in > f1_old:
            #     best_estimator_in = estimator_in
            #     f1_old = f1_score_in
            #     best_subfold_in = subfold_in
            #     print(f'best subfold_in: {best_subfold_in}')
            #     # print('best estimator_in:',best_estimator_in)
        best_estimator_in = estimator_in


        predY = best_estimator_in.predict(X1)
        scores = np.array(best_estimator_in.predict_proba(X1)).astype(float)
        # add the prediction at the right index in the lsit
        for i in range(len(test_idx)):
            y_trueList[test_idx[i]] = y[test_idx[i]]
            predY_list[test_idx[i]] = predY[i]
            scores_list[test_idx[i]] = scores[i]

    # predict_unseen = best_estimator_in.predict(X1_unseen)
    # scores_unseen = np.array(best_estimator_in.predict_proba(X1_unseen)).astype(float)

    for j in range(len(test_idx)):
        y_trueList[test_idx[j]] = y[test_idx[j]]
        predY_list[test_idx[j]] = predY[j]
        scores_list[test_idx[j]] = scores[j]


    # Assuming `model` is your trained model and `X_test` is your test dataset.


    print('----Outer loop----')
    # print(f'best outer model evaluation from {best_subfold}:')
    # mf.evaluation(best_trueLabel,best_pred,best_scores)

    print('----Final evaluation/ Mean ----')
    print(f'y shape: {len(y)}, predY shape: {len(predY_list)}, scores shape: {len(scores_list)}')

    column_name,list_eval = mf.evaluation(y_trueList,predY_list,scores_list)[1:]
    column_name.insert(0,'Models combined')
    list_eval.insert(0,args.lst_models)
    column_name.insert(1,'Method HT')
    list_eval.insert(1,method_ht)
    # column_name.insert(len(column_name),'Best estimator param')
    # list_eval.insert(len(column_name),best_outerModel.get_params())

    # column_name.insert(0,'Total')
    # list_eval.insert(0,len(predY_list))
    mf.write_files(output+'ModelEval.csv',column_name,list_eval)

    #write file with the predictions
    df = pd.DataFrame({'True label':y_trueList,'Predicted label':predY_list,'Predicted proba':scores_list})
    df.to_csv(output+'Predictions_trueLab.csv',index=False)

    # explainer = shap.TreeExplainer(best_estimator_in)
    # shap_values = explainer.shap_values(pred01)

    # # For visualization, you can use a summary plot to see the most important features:
    # shap.summary_plot(shap_values, pred01, feature_names=A.columns)

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

    seeds = [2023,2024,2025]
    for seed in seeds:
        print(f'=====Seed number {seed}=====')
        main(args,seed)