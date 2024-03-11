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

def split_argList(arg):
    return arg.split(',')

def choose_CombModel(method,seed):

    if method == 0:
        model =HistGradientBoostingClassifier(n_iter_no_change=10,validation_fraction=0.15,random_state=seed)
        param_grid = [
        {'max_iter': [80,150,100,400,500],  # Reduced upper limit to avoid overfitting
        'learning_rate': [0.005,0.01],  # Added a smaller step to explore more fine-grained learning rates
        'max_depth': [2,3],  # Included shallower trees to prevent overfitting
        'min_samples_leaf': [1,2],  # Adjusted to avoid too fine granularity which may lead to overfitting
        'max_leaf_nodes': [10],  # Adjusted to avoid too fine granularity which may lead to overfitting
        'l2_regularization': [0.01],  # Explored higher regularization to control model complexity
        'max_bins': [10,30], # Reduced the range to test the effect of fewer bins
        'class_weight': ['balanced']
        }
        ]

    if method == 1:
        model = xgb.XGBClassifier(objective='binary:logistic',eval_metric=mt.f1_score,random_state=seed)
        # Param grid for combination of theproba of different seeds
        # SVM,glmnet,nnet
        # param_grid = {
        # 'n_estimators': [100, 150],  # Fewer boosting rounds to prevent overfitting.
        # 'learning_rate': [0.1],  # Slightly higher to compensate for fewer rounds but still cautious.
        # 'max_depth': [1,2,3],  # Lower maximum depth to simplify models.
        # 'min_child_weight': [1, 3],  # Including a lower bound to allow for some flexibility.
        # 'gamma': [0.2],  # Increasing the minimum loss reduction for further partitioning.
        # 'subsample': [0.7, 0.9],  # Subsampling less than 1 to prevent overfitting.
        # 'colsample_bytree': [0.5,0.6,0.7],  # Using a smaller subset of features for each tree.
        # 'reg_alpha': [0.1,0.5],  # Increasing L1 regularization to control model complexity.
        # 'reg_lambda': [0.1,0.5],  # Increasing L2 regularization for the same reason.
        # 'scale_pos_weight': [1]  # Assuming balanced classes; adjust as needed for your dataset.
        # }

        # Param grid combination 3 seeds
        # lda2, glmnet, nnet -> The order is really important (accuracy: 67%)
        # param_grid = {
        # 'n_estimators': [100,150,200,500],  # Exploring a slightly broader range
        # 'learning_rate': [0.01,0.05,0.025,0.1],  # Testing lower rates as well
        # 'max_depth': [1,2,3],  # Sticking to simpler models
        # 'max_delta_step': [1,3,7,10],
        # 'min_child_weight': [1,2,3],  # minimum sum of instance weight (hessian) needed in a child
        # 'gamma': [0.05,0.1,0.2,0.3],  # Slight adjustments around 0.1
        # 'subsample': [0.6,0.7,0.75,0.9],  # A broader range around preferred values
        # 'colsample_bytree': [0.3,0.5,0.6,0.7,1],  # number features selected
        # 'reg_alpha': [0, 0.1,0.5,1,2],  # Including a no regularization option
        # 'reg_lambda': [0, 0.1,0.5,1,2],  # Similarly, for L2
        # 'tree_method': ['hist','auto'],
        # 'scale_pos_weight': [1,4]  # Keeping balanced class weight
        # }

        # Param grid for 17/18 bests combined models from Mean_performances seed 2024
        #with eval_metric= mt.f1_score  (79% accuracy)
        param_grid = {
        'n_estimators': [80,200],  # Fewer boosting rounds to prevent overfitting.
        'learning_rate': [0.1],  # Slightly higher to compensate for fewer rounds but still cautious.
        'max_depth': [1,3],  # Lower maximum depth to simplify models.
        'min_child_weight': [1, 3],  # Including a lower bound to allow for some flexibility.
        # 'max_delta_step': [3,7,10],  # Including a lower bound to allow for some flexibility.
        'gamma': [0.05],  # Increasing the minimum loss reduction for further partitioning.
        'subsample': [0.7, 0.9],  # Subsampling less than 1 to prevent overfitting.
        'colsample_bytree': [0.5,0.6,0.7],  # Using a smaller subset of features for each tree.
        'reg_alpha': [0.1,0.25,0.5],  # Increasing L1 regularization to control model complexity.
        'reg_lambda': [0.1,0.25,0.5],  # Increasing L2 regularization for the same reason.
        # 'tree_method': ['auto','hist'],  # Assuming balanced classes; adjust as needed for your dataset.
        'scale_pos_weight': [1]  # Assuming balanced classes; adjust as needed for your dataset.
        }

        #Param grid for 18 best models according to validattion set. 3 seeds.
        # use of GridSearchCV for best results
        # param_grid = {
        # 'n_estimators': [80,100,150,200],  # Fewer boosting rounds to prevent overfitting.
        # 'learning_rate': [0.1],  # Slightly higher to compensate for fewer rounds but still cautious.
        # 'max_depth': [1,3],  # Lower maximum depth to simplify models.
        # 'min_child_weight': [1, 3],  # Including a lower bound to allow for some flexibility.
        # # 'max_delta_step': [3,7,10],  # Including a lower bound to allow for some flexibility.
        # 'gamma': [0.05],  # Increasing the minimum loss reduction for further partitioning.
        # 'subsample': [0.7, 0.9],  # Subsampling less than 1 to prevent overfitting.
        # 'colsample_bytree': [0.5,0.6,0.7],  # Using a smaller subset of features for each tree.
        # 'reg_alpha': [0.1,0.25,0.5],  # Increasing L1 regularization to control model complexity.
        # 'reg_lambda': [0.1,0.25,0.5],  # Increasing L2 regularization for the same reason.
        # # 'tree_method': ['auto','hist'],  # Assuming balanced classes; adjust as needed for your dataset.
        # 'scale_pos_weight': [1]  # Assuming balanced classes; adjust as needed for your dataset.
        # }

    if method == 2:
        model = GradientBoostingClassifier(n_iter_no_change=10,random_state=seed)
        # Param grid for 17 best combined model from validation set- 3 seeds
        # mean acc 3 seeds : 66%
        param_grid= {
        'loss': ['log_loss','exponential'],
        'n_estimators': [100],
        'learning_rate': [0.15,0.3,0.45],
        'max_depth': [1,2],
        'subsample': [0.5,1],
        'min_samples_split': [2,4],  # Increased the minimum number of samples to split
        'max_features' : [0.45,0.5],
        'tol': [0.0001,0.001],
        'ccp_alpha': [0.0001],
        }

        # param_grid= {
        # 'loss': ['log_loss','exponential'],
        # 'n_estimators': [100],
        # 'learning_rate': [0.15,0.3,0.45],
        # 'max_depth': [1,2],
        # 'subsample': [0.5,1],
        # 'min_samples_split': [2,4],  # Increased the minimum number of samples to split
        # 'max_features' : [1],
        # }




    return model, param_grid


def main(args,seed):
    input = args.input
    output = args.output

    time_start = time.time()
    # Parameters
    nb_column = 10

    # Set seed for reproducibility

    np.random.seed(seed)


    A = pd.read_csv("TMJOAI_Long_040422_Norm.csv")
    y = A.iloc[40:, 0].values
    print('y shape:',y.shape)
    X = A.iloc[40:, 1:]
    Nfold = 10
    N = 10

    # Choose the model to combine
    method=1
    model, param_grid = choose_CombModel(method,seed)
    # # Initialize empty matrix for features
    # fea = np.nan * np.empty((X.shape[1], Nfold * N))
    # fea = pd.DataFrame(fea, index=X.columns)


    # Creating folds for cross-validation
    out_kf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=seed)
    in_kf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=seed)
    foldsCVT = [(train_index, test_index) for train_index, test_index in out_kf.split(X,y)]

    # Preparing file paths for reading predictions
    file0 = []
    seeds = [2023,2024,2025]
    list_FS=['glmnet','xgbTree','AUC']
    for seedF in seeds:
        for model_combine in args.lst_models:
            input_path_seedF = input + f'_{seedF}/'
            file0 += glob(f'{input_path_seedF}out_valid/{model_combine}.csv')
            # for FS in list_FS:
            #     file0 += glob(f'{input_path_seedF}out_valid/{FS}_{model_combine}.csv')
    print('input_path_seed:',input_path_seedF)

    # for model_combine in args.lst_models:
    #     file0 += glob(f'{input}out_valid/*_{model_combine}*.csv')
    # file0 = glob(f'{input}out_valid/nnet_glmnet.csv') + glob(f'{input}out_valid/nnet_svmLinear.csv') + glob(f'{input}out_valid/nnet_lda2.csv') + glob(f'{input}out_valid/glmnet_lda2.csv') + glob(f'{input}out_valid/glmboost_nnet.csv')
    # file0 = glob(f'{input}out_valid/*_glmnet*.csv') + glob(f'{input}out_valid/*_hdda*.csv')
    L00 = len(file0)
    # print('file0:',file0)
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


    predY_list = ['Na']*len(y)
    scores_list = ['Na']*len(y)
    y_truelist = ['Na']*len(y)

    auc_inner = []
    f1_inner = []
    acc_inner = []
    for subfold in range(Nfold):
        print(f'=====Loop number {Nfold-subfold}=====')
        # Get the index of the testing set
        train_idx, test_idx = foldsCVT[subfold]
        y0 = y[train_idx]
        X0= pred00[train_idx, subfold, :]
        X1 = pred01[test_idx, :]
        print('X0:',X0.shape)
        print('X1:',X1.shape)
        print('train_idx:',train_idx)

        #save the different file with the training data X0
        output_file= f'{os.path.dirname(output)}/pred00_{Nfold-subfold}_Seed{seed}.csv'
        pred00_df = pd.DataFrame(pred00[:, subfold, :])
        pred00_df.to_csv(output_file)

        if subfold == 0:
            output_fileX0 = f'{os.path.dirname(output)}/X0_{Nfold-subfold}_Seed{seed}.csv'
            X0_df = pd.DataFrame(X0)
            X0_df.to_csv(output_fileX0)

        best_estimator,method_ht= hyperparam_tuning(model, param_grid, X0, y0,in_kf,seed)
        print('best_estimator:',best_estimator.get_params())
        print('method fine tuning:',method_ht)
        # save performances of validation set in inner loop  acc,f1 and auc

        AUCscore_inner = cross_val_score(best_estimator, X0, y0, cv=in_kf, scoring='roc_auc')
        auc_inner.append(AUCscore_inner.mean())
        F1score_inner = cross_val_score(best_estimator, X0, y0, cv=in_kf, scoring='f1')
        f1_inner.append(F1score_inner.mean())
        Accscore_inner = cross_val_score(best_estimator, X0, y0, cv=in_kf, scoring='accuracy')
        acc_inner.append(Accscore_inner.mean())


        predY = best_estimator.predict(X1)
        scores = np.array(best_estimator.predict_proba(X1)).astype(float)

        for i in range(len(test_idx)):
            y_truelist[test_idx[i]] = y[test_idx[i]]
            predY_list[test_idx[i]] = predY[i]
            scores_list[test_idx[i]] = scores[i]

    print('len scores_list:',len(scores_list))
    print('----Outer loop----')
    # mf.evaluation(best_trueLabel,best_pred,best_scores)

    print('----Final evaluation/ Mean ----')
    print(f'y shape: {len(y)}, predY shape: {len(predY_list)}, scores shape: {len(scores_list)}')
    column_name,list_eval = mf.evaluation(y,predY_list,scores_list)[1:]
    column_name.insert(0,'Models combined')
    list_eval.insert(0,args.lst_models)
    column_name.insert(1,'Method HT')
    list_eval.insert(1,method_ht)
    column_name.insert(2,'Seed')
    list_eval.insert(2,seed)
    column_name.insert(len(column_name),'Best estimator param')
    list_eval.insert(len(column_name),best_estimator.get_params())
    column_name.insert(len(column_name),'Param grid used')
    list_eval.insert(len(column_name),param_grid)

    mf.write_files(output,column_name,list_eval)

    #File to save inner loop performance
    InnerPerf_file= f'{os.path.dirname(output)}/eval_innerLoop.csv'
    header =['Models combined','Model to combine ','Method HT','Seed','Accuracy','F1','AUC']
    mean_auc = np.mean(auc_inner)
    mean_f1 = np.mean(f1_inner)
    mean_acc = np.mean(acc_inner)
    data = [args.lst_models,method,method_ht,seed,auc_inner,f1_inner,mean_acc,mean_f1,mean_auc]
    mf.write_files(InnerPerf_file,header,data)
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
    parser.add_argument('--output', type=str, default='Step2/file_result.csv', help='Directory to the file with the evaluation of the models.')
    parser.add_argument('--lst_models', type=split_argList, default=['glmnet', 'hdda'], help='list of the models to combine',required=False)
    args = parser.parse_args()


    out_dir = os.path.dirname(args.output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seeds = [2024,2018,2019,2020,2021,2022,2023,2025,2026,2027,2028,2029,2030,2031,2032]
    for seed in seeds:
        print(f'=====Seed number {seed}=====')
        main(args,seed)