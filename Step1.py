
# Functions used for splitting the data in folds, hyperparameters tuning, feature selection, evaluation, etc.
from sklearn.model_selection import StratifiedKFold, KFold,GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import  SelectFromModel
from sklearn import metrics

# Import models from sklearn and from other packages
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.mixture import GaussianMixture # HDDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb
except:
    import sys
    import subprocess
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', 'xgboost'])
    import xgboost as xgb

# Import other files of the project
import modelFunctions as mf
import Hyperparameters as hp

# Useful libraries
import pickle
import pandas as pd
import numpy as np
import subprocess
import time
import os

import csv

def choose_model(method,seed0):
    """
    Give the right model and the right hyperparameters grid for the method
    Input: method: name of the method
    Output: model: model from the method list, param_grid: hyperparameters grid
    """
    if method == "glmnet" or method == "AUC":
        # fit a logistic regression model using the glmnet package
        model = LogisticRegression(random_state=seed0)
        param_grid = hp.param_grid_lr

    elif method == "svmLinear":
        # fit a linear support vector machine model using the svm package
        model = SVC(probability=True,random_state=seed0)
        param_grid = hp.param_grid_svm

    elif method == "rf":
        # fit a random forest model using the random forest package
        model = RandomForestClassifier(random_state=seed0)
        param_grid = hp.param_grid_rf

    elif method == "xgbTree":
        # fit an XGBoost model using the xgboost package
        model = xgb.XGBClassifier(random_state=seed0)
        param_grid = hp.param_grid_xgb

    elif method == "lda2":
        # fit a linear discriminant analysis model using the LDA package
        model = LinearDiscriminantAnalysis()
        param_grid = hp.param_grid_lda

    elif method == "nnet":
        # fit a neural network model using the MLPClassifier package
        model = MLPClassifier(early_stopping=True,random_state=seed0)
        param_grid = hp.param_grid_nnet

    elif method == "glmboost":
        model = GradientBoostingClassifier(n_iter_no_change=5,random_state=seed0)
        param_grid = hp.param_grid_glmboost

    elif method == "hdda":
        # fit a high-dimensional discriminant analysis model using the pls package
        model = GaussianMixture(n_components=2,random_state=seed0)
        param_grid = hp.param_grid_hdda


    else:
        raise ValueError("Invalid method name. Choose from 'glmnet', 'svmLinear', 'rf', 'xgbTree', 'lda2', 'nnet', 'glmboost', 'hdda'.")

    return model, param_grid

def runFS(modelPM,model,X,y,X_excludedOut,y_excludedOut,param_grid,inner_cv,fold, filename,seed0):
    """
    Features selection -> Best "nb_features" features selected, nb_features depending on the best AUC
    Input: model: model from the method list, X_train: Data, y_train: Target, y_valid: Correct answer
    Output: nb_selected: nb of features selected,
            top_indices: indices of the best features selected
    """
    auc0 =0.5
    nb_features = [5,10,15,20]

    listHeader = ['model FS_PM','fold','5','10','15','20']
    filename = filename.split('/')[0]+'/'+filename.split('/')[1]+'/Results/AUC_featuresSelection.csv'
    auc_perSelection =[f'{model}_{modelPM}',fold]

    nb_selected_features= 0
    auc_perNb=[]
    top_indicesperNb = []
    top_N_largest_indices_final = []
    for nb in nb_features:
        y_trueList_fs = []
        scoresList_fs = []
        top_indicesperSubfold = []
        dict_features = {}
        list_feature = [0]*X.shape[1]

        for subfoldFS, (train_FS_idx, valid_FS_idx) in enumerate(inner_cv.split(X,y)):
            print(f'________Feature Selection {subfoldFS}_________')
            print('valid_idx',valid_FS_idx.shape)
            print('train_idx',train_FS_idx.shape)

            # Split data
            X_train, X_valid = X[train_FS_idx], X[valid_FS_idx]
            y_train, y_valid = y[train_FS_idx], y[valid_FS_idx]

            # Add 40 data back into the training set
            # X_train = np.concatenate((X_excludedOut,X_train),axis=0)
            # y_train = np.concatenate((y_excludedOut,y_train),axis=0)

            model_hyp = hyperparam_tuning(model, param_grid, X_train, y_train, inner_cv,seed0)[0]
            print('model_hyp in runFS',model_hyp)
            model_hyp.fit(X_train, y_train)

            # If the model_hyp doesn't have a coef_ attribute, use feature_importances_ attribute
            if hasattr(model_hyp, "feature_importances_"):
                Coefficient = np.abs(model_hyp.feature_importances_)
            if hasattr(model_hyp, "coef_"):
                Coefficient=np.abs(model_hyp.coef_)[0]
            if not hasattr(model_hyp, "feature_importances_") and not hasattr(model_hyp, "coef_"):
                # For the nnet model, use the sum of the absolute values of the weights
                Coefficient = np.sum(np.abs(model_hyp.coefs_[0]), axis=1)

            features_auc =[]
            features_auc_index = []

            # sort the coefficients from the smallest to the largest
            sorted_indices = np.argsort(Coefficient)


            if model == 'AUC':
                #use AUC to score the features importance or the coef
                for i, col in enumerate(X_train.T):
                    # Calculate AUC score for this feature
                    features_imp = metrics.roc_auc_score(y_valid, X_valid[:,i:i+1]) #probs

                    features_auc_index.append(i)
                    features_auc.append(features_imp)

                # Sort features by AUC score
                sorted_features_index = np.argsort(features_auc)
                top_N_largest_indices = sorted_features_index[-nb:]
                for ind in top_N_largest_indices:
                    if ind in dict_features:
                        dict_features[ind].append(features_auc[ind])
                    else:
                        dict_features[ind] = features_auc[ind]

            else:
                # select the top nb_features (nb) largest indices
                top_N_largest_indices = sorted_indices[-nb:]

                for ind in top_N_largest_indices:
                    if ind in dict_features:
                        dict_features[ind].append(Coefficient[ind])
                    else:
                        dict_features[ind] = [Coefficient[ind]]

            # Save predicted proba, and y_true in a list for final evaluation
            X_train_selected = X_train[:,top_N_largest_indices]
            model_hyp.fit(X_train_selected, y_train)
            X_valid_selected = X_valid[:,top_N_largest_indices]
            y_scores_fs = model_hyp.predict_proba(X_valid_selected)[:,1]

            for j in range(len(y_valid)):
                y_trueList_fs.append(y_valid[j])
                scoresList_fs.append(y_scores_fs[j])

            top_indicesperSubfold.append(top_N_largest_indices)

        auc = round(metrics.roc_auc_score(y_trueList_fs,scoresList_fs),3)
        auc_perNb.append(auc)
        top_indicesperNb.append(top_indicesperSubfold) # Should be a list of lists of lists of indices


        # Save the nb bests features indices out of the 10 subfold with this function: sum(lst_coef)/(nb-len(lst_coef))
        # This way,we use the number of times a feature was selected AND the value of its importance.
        for key in dict_features:
            if len(dict_features[key])==nb:
                list_feature[key]=sum(dict_features[key]) # avoid 0 division
            else:
                list_feature[key]=sum(dict_features[key])/(nb-len(dict_features[key]))

        auc_perSelection.append(auc)
        if auc >=auc0:
            auc0 = auc
            nb_selected_features = nb
            top_N_largest_indices_final = np.argsort(list_feature)[-nb_selected_features:] #save indices of the best features selected
            print('top_N_largest_indices_final',top_N_largest_indices_final)

    if len(top_N_largest_indices_final) == 0:
        print("\033[91m-- No features selected - all of them where kept --\033[0m")
        top_N_largest_indices_final = np.arange(X.shape[1])

    mf.write_files(filename,listHeader,auc_perSelection)
    return nb_selected_features, top_N_largest_indices_final

def hyperparam_tuning(model, param_grid, X_train, y_train, inner_cv,seed0):
    '''
    Looking for the best set of hyperparameter for the model used from the grid of hyperparameters.
    RandomizedSearchCV is used to find the best hyperparameters because it is faster than GridSearchCV and avoid overfitting.

    Input: model: model from the method list, param_grid: hyperparameters grid, X_train: Data, y_train: Target,
                inner_cv: cross-validation method and seed0: seed
    Output: best_estimator: model with the best hyperparameters
    '''
    scorer = metrics.make_scorer(metrics.roc_auc_score)

    if len(param_grid) > 1:
        method = 'RandomizedSearchCV'
        hyperparam_tuning = RandomizedSearchCV(estimator=model,
                            param_distributions=param_grid,
                            n_iter=10,
                            scoring=scorer,
                            n_jobs=-1,
                            cv=inner_cv,
                            verbose=0,
                            refit=True,
                            error_score='raise',
                            random_state=seed0)
    else: # Currently not used
        method = 'GridSearchCV'
        hyperparam_tuning = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring=scorer,
                            n_jobs=-1,
                            cv=inner_cv,
                            verbose=0,
                            refit=True,
                            error_score='raise')

    hyperparam_tuning.fit(X_train, y_train)
    best_estimator = hyperparam_tuning.best_estimator_
    return best_estimator,method


def run_middleLoop(methodFS,methodPM, filename,X_trainOut,y_trainOut, X_excludedOut,y_excludedOut,testOut_idx,fold,df_predict,seed0):
    '''
    The purpose of the middle loop is to select the best features of the dataset for
    the prediction.

    This loop is splitted in 2 parts:

    1) Feature selection with the methodFS (model for the feature selection)
    2) Call the inner loop with the training set of the middle -- The inner loop is the hyperparameter tuning of the PM model

    The training data from the folds of the outer loop are splitted in NsubFolds.
    We evaluate the generalization of the model on the validation set of the middle loop.

    -> We keep the predicted probabilities of each best_estimator from the hyperparameter tuning of the inner loop in a csv file.
    This file must contains Nfolds columns and 74 rows (74 is the total number of data in the dataset - 40 excluded data 'NA' values)

    Input: methodFS: name of the method for the feature selection, methodPM: name of the method for the predictive model,
           filename: name of the file for the results, X: Data, y: Target, fold: index of the outer fold
           df_predict: dataframe to save the predictions proba of the inner loop, seed0: seed

    Output: predYT: Predictions of the model, y_trueList: Correct answer, bestM_filename: Name of the file for the best model,
            NbFeatures: Number of features selected
    '''
    # Take the right model according to the name of the method
    modelFS,param_gridFS = choose_model(methodFS,seed0)
    modelPM,param_gridPM = choose_model(methodPM,seed0)

    # Remove the first 4 datas from the validation set to use them as training set
    print('X shape',X_trainOut.shape)
    print('y shape',y_trainOut.shape)

    # Split the data in NsubFolds
    Nsubfolds = 10
    inner_cv = StratifiedKFold(n_splits=Nsubfolds, shuffle=True, random_state=seed0)
    fs_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed0)
    # Init lists for the evaluation
    predYT = [] # all predictions
    y_trueList = [] # all correct answers
    scoresList = [] # all probabilities of the predictions

    predYT_train = [] # all predictions
    y_trueList_train = [] # all correct answers
    scoresList_train = [] # all probabilities of the predictions

    # Init variables for the best model
    idx = 0
    acc0,f1_scoresMid_init = 0.5,0
    file_toDel = 0
    bestMM_filename = filename.split('.')[0]+'_bestMModel'+'_NULL'+'.pkl'

    best_top40 = []

    best_predict_proba=['NA']*74 ## 74 is the total number of data in the dataset
    # Loop
    valid_idx_list = []
    list_data_idx = []

    filenameMid =filename.split('/')[0]+'/'+filename.split('/')[1]+'/Results/AUC_middleloop.csv'
    header = ['model FS_PM','Fold Out','AUC train','AUC','F1 train','F1']
    eval_midRow = [f'{methodFS}_{methodPM}',fold]
    auc_midRow = []
    f1_midRow = []

    filename_Bestparam = filename.split('/')[0]+'/'+filename.split('/')[1]+'/Results/BestParam_middleloop.csv'
    header_bestParam = ['model FS_PM']
    best_param_row = [f'{methodFS}_{methodPM}']

    NbFeatures, top_features_idx = runFS(methodPM,modelFS,X_trainOut,y_trainOut,X_excludedOut,y_excludedOut,param_gridFS,fs_cv,fold,filename,seed0)

    for subfold, (train_idx, valid_idx) in enumerate(inner_cv.split(X_trainOut,y_trainOut)):
        print(f'________Middle Loop {subfold}_________')
        header_bestParam.append(f'Fold {subfold}')

        nb_loop = f'{fold}-{subfold}'
        idx += 1
        # print the day and hour
        print('run began at',time.localtime())
        print('testOut_idx_',testOut_idx)
        print('valid_idx',valid_idx)
        print('train_idx',train_idx.shape)
        valid_idx_list.append(valid_idx)

        # Split data
        X_train, X_valid = X_trainOut[train_idx], X_trainOut[valid_idx]
        y_train, y_valid = y_trainOut[train_idx], y_trainOut[valid_idx]

        # Remove unnecessary features
        print('len(top_features_idx)',len(top_features_idx))
        X_excludedOut_selected = X_excludedOut[:,top_features_idx]
        X_trainIn_selected = X_train[:,top_features_idx]
        X_valid_selected = X_valid[:,top_features_idx]

        # Add the excluded top 40 data back into the training set
        X_train_selected= np.concatenate((X_excludedOut_selected,X_trainIn_selected),axis=0)
        y_train = np.concatenate((y_excludedOut,y_train),axis=0)

        # Hyperparameter tuning for the predictive model (PM)
        bestIn_estimator=hyperparam_tuning(modelPM, param_gridPM, X_train_selected, y_train, inner_cv,seed0)[0]
        print('\033[94m methodPM:',methodPM)
        print(f"\033[94m estimator's parameters: {bestIn_estimator.get_params()} \033[0m")

        best_param_row.append(bestIn_estimator.get_params())

        # Evaluation validation sets
        y_scores = bestIn_estimator.predict_proba(X_valid_selected)[:,1]
        y_pred = bestIn_estimator.predict(X_valid_selected)

        # Save predictions in the lists for total evaluation
        for i in range(len(y_pred)):
            predYT.append(y_pred[i])
            y_trueList.append(y_valid[i])
            scoresList.append(y_scores[i])

        #Evaluation training sets
        y_scoresTrain = bestIn_estimator.predict_proba(X_train_selected)[:,1]
        y_predTrain = bestIn_estimator.predict(X_train_selected)

        for i in range(len(y_predTrain)):
            predYT_train.append(y_predTrain[i])
            y_trueList_train.append(y_train[i])
            scoresList_train.append(y_scoresTrain[i])
      

        '''
        Give the real data index of the validation set for /out_valid
        X doesn't have the same shape as in the outer loop
        so we need to calculate the data index given by valid_idx
        and add 40 because we remove the 40 first data from X
        '''
        count_incr=0 #count how many increment we should add to the index
        len_testOut_idx = len(testOut_idx)

        for i, data_idx in enumerate(valid_idx):
            good_idx = 40
            if data_idx < testOut_idx[0]:
                good_idx += data_idx
            else :
                for i in range(len_testOut_idx-1):

                    if testOut_idx[i] <= data_idx < testOut_idx[i+1]:
                        count_incr=i+1

                if  testOut_idx[len(testOut_idx)-2] <= data_idx < testOut_idx[len(testOut_idx)-1]:
                    count_incr += 1
                if data_idx >= testOut_idx[len(testOut_idx)-1]:
                    count_incr= len(testOut_idx)

                good_idx =good_idx+ data_idx+count_incr

            list_data_idx.append([data_idx+40,good_idx])

            best_predict_proba[good_idx] = round(y_scores[i],4)


    # Save predicted probabilities of the inner loop in a csv file
    mount_point =filename.split('/')[0]+'/'+filename.split('/')[1]
    if not os.path.exists(os.path.dirname(mount_point+'/out_valid/')):
        os.makedirs(os.path.dirname(mount_point+'/out_valid/'))
    prediction_filename = f"{mount_point}/out_valid/"+f'{methodFS}_{methodPM}.csv'
    df_predict.insert(fold-1,None,best_predict_proba)
    df_predict.to_csv(prediction_filename, index=True)

    # Save Evaluation of the middle loop
    auc_midRow= round(metrics.roc_auc_score(y_trueList,scoresList),3)
    f1_midRow = round(metrics.f1_score(y_trueList,predYT, average='macro'),3)

    auc_midRow_train= round(metrics.roc_auc_score(y_trueList_train,scoresList_train),3)
    f1_midRow_train = round(metrics.f1_score(y_trueList_train,predYT_train, average='macro'),3)

    eval_midRow.append(auc_midRow_train)
    eval_midRow.append(auc_midRow)
    eval_midRow.append(f1_midRow_train)
    eval_midRow.append(f1_midRow)
    mf.write_files(filenameMid,header,eval_midRow)

    mf.write_files(filename_Bestparam,header_bestParam,best_param_row)

  
    return predYT, y_trueList, bestMM_filename,NbFeatures, top_features_idx, auc_midRow, f1_midRow, bestIn_estimator,auc_midRow_train, f1_midRow_train



def OuterLoop(X, y,methodFS, methodPM, innerL_filename, outerL_filename,folder_output,seed0):
    """
    Outer loop of the nested cross validation:

    Call the inner loop for each fold of the outer loop
    Evaluate the best model of the inner loop on the test set of the outer loop
    then keep the best model of the outer loop according to the AUC.

    Lot of files are created during the process:
    - "<innerL_filename>.csv" file: Evaluation of the inner loop models
    - "<innerL_filename>_bestModel_<fold>.pkl" file: Best model of the inner loop
    - "<innerL_filename>_Innerpredictions<fold>.csv" file: Predictions of the inner loop models

    - "<outerL_filename>.csv" file: Evaluation of the outer loop models
    - "<outerL_filename>_bestModelOuter_<fold>.pkl" file: Best model of the outer loop
    - "<outerL_filename>_Finalpredictions.csv" file: Predictions of the outer loop models
    - "<outerL_filename>_Evaluation.csv" file: Evaluation of the best model of the outer loop

    - "Final_Performance48.csv": Evaluation of all the models runned in the outer loop

    Input: X: Data, y: Target, innerL_filename: Name of the file for inner loop results,
    outerL_filename: Name of the file for outer loop results, methods_list: List of the methods, iii: index of the method
    Output: // files //
    """
    print('********Outer Loop**********')

    Nfold = 10
    # seed0 = 2024
    # seed0 = np.random.seed(seed0)
    print(f' \033[34mseed0: {seed0} \033[0m')

    X_excluded = X[:40,:]
    y_excluded = y[:40]

    X_remaining = X[40:,:]
    y_remaining = y[40:]


    folds_CVT = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=seed0)

    y_predicts = []
    y_scoresList = []  # Save predictions in the lists
    y_trueList = []

    scores_trainList = []
    y_trainList = []
    y_predictsTrain = []

    # Files to save evaluation
    mid_auc = []
    mid_f1 = []
    mid_aucTrain = []
    mid_f1Train = []

    header_mid_AUCeval,header_mid_f1eval = ['model FS_PM'],['model FS_PM']
    row_mid_AUCeval,row_mid_f1eval = [f'{methodFS}_{methodPM}'],[f'{methodFS}_{methodPM}']
    header_topFeatures =['model FS_PM']
    row_topFeatures = [f'{methodFS}_{methodPM}']

    nb_features_selected= []
    top_features_outer = []
    best_top40 = []

    df_predict_inner = pd.DataFrame() # dataframe to save the predictions proba of the inner loop
    best_predict_proba=['NA']* np.shape(X)[0] ## 74 is the total number of data in the dataset

    idx_split_train={}
    idx_split_test={}

    for fold, (train_idx, test_idx) in enumerate(folds_CVT.split(X_remaining,y_remaining)):
        print(f"================Fold {fold+1}================")
        header_mid_AUCeval.append(f'AUC {fold}')
        header_mid_f1eval.append(f'F1 {fold}')
        header_topFeatures.append(f'Fold {fold}')

        # Split data
        X_train, X_test = X_remaining[train_idx], X_remaining[test_idx]
        y_train, y_test = y_remaining[train_idx], y_remaining[test_idx]

        print('X_remaining shape',X_remaining.shape)
        print('train_idx',train_idx)
        idx_split_train[fold] = train_idx
        idx_split_test[fold] = test_idx
        print('X_train shape',X_train.shape)

       

        predictInL, correctPred_InL, bestInnerM_filename, NbFeatures, top_features_idx, auc_validation, f1_validation,bestMid_estimator,auc_midTrain,f1_midTrain = run_middleLoop(methodFS,methodPM, innerL_filename,X_train,y_train,X_excluded,y_excluded,test_idx,fold+1,df_predict_inner,seed0)

        nb_features_selected.append(NbFeatures)
        row_topFeatures.append(f"{NbFeatures}: {top_features_idx}")
        mid_auc.append(auc_validation)
        row_mid_AUCeval.append(auc_validation)
        mid_f1.append(f1_validation)
        row_mid_f1eval.append(f1_validation)

        mid_aucTrain.append(auc_midTrain)
        mid_f1Train.append(f1_midTrain)

        # Add the excluded top 40 data back into the training set
        X_train = np.concatenate(( X_excluded, X_train), axis=0)
        y_train = np.concatenate((y_excluded,y_train ), axis=0)
        print('\033[34mX_train shape',X_train.shape)
        print('y_train shape\033[0m',y_train.shape)

        # Test the best model from Middle loop
        # best_innerModel = pickle.load(open(bestInnerM_filename,'rb'))
        # bestMid_estimator.fit(X_train[:,top_features_idx], y_train)
        y_Fpred = bestMid_estimator.predict(X_test[:,top_features_idx])

        # Test if the model has a predict_proba method
        if hasattr(bestMid_estimator, "predict_proba"):
            y_Fscores = bestMid_estimator.predict_proba(X_test[:,top_features_idx])[:,1]
        else:
            y_Fscores = ['NA'] * len(y_Fpred)

        # Save all predictions in lists to evaluate the entire outer loop and not each fold
        for i in range(len(y_Fpred)):
            y_predicts.append(y_Fpred[i])
            y_scoresList.append(y_Fscores[i])
            y_trueList.append(y_test[i])

        # Test the training dataset
        y_predTrain = bestMid_estimator.predict(X_train[:,top_features_idx])
        y_scoresTrain = bestMid_estimator.predict_proba(X_train[:,top_features_idx])[:,1]
        y_scoresTrain = np.array(y_scoresTrain).astype(float)

        for i in range(len(y_predTrain)):
            y_predictsTrain.append(y_predTrain[i])
            y_trainList.append(y_train[i])
            scores_trainList.append(y_scoresTrain[i])

        #Keep the best model with the best AUC

        for idx, data_idx in enumerate(test_idx):
                best_predict_proba[data_idx+40] = np.round(y_Fscores[idx],4)


    print(f'shape of y_trueList {np.shape(y_trueList)}, shape of y_predicts {np.shape(y_predicts)}, shape of y_scoresList {np.shape(y_scoresList)}')
    auc_test = round(metrics.roc_auc_score(y_trueList,y_scoresList),3)
    auc_train = round(metrics.roc_auc_score(y_trainList,scores_trainList),3)
    f1_test = round(metrics.f1_score(y_trueList,y_predicts, average='macro'),3)
    f1_train = round(metrics.f1_score(y_trainList,y_predictsTrain, average='macro'),3)

    globalEval_filename = outerL_filename.split('/')[0]+'/'+outerL_filename.split('/')[1]+'/Results/GlobEvaluation.csv'
    column_nameTrain = ['Model FS_PM' ,'AUC train (O)','AUC test (O)','AUC train (M)','AUC validation (M)','F1 train (O)','F1 test (O)','F1 train (M)','F1 validation (M)']
    list_evalTrain = [f'{methodFS}_{methodPM}',auc_train,auc_test,sum(mid_aucTrain)/len(mid_aucTrain),sum(mid_auc)/len(mid_auc),f1_train,f1_test,sum(mid_f1Train)/len(mid_f1Train),sum(mid_f1)/len(mid_f1)]
    mf.write_files(globalEval_filename,column_nameTrain,list_evalTrain)

    # Save predictions of the inner loop in a csv file
    prediction_filename = innerL_filename.split('.')[0]+f'_Innerpredictions{fold}'+'.csv'
    df_predict = pd.DataFrame({'Actual':correctPred_InL , 'Predicted': predictInL})
    df_predict.to_csv(prediction_filename, index=False)

    # AUC from middle loop
    auc_filename = outerL_filename.split('/')[0]+'/'+outerL_filename.split('/')[1]+'/Results/AUC_middleloop_fromOut.csv'
    mf.write_files(auc_filename,header_mid_AUCeval,row_mid_AUCeval)
    # F1 from middle loop
    f1_filename = outerL_filename.split('/')[0]+'/'+outerL_filename.split('/')[1]+'/Results/F1_middleloop_fromOut.csv'
    mf.write_files(f1_filename,header_mid_f1eval,row_mid_f1eval)
    # Top features from middle loop
    topFeatures_filename = outerL_filename.split('/')[0]+'/'+outerL_filename.split('/')[1]+'/Results/TopFeatures_middleloop_fromOut.csv'
    mf.write_files(topFeatures_filename,header_topFeatures,row_topFeatures)

    # Evaluation of outer loop models and save results in files
    print("********Final**********")
    column_name,list_eval = mf.evaluation(y_trueList,y_predicts,y_scoresList)[1:]
    column_name.insert(0,'Nb Features Selected')
    list_eval.insert(0,f'{nb_features_selected}')
    mf.write_files(outerL_filename,column_name,list_eval)

    #Save evaluation in a csv file for each model MethodPM_MethodFS that is runned in outerloop and add to the previous data
    # Add in the first index the name of the model
    list_eval.insert(0,f'{methodFS}_{methodPM}')
    column_name.insert(0,'Model FS_PM')
    performance_filename = outerL_filename.split('/')[0]+'/'+outerL_filename.split('/')[1]+'/Results/Performances48.csv'
    mf.save_performance(list_eval, column_name, performance_filename)

    # Save predictions of the outer loop in a csv file
    prediction_filename = outerL_filename.split('.')[0]+'_Finalpredictions'+'.csv'
    df_predict = pd.DataFrame({'Actual': y_trueList, 'Predicted': y_predicts})
    df_predict.to_csv(prediction_filename, index=False)

    # Save the best predicted probabilities of the outer loop in a csv file
    if not os.path.exists(os.path.dirname(folder_output+'out/')):
        os.makedirs(os.path.dirname(folder_output+'out/'))
    prediction_filename = f'{folder_output}out/{methodFS}_{methodPM}.csv'
    df_predict = pd.DataFrame({'Predicted proba': best_predict_proba})
    df_predict.to_csv(prediction_filename, index=False)

    return top_features_outer, nb_features_selected, best_top40
