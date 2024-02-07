
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

def runFS(modelPM,model,X_train,X_valid,y_train,y_valid,param_grid,inner_cv,fold_subfold):
    """
    Features selection -> Best "nb_features" features selected, nb_features depending on the best AUC
    Input: model: model from the method list, X_train: Data, y_train: Target, y_valid: Correct answer
    Output: nb_selected: nb of features selected,
            X_train_selected: Data with selected features,
            X_valid_selected: Data with selected features
    """

    auc0, f1_scoresFS_init =0.5,0
    nb_features = [5,10,15,20]
    model_hyp = hyperparam_tuning(model, param_grid, X_train, y_train, inner_cv)[0]
    print('model_hyp in runFS',model_hyp)
    model_hyp.fit(X_train, y_train) # should I choose hyperparam before?

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
    nb_selected_features= 0
    top_N_largest_indices_final = []
    top_40_largest_indices = []

    X_trainSelected_final = X_train
    X_validSelected_final = X_valid
    top_N_largest_indices_final= sorted_indices[:]

    listHeader = ['model FS_PM','fold-subfold','5','10','15','20']
    filename = 'Results/TestAUC/PbFea4/AUC_featuresSelection.csv'
    auc_perSelection =[f'{model}_{modelPM}',fold_subfold]

    for nb in nb_features:

        if model == 'AUC':
            #use AUC to score the features importance or the coef
            for i, col in enumerate(X_train.T):
                # Calculate AUC score for this feature
                features_imp = metrics.roc_auc_score(y_valid, X_valid[:,i:i+1]) #probs

                features_auc_index.append(i)
                features_auc.append(features_imp)

            # Sort features by AUC score
            sorted_features = sorted(features_auc)
            sorted_features_index = np.argsort(features_auc)

            top_N_largest_indices = sorted_features_index[-nb:]
            X_train_selected = X_train[:,top_N_largest_indices]
            X_valid_selected = X_valid[:,top_N_largest_indices]

        else:
            best_estimatorFS= hyperparam_tuning(model, param_grid, X_train, y_train, inner_cv)[0]

            # select the top nb_features (nb) largest indices
            top_N_largest_indices = sorted_indices[-nb:]
            # print('len top_N_largest_indices',len(top_N_largest_indices))
            X_train_selected = X_train[:,top_N_largest_indices]
            X_valid_selected = X_valid[:,top_N_largest_indices]


        print('best_estimatorFS in runFS',best_estimatorFS)
        #Evaluate the model with the selected features
        #to choose the right number of features to get the best AUC
        best_estimatorFS.fit(X_train_selected, y_train)
        y_scores = best_estimatorFS.predict_proba(X_valid_selected)[:,1]
        y_scores = np.array(y_scores).astype(float)

        auc = metrics.roc_auc_score(y_valid,y_scores)
        auc_perSelection.append(auc)

        f1_scores_FS = metrics.f1_score(y_valid,best_estimatorFS.predict(X_valid_selected), average='macro')
        if auc >= auc0 :
            if f1_scores_FS > f1_scoresFS_init:
                f1_scoresFS_init = f1_scores_FS
                auc0 = auc

                nb_selected_features = nb
                X_trainSelected_final = X_train_selected
                X_validSelected_final = X_valid_selected
                top_N_largest_indices_final = top_N_largest_indices

        # print('nb_selected_features in runFS',nb_selected_features)
        print('top_N_largest_indices in runFS',top_N_largest_indices_final)

    mf.write_files(filename,listHeader,auc_perSelection)
    return nb_selected_features, X_trainSelected_final, X_validSelected_final, top_N_largest_indices_final, top_40_largest_indices


def hyperparam_tuning(model, param_grid, X_train, y_train, inner_cv):
    '''
    Looking for the best set of hyperparameter for the model used from the grid of hyperparameters.
    Input: model: model from the method list, param_grid: hyperparameters grid, X_train: Data, y_train: Target
    Output: best_estimator: model with the best hyperparameters
    '''

    # Hyperparameter tuning with GridSearchCV if the grid is not too big,
    # otherwise use RandomizedSearchCV - faster
    # if model == 'GaussianMixture()':
    #     scorer = metrics.make_scorer(metrics.accuracy_score)
    # else:
    scorer = metrics.make_scorer(metrics.roc_auc_score)

    if len(param_grid) > 1:
        # print('use RamdomizedSearchCV')
        method = 'RandomizedSearchCV'
        hyperparam_tuning = RandomizedSearchCV(estimator=model,
                            param_distributions=param_grid,
                            n_iter=10,
                            scoring=scorer,
                            n_jobs=-1,
                            cv=inner_cv,
                            verbose=0,
                            refit=True,
                            error_score='raise')
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

def run_innerLoop(modelPM,X_trainMid,y_trainMid,param_gridPM,inner_cv,top_N_features_idx,filename,fold,nb_loop):
    '''
    Inner Loop for the cross validation method.

    2) Hyperparameter tuning with the methodPM (model for the predictive model)
    Input : X_trainMid: training Dataset with the selected features only,
            y_trainMid: Target, param_gridPM: hyperparameters grid, inner_cv: cross validation method

    '''

    # Init lists for the evaluation
    predYT = [] # all predictions
    y_trueList = [] # all correct answers
    scoresList = [] # all probabilities of the predictions

    bestM_filename = filename.split('.')[0]+'_bestInnerModel'+'_NULL'+'.pkl'
    acc0_in,f1_scoresIn_init = 0.5,0
    file_toDel = 0

    filenameIn = 'Results/TestAUC/PbFea4/AUC_innerloop.csv'
    header = ['model PM','Fold-subfold']
    auc_row = [f'{modelPM}',nb_loop]

    for subfold, (train_In_idx, valid_In_idx) in enumerate(inner_cv.split(X_trainMid,y_trainMid)):
        print(f'________Inner Loop {subfold}_________')
        header.append(f'AUC {subfold}')

        print('valid_idx',valid_In_idx)
        print('train_idx',train_In_idx.shape)

        # Split your data
        X_trainIn, X_validIn = X_trainMid[train_In_idx], X_trainMid[valid_In_idx]
        y_trainIn, y_validIn = y_trainMid[train_In_idx], y_trainMid[valid_In_idx]

        ## Hyperparameter tuning for the predictive model (PM) ##
        best_estimator = hyperparam_tuning(modelPM, param_gridPM, X_trainIn, y_trainIn, inner_cv)[0]
        print('best_estimator for PM:', best_estimator)

        # ## Evaluation of the model with the validation set ##
        best_estimator.fit(X_trainIn, y_trainIn)

        y_pred = best_estimator.predict(X_validIn)
        y_scores = best_estimator.predict_proba(X_validIn)[:,1]
        y_scores = np.array(y_scores).astype(float)

        # Save predictions in the lists
        for i in range(len(y_pred)):
            predYT.append(y_pred[i])
            y_trueList.append(y_validIn[i]) #y_validVal[i]
            scoresList.append(y_scores[i])

        ## Save the best model according to the AUC ##
        aucIn = metrics.roc_auc_score(y_validIn,y_scores) #y_validVal
        auc_row.append(aucIn)

        f1_scores_In = metrics.f1_score(y_validIn,y_pred, average='macro')
        if aucIn >= acc0_in :
            if f1_scores_In > f1_scoresIn_init:
                acc0_in = aucIn
                f1_scoresIn_init = f1_scores_In
                # Save the model into "byte stream"
                if file_toDel != 0:
                    mf.delete_file(bestM_filename)
                bestM_filename = filename.split('.')[0]+'_bestModel_Inner'+f'_{fold}-{subfold}'+'.pkl'

                bestIn_estimator = best_estimator
                pickle.dump(best_estimator, open(bestM_filename, 'wb'))
                file_toDel = 1

        # see if bestM_filename exists and save the current model if not
        if not os.path.exists(bestM_filename):
            bestM_filename.replace('_bestModel_Inner', '_NoBest_Inner')
            pickle.dump(best_estimator, open(bestM_filename, 'wb'))

    mf.write_files(filenameIn,header,auc_row)
    ## Evaluation of the NsubFolds of the Inner Loop and save the results ##
    column_name,list_eval = mf.evaluation(y_trueList,predYT,scoresList)[1:]
    column_name.insert(0,'Nb Features Selected')
    list_eval.insert(0,len(top_N_features_idx))
    mf.write_files(filename,column_name,list_eval)

    return bestM_filename, bestIn_estimator

def run_middleLoop(methodFS,methodPM, filename,X,y ,testOut_idx,fold,df_predict,seed0):
    '''
    The purpose of the middle loop is to select the best features of the dataset for
    the prediction.


    This loop is splitted in 2 parts:

    1) Feature selection with the methodFS (model for the feature selection)
    2) Call the inner loop with the training set of the middle

    The training data from the folds of the outer loop are splitted in NsubFolds.
    The best model of the inner loop is kept according to the AUC.

    All the models runned in the inner loop are evaluated and the result is saved in a csv file.

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
    print('X shape',X.shape)
    print('y shape',y.shape)

    # X_excluded = X[:40,:]
    # y_excluded = y[:40]

    X_remaining = X[:,:]
    y_remaining = y[:]

    # Split the data in NsubFolds
    Nsubfolds = 10
    inner_cv = StratifiedKFold(n_splits=Nsubfolds, shuffle=True, random_state=seed0) # should I put a seed here too (if yes, the same as in outer_cv)?

    # Init lists for the evaluation
    predYT = [] # all predictions
    y_trueList = [] # all correct answers
    scoresList = [] # all probabilities of the predictions

    # Init variables for the best model
    idx = 0
    acc0,f1_scoresMid_init = 0.5,0
    file_toDel = 0
    bestMM_filename = filename.split('.')[0]+'_bestMModel'+'_NULL'+'.pkl'
    top_features_inner = []
    best_nb_features = 0
    best_top40 = []

    best_predict_proba=['NA']*74 ## 74 is the total number of data in the dataset
    # Loop
    valid_idx_list = []
    list_data_idx = []

    filenameMid ='Results/TestAUC/PbFea4/AUC_middleloop.csv'
    header = ['model FS_PM','Fold O']
    auc_midRow = [f'{methodFS}_{methodPM}',fold]
    for subfold, (train_idx, valid_idx) in enumerate(inner_cv.split(X_remaining,y_remaining)):
        print(f'________Middle Loop {subfold}_________')

        header.append(f'AUC {subfold}')
        nb_loop = f'{fold}-{subfold}'


        idx += 1
        # print the day and hour
        print('run began at',time.localtime())
        print('testOut_idx_',testOut_idx)
        print('valid_idx',valid_idx)
        print('train_idx',train_idx.shape)
        valid_idx_list.append(valid_idx)

        # Split data
        X_train, X_valid = X_remaining[train_idx], X_remaining[valid_idx]
        y_train, y_valid = y_remaining[train_idx], y_remaining[valid_idx]

        # 2nd: run the FS model with the best hyperparameters
        NbFeatures, X_train_selected,X_valid_selected, top_features_idx, top_40 = runFS(methodPM,modelFS,X_train, X_valid,y_train,y_valid,param_gridFS,inner_cv,nb_loop)
        print('top_features_idx in inner loop',top_features_idx)

        # Hyperparameter tuning for the predictive model (PM)
        bestInnerM_filename,bestIn_estimator = run_innerLoop(modelPM,X_train_selected,y_train,param_gridPM,inner_cv,top_features_idx,filename,subfold,nb_loop)

        # Test the best model from inner loop
        best_innerModel = pickle.load(open(bestInnerM_filename,'rb'))
        # bestIn_estimator.fit(X_train_selected, y_train) # Is it right to train the model again?

        y_scores = bestIn_estimator.predict_proba(X_valid_selected)[:,1]
        y_pred = bestIn_estimator.predict(X_valid_selected)

        auc = metrics.roc_auc_score(y_valid,y_scores)
        auc_midRow.append(auc)


        # Save predictions in the lists for total evaluation
        for i in range(len(y_pred)):
            predYT.append(y_pred[i])
            y_trueList.append(y_valid[i])
            scoresList.append(y_scores[i])

        f1_scores_mid = metrics.f1_score(y_valid,y_pred, average='macro')
        ## Save the best model according to the AUC ##
        if auc > acc0 :
            if f1_scores_mid > f1_scoresMid_init:
                acc0 = auc
                f1_scoresMid_init = f1_scores_mid
                # Save the model into "byte stream"
                if file_toDel != 0:
                    mf.delete_file(bestMM_filename)
                bestMM_filename = filename.split('.')[0]+'_bestModel_Middle'+f'_{fold}-{subfold}'+'.pkl'
                pickle.dump(best_innerModel, open(bestMM_filename, 'wb'))

                bestMid_estimator = bestIn_estimator

                top_features_inner = top_features_idx
                best_nb_features = NbFeatures
                best_top40 = top_40
                file_toDel = 1

        # see if bestMM_filename exists and save the current model if not
        if not os.path.exists(bestMM_filename):
            bestMM_filename.replace('_bestModel_Middle', '_NoBest_Middle')
            pickle.dump(best_innerModel, open(bestMM_filename, 'wb'))

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

            best_predict_proba[good_idx] = np.round(y_scores[i],4)

    print('LEN auc_midRow',len(auc_midRow))
    mf.write_files(filenameMid,header,auc_midRow)

    # Save predicted probabilities of the inner loop in a csv file
    mount_point =filename.split('/')[0]+'/'+filename.split('/')[1]
    if not os.path.exists(os.path.dirname(mount_point+'/out_valid/')):
        os.makedirs(os.path.dirname(mount_point+'/out_valid/'))
    prediction_filename = f"{mount_point}/out_valid/"+f'{methodFS}_{methodPM}.csv'
    df_predict.insert(fold-1,None,best_predict_proba)
    df_predict.to_csv(prediction_filename, index=True)


    # return the average evaluation of the subfolds
    auc_valid = round(metrics.roc_auc_score(y_trueList,scoresList),3)
    f1_valid = round(metrics.f1_score(y_trueList,predYT, average='macro'),3)


    return predYT, y_trueList, bestMM_filename,best_nb_features,top_features_inner, best_top40, auc_valid, f1_valid, bestMid_estimator


def OuterLoop(X, y,methodFS, methodPM, innerL_filename, outerL_filename,folder_output):
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
    seed0 = 2024
    seed0 = np.random.seed(seed0)

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

    mid_auc = []
    mid_f1 = []

    best_nb_features = 0
    top_features_outer = []
    best_top40 = []
    acc0,f1_scoresOut_init=0.5,0
    fileToDel = 0

    df_predict_inner = pd.DataFrame() # dataframe to save the predictions proba of the inner loop
    best_predict_proba=['NA']* np.shape(X)[0] ## 74 is the total number of data in the dataset

    # evaluation list for final mean
    accuracy = []
    specificity = []
    recall_case = []
    recall_control = []
    precision_case = []
    precision_control = []
    auc_outer = []
    f1_outer = []

    for fold, (train_idx, test_idx) in enumerate(folds_CVT.split(X_remaining,y_remaining)):
        print(f"================Fold {fold+1}================")

        # Split data
        X_train, X_test = X_remaining[train_idx], X_remaining[test_idx]
        y_train, y_test = y_remaining[train_idx], y_remaining[test_idx]

        print('X_remaining shape',X_remaining.shape)
        print('train_idx',train_idx)
        print('X_train shape',X_train.shape)

        predictInL, correctPred_InL, bestInnerM_filename, NbFeatures, top_features_idx, top_40, auc_validation, f1_validation,bestMid_estimator = run_middleLoop(methodFS,methodPM, innerL_filename,X_train,y_train,test_idx,fold+1,df_predict_inner,seed0)

        mid_auc.append(auc_validation)
        mid_f1.append(f1_validation)

        # Add the excluded top 40 data back into the training set
        X_train = np.concatenate(( X_excluded, X_train), axis=0)
        y_train = np.concatenate((y_excluded,y_train ), axis=0)

        # Test the best model from Middle loop
        best_innerModel = pickle.load(open(bestInnerM_filename,'rb'))
        bestMid_estimator.fit(X_train[:,top_features_idx], y_train)
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
        auc = metrics.roc_auc_score(y_test,y_Fscores)
        f1_scores_out = metrics.f1_score(y_test,y_Fpred, average='macro')
        if auc > acc0:
            if f1_scores_out > f1_scoresOut_init:
                if fileToDel != 0:
                    mf.delete_file(bestOuterM_filename)
                    mf.delete_file(bestOuterM_eval)
                acc0 = auc
                f1_scoresOut_init = f1_scores_out
                # Save the model into "byte stream"
                bestOuterM_filename = outerL_filename.split('.')[0]+'_bestModelOuter'+f'_{fold+1}'+'.pkl'
                pickle.dump(best_innerModel, open(bestOuterM_filename, 'wb'))

                # evaluation of the best model
                column_name,list_eval = mf.evaluation(y_test,y_Fpred,y_Fscores)[1:]
                bestOuterM_eval = bestOuterM_filename.split('.')[0]+'_Evaluation.csv'
                mf.write_files(bestOuterM_eval,column_name,list_eval)
                top_features_outer = top_features_idx
                best_nb_features = NbFeatures
                best_top40 = top_40
                fileToDel = 1

                print('best Outer model current:', bestOuterM_filename)

        for idx, data_idx in enumerate(test_idx):
                best_predict_proba[data_idx+40] = np.round(y_Fscores[idx],4)

        # Final evaluation without list with all the values
        confusMat= metrics.confusion_matrix(y_test,y_Fpred)
        # Check the shape to make sure it's a 2x2 matrix
        if confusMat.shape != (2, 2):
            if y_test[0]==0:
                # Manually construct a 2x2 matrix
                confusMat = np.array([[confusMat[0][0], 0], [0, 0]])
            else:
                confusMat = np.array([[0, 0], [0, confusMat[0][0]]])

        tn,fp,fn,tp= confusMat.ravel()

        ones = np.ones(len(y_test))

        accuracy.append(round(metrics.accuracy_score(y_test,y_Fpred)*100,3))
        precision_case.append(round(metrics.precision_score(y_test,y_Fpred,zero_division="warn"),3))
        precision_control.append(round(metrics.precision_score(ones-y_test,ones-y_Fpred,zero_division="warn"),3))
        recall_case.append(round(metrics.recall_score(y_test,y_Fpred),3))
        recall_control.append(round(metrics.recall_score(ones-y_test,ones-y_Fpred),3))
        specificity.append(round(tn/(tn+fp),3))
        auc_outer.append(auc)
        f1_outer.append(round(metrics.f1_score(y_test,y_Fpred, average='macro'),3))

    # Create  csv file to compare with "Performance48.csv"
    header_eval=['Model FS_PM','Nb Features Selected','Accuracy','Precision Case','Precision Control','Recall Case','Recall Control','Specificity','F1','AUC']
    row_eval=[f'{methodFS}_{methodPM}',NbFeatures,mf.mean(accuracy),mf.mean(precision_case),mf.mean(precision_control),mf.mean(recall_case),mf.mean(recall_control),mf.mean(specificity),mf.mean(f1_outer),mf.mean(auc_outer)]
    mf.save_performance(row_eval, header_eval, 'Results/TestAUC/PbFea4/Performances48_evalfromOut.csv')

    print(f'shape of y_trueList {np.shape(y_trueList)}, shape of y_predicts {np.shape(y_predicts)}, shape of y_scoresList {np.shape(y_scoresList)}')
    auc_test = round(metrics.roc_auc_score(y_trueList,y_scoresList),3)
    auc_train = round(metrics.roc_auc_score(y_trainList,scores_trainList),3)
    f1_test = round(metrics.f1_score(y_trueList,y_predicts, average='macro'),3)
    f1_train = round(metrics.f1_score(y_trainList,y_predictsTrain, average='macro'),3)

    column_nameTrain = ['Model FS_PM' ,'AUC train (O)','AUC test (O)','AUC validation (M)','F1 train (O)','F1 test (O)','F1 validation (M)']
    list_evalTrain = [f'{methodFS}_{methodPM}',auc_train,auc_test,sum(mid_auc)/len(mid_auc),f1_train,f1_test, sum(mid_f1)/len(mid_f1)]
    mf.write_files('Results/TestAUC/PbFea4/GlobEvaluation.csv',column_nameTrain,list_evalTrain)

    # Save predictions of the inner loop in a csv file
    prediction_filename = innerL_filename.split('.')[0]+f'_Innerpredictions{fold}'+'.csv'
    df_predict = pd.DataFrame({'Actual':correctPred_InL , 'Predicted': predictInL})
    df_predict.to_csv(prediction_filename, index=False)

    # Evaluation of outer loop models and save results in files
    print("********Final**********")
    column_name,list_eval = mf.evaluation(y_trueList,y_predicts,y_scoresList)[1:]
    column_name.insert(0,'Nb Features Selected')
    list_eval.insert(0,best_nb_features)
    mf.write_files(outerL_filename,column_name,list_eval)

    #Save evaluation in a csv file for each model MethodPM_MethodFS that is runned in outerloop and add to the previous data
    # Add in the first index the name of the model
    list_eval.insert(0,f'{methodFS}_{methodPM}')
    column_name.insert(0,'Model FS_PM')
    performance_filename = 'Results/TestAUC/PbFea4/Performances48.csv'
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

    return top_features_outer, best_nb_features, best_top40
