
# This file is used to run the entire project
#################################################
##        Contributor: Jeanne CLARET           ##
##                                             ##
##            TMJOA_PROGNOSIS                  ##
##                                             ##
#################################################

import pandas as pd
import os
# from sklearn.datasets import load_breast_cancer

import modelFunctions as mf
import Step1 as st1
import modelFunctions as mf
# List of equivalent models from R
methods_list = ["glmnet", "svmLinear", "rf", "xgbTree", "lda2", "nnet", "glmboost", "hdda"]
methods_FS = ["glmnet","rf","xgbTree","lda2","nnet","glmboost","AUC"]

vecT = [(i, j) for i in range(0, 8) for j in range(0,7) ]
print('vecT:',vecT)
A = pd.read_csv("./TMJOAI_Long_040422_Norm.csv")

y = A.iloc[:, 0].values
X = A.iloc[:, 1:].values

# X, y = load_breast_cancer(return_X_y=True)
# # X= X[:120]
# # y= y[:120]

# Write the folder in which you want your output to be saved without '/' at the end
folder_output='Predictions/ReviewMiddle_2024'
folder_output+='/'

folder_results = folder_output+'Results/'
if not os.path.exists(os.path.dirname(folder_results)):
    os.makedirs(os.path.dirname(folder_results))

for iii in range(0, len(vecT)):
    i_PM = vecT[iii][0]
    i_FS = vecT[iii][1]


    print(f'====== FS with {methods_FS[i_FS]} ======')
    print(f'________ Model trained - {methods_list[i_PM]} ________')

    # Init files for results
    innerL_filename = f"{folder_output}{methods_FS[i_FS]}_{methods_list[i_PM]}/scores_{methods_list[i_PM]}_InnerLoop.csv"
    outerL_filename = f"{folder_output}{methods_FS[i_FS]}_{methods_list[i_PM]}/result_{methods_list[i_PM]}_OuterLoop.csv"
    if not os.path.exists(os.path.dirname(innerL_filename)):
        os.makedirs(os.path.dirname(innerL_filename))

    # Remove files if they exist
    mf.delete_file(innerL_filename)
    mf.delete_file(outerL_filename)


    top_features_idx,nb_features, best40FS = st1.OuterLoop(X, y, methods_FS[i_FS], methods_list[i_PM], innerL_filename, outerL_filename, folder_output)

    print('top_features_idx in main:',top_features_idx)
    # Save in a file the top features
    data = [f'{methods_FS[i_FS]}_{methods_list[i_PM]}', nb_features]
    for i in range(len(top_features_idx)):
        data.append(f'{A.columns[top_features_idx[i]+1]}')
    #csv file to save the top features and the model who used them
    first_row = ['model FS_PM','Nb features', 'top feature 1', 'top feature 2', 'top feature 3', 'top feature 4', 'top feature 5', 'top feature 6', 'top feature 7', 'top feature 8', 'top feature 9', 'top feature 10','top feature 11','top feature 12','top feature 13','top feature 14','top feature 15','top feature 16','top feature 17','top feature 18','top feature 19','top feature 20','top feature 21','top feature 22','top feature 23','top feature 24','top feature 25','top feature 26','top feature 27','top feature 28','top feature 29','top feature 30','top feature 31','top feature 32','top feature 33','top feature 34','top feature 35','top feature 36','top feature 37','top feature 38','top feature 39','top feature 40']

    mf.write_files(f"{folder_output}topFeatures.csv", first_row, data)
