import os
import sklearn.metrics as mt
import numpy as np
import csv



def convertToArray(list1):
    arr = np.array(list1)
    return arr

def evaluation(y_true,y_pred,y_scores):
    """
    Evaluation of the model
    Input: y_true: Correct answer, y_pred: Predictions from the model, y_scores: Probability of the predictions
    Output: int(Accuracy) + list useful to create the csv file
    """

    #confusion matrix with sklearn
    # print('y_true labels:', y_true)
    # print('y_pred labels:', y_pred)

    confusMat= mt.confusion_matrix(y_true,y_pred)
    # Check the shape to make sure it's a 2x2 matrix
    if confusMat.shape != (2, 2):
        if y_true[0]==0:
            # Manually construct a 2x2 matrix
            confusMat = np.array([[confusMat[0][0], 0], [0, 0]])
        else:
            confusMat = np.array([[0, 0], [0, confusMat[0][0]]])

    tn,fp,fn,tp= confusMat.ravel()
    print('confusion matrix tn:', tn)
    print('confusion matrix tp:', tp)
    ones = np.ones(len(y_true))

    errors = fp+fn
    accuracy = round(mt.accuracy_score(y_true,y_pred)*100,3)
    precision_case = round(mt.precision_score(y_true,y_pred,zero_division="warn",average='macro'),3)
    precision_control = round(mt.precision_score(ones-y_true,ones-y_pred,zero_division="warn",average='macro'),3)
    recall_case = round(mt.recall_score(y_true,y_pred),3)
    recall_control = round(mt.recall_score(ones-y_true,ones-y_pred),3)
    specificity = round(tn/(tn+fp),3)
    f1 = round(mt.f1_score(y_true,y_pred, average='macro'),3) # should be the same as f1_score
    # f1_score = round((mt.f1_score(y_true,y_pred)+mt.f1_score(ones-y_true,ones-y_pred))/2,3)

    y_true = np.array(y_true).astype(int)
    y_scores = np.array(y_scores).astype(float)
    if y_scores.ndim > 1 and y_scores.shape[1] > 1:
        y_scores = y_scores[:, 1]

    auc = round(mt.roc_auc_score(y_true,y_scores),3)

    print('-----evaluation-----')
    print('Number of errors: ', errors)
    print(f'Accuracy: {accuracy} %')
    print(f'Precision score tp/(tp+fp) : {precision_case} ') #best value is 1 - worst 0
    print(f'Recall score tp/(fn+tp): {recall_case} ') # Interpretatiom: High recall score => model good at identifying positive examples
    print(f'Specificity tn/(tn+fp): {specificity} ')
    # Most important scores:
    print(f'F1 : {f1} ')
    print(f'AUC : {auc} ') # best is 1


    column_name = ['Total','Nb Errors', 'Accuracy', 'Precision Case', 'Precision Control', 'Recall Case', 'Recall Control','Specificity','F1', 'AUC']
    list_eval = [len(y_pred),errors, accuracy, precision_case, precision_control,recall_case, recall_control,specificity, f1, auc]


    return auc,column_name,list_eval

def write_files(filename,listHeader,listParam):
    """
    Write the results in a csv file
    Input: filename: Name of the file, listHeader: List of the header, listParam: List of the parameters
    Output: "<filename>.csv" file
    """
    existing_file = os.path.exists(filename)

    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        if not existing_file:
            csvwriter.writerow(listHeader)

        csvwriter.writerow(listParam)

import pandas as pd
import os

def save_performance(list_eval, column_name, filename):
    """
    Function to create a csv file with the header column_name and the row list_eval
    If the file has already 49 rows, it is deleted and a new empty file is created
    It is used to save the performance of the model
    Input: list_eval: list of the performance, column_name: list of the header, filename: name of the file
    Output: "<filename>.csv" file
    """
    # Check if the file already exists
    if os.path.isfile(filename):
        # If it does, read it
        df = pd.read_csv(filename)
        # Check if the file has 49 rows
        if len(df) == 58:
            # If it does, delete the file
            os.remove(filename)

            # And create a new empty file with the same name
            df = pd.DataFrame(columns=column_name)
            df.to_csv(filename, index=False)
    else:
        # If the file does not exist, create a new DataFrame with column names and save it
        df = pd.DataFrame(columns=column_name)
        df.to_csv(filename, index=False)

    # Convert the list_eval to a DataFrame row
    new_row = pd.DataFrame([list_eval], columns=column_name)
    new_row.to_csv(filename, mode='a', header=False, index=False)



def delete_file(filename) -> None:
    """
    Remove file named filename if it exist
    Input: Name of the file
    """
    if os.path.exists(filename) :
        os.remove(filename)

def mean(lst):
    """
    Calculate the mean of a list
    Input: list
    Output: mean of the list
    """
    return sum(lst) / len(lst)


def convert_ndarray(d):
    for key in d:
        if isinstance(d[key], np.ndarray):
            d[key] = d[key].tolist()  # Convert ndarray to list
        elif isinstance(d[key], dict):
            d[key] = convert_ndarray(d[key])  # Recurse into sub-dictionaries
    return d

def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_list(v) for v in obj]
    else:
        return obj