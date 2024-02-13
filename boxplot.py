import csv
#COunt the number of class 0 and 1 in the dataset
# filename = './TMJOAI_Long_040422_Norm.csv'
# class_0_count = 0
# class_1_count = 0

# with open(filename, 'r') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip the header row
#     for row in reader:
#         if row[0] == '0':
#             class_0_count += 1
#         elif row[0] == '1':
#             class_1_count += 1

# print(f"Number of class 0: {class_0_count}")
# print(f"Number of class 1: {class_1_count}")
import pandas as pd
import matplotlib.pyplot as plt

input_file = './Predictions/40in_FScv_saveIDX_2023/Results/Performances48_2023.csv'
data_features = pd.read_csv(input_file)  # Read only the header
feature_names = data_features.columns[1:]  # Exclude the label column

methods_list = ["glmnet", "svmLinear", "rf", "xgbTree", "lda2", "nnet", "glmboost", "hdda"]
methods_FS = ["glmnet","rf","xgbTree","lda2","nnet","glmboost","AUC"]

total_boxplot = []
# read the F1 score of every row starting with methods_Fs
for method_FS in methods_list:
    print('method_FS:',method_FS)
    data = pd.read_csv(input_file)
    # find every combination of method FS_PM starting with method_FS in the column

    all_row = data[data['Model FS_PM'].str.endswith(method_FS)]['Model FS_PM'].unique()
    F1_score =[]
    for name in all_row:
        F1_score.append(data[data['Model FS_PM']==name]['AUC'].values[0])
    # Box plot of the F1 score
    total_boxplot.append(F1_score)

fig = plt.figure(figsize =(10, 7))
#median line in dark, point in blue and larger median line
plt.boxplot(total_boxplot,patch_artist=True,boxprops=dict(facecolor='lightyellow'),medianprops=dict(color='black',markersize=15),flierprops=dict(marker='o', markerfacecolor='blue'))
# limit y axis from 0.35 to0.65
plt.ylim(0.35,0.80)
# add name of the methods on the x axis
plt.xticks([i for i in range(1,9)],methods_list)
plt.show()
