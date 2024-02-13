import pandas as pd
from collections import Counter

# Load the feature names from the data file
data_features_path = './TMJOAI_Long_040422_Norm.csv'
data_features = pd.read_csv(data_features_path, nrows=0)  # Read only the header
feature_names = data_features.columns[1:]  # Exclude the label column

# Function to extract numbers from strings
def extract_numbers(s):
    return [int(n) for n in s.split() if n.isdigit()]

# Load the new CSV file with the feature indices
file_path = './Predictions/40in_FScv_saveIDX_2023/Results/TopFeatures_middleloop_fromOut_2023.csv'
data = pd.read_csv(file_path)

# Initialize a counter for all feature indices
feature_counts = Counter()

# Iterate through each fold column to extract and count feature indices
# for fold_col in [col for col in data.columns if col.startswith('Fold')]:
#     print('fold_col:',fold_col)
#     for row in data[fold_col]:
#         print('row:',row)
#         indices = extract_numbers(row)
#         print('indices:',indices)
#         feature_counts.update(indices)
#extract the str of the column and remove the thing with ':', just keep the list
dict_counts = {}

for i in range(0,data.shape[0]):
    row = data.iloc[i,1:]
    for col in row:
        clean_list_indices = []
        list_indices = col.split(':')[1]
        st_indices_trimmed = list_indices.strip(" [").strip("]")

        # Step 2: Split the string on whitespace
        indices_str_list = st_indices_trimmed.split()
        for index in indices_str_list:
            clean_list_indices.append(index)
        # Step 3: Convert each string in the list to an integer
        for indice in clean_list_indices:
            # print('indice:',indice)
            if not indice in dict_counts:
                dict_counts[indice] = 1
            else:
                dict_counts[indice] +=1
print('dict_counts:',dict_counts)

sorted_dict_by_values = {k: v for k, v in sorted(dict_counts.items(), key=lambda item: item[1])}

rank_dict = {}
for index in sorted_dict_by_values.keys():
    feature_name = feature_names[int(index)]
    rank_dict[feature_name] = sorted_dict_by_values[index]

print('len dict (nb features total):',len(sorted_dict_by_values))
print('rank_dict:',rank_dict)