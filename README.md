# TMJOA_prog

The dataset is "TMJOAI_Long_040422_Norm.csv". The total length is 74 with 70 features for each patient.

## Step 1
### Run the training
You can choose how many models you want to run with the length of vecT in main.py.
1) Choose the name of the folder with the predictions, the saved models, the top features and the predicted probabilities in out/ and out_valid/ used for step 2, with "folder_output".

2) Create folders named "Results/TestAUC/" and change the name of the files by hand in Step1.py (sorry).

``` python3 main.py ```

It will take around 6 hours for the 57 machines learning. 

### Understand the output

The files in out/ and out_valid/ in the prediction folder are the predicted probabilities of the outer loop and the middle loop respectively. These probabilities are used to combine the model in step 2 (EHPN).

In the "Results" folder, we can find:
* AUC_result_xxxx.csv: Compare the AUC and F1 scores of the training, testing and validation sets
* Performances48_xxx.csv: Performance of the final model in the outer loop. Compare all combination of feature selection and predictive models.
* AUC_innerloop: AUC score of the inner loop
* AUC_featureSelection : AUC score at the end of the feature selection.
* AUC_middleloop: AUC of the predicted probabilities calculated in out_valid/
  
## Step 2 
### Run the training
Inputs files from out/ and out_valid/

Choose the folder that contains both out/ and out_valid/ and the list of models you want to ccombine in this step (3 bests from the step 1)

``` python3 Step2_EHPN.py --input 'Predictions/'  --output 'Output_EHPN/' --lst_models 'glmnet','lda2','svmLinear' ```
