#Given a original df and modified df get the various ml metrics to be logged
from sklearn.metrics import accuracy_score,recall_score,f1_score
import datetime
import pandas as pd
def calucate_metrics(edited_df,pushed_df):
    #compare edited df with the original df
    #compute accuracy, recall, precision, f1 and ROC for function, level, industry
    accuracy_function = accuracy_score(edited_df['Predicted Function'],pushed_df['Predicted Function'])
    accuracy_level = accuracy_score(edited_df['Predicted Level'],pushed_df['Predicted Level'])
    accuracy_industry = accuracy_score(edited_df['Predicted Industry'],pushed_df['Predicted Industry'])
    # Calculate recall scores
    recall_function = recall_score(edited_df['Predicted Function'],pushed_df['Predicted Function'], average='weighted')
    recall_level = recall_score(edited_df['Predicted Level'],pushed_df['Predicted Level'], average='weighted')
    recall_industry = recall_score(edited_df['Predicted Industry'],pushed_df['Predicted Industry'], average='weighted')

    # Calculate F1 scores
    f1_function = f1_score(edited_df['Predicted Function'],pushed_df['Predicted Function'], average='weighted')
    f1_level = f1_score(edited_df['Predicted Level'],pushed_df['Predicted Level'], average='weighted')
    f1_industry = f1_score(edited_df['Predicted Industry'],pushed_df['Predicted Industry'], average='weighted')
    # Create a dictionary to store the metrics
    metrics = {
        'accuracy_function': accuracy_function,
        'accuracy_level': accuracy_level,
        'accuracy_industry': accuracy_industry,
        'recall_function': recall_function,
        'recall_level': recall_level,
        'recall_industry': recall_industry,
        'f1_function': f1_function,
        'f1_level': f1_level,
        'f1_industry': f1_industry
    }
    return metrics