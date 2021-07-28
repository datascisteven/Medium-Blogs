from collections import Counter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, auc, average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns


def auc(X, y, model):
    probs = model.predict_proba(X)[:,1] 
    return roc_auc_score(y, probs)


def aps(X, y, model):
    probs = model.predict_proba(X)[:,1]
    return average_precision_score(y, probs)


def get_metrics(X, y, y_pred, model):
    """
        Function to calculate the following metrics for evaluating the model:
        Accuracy, F1, ROC-AUC, Recall, Precision, and PR-AUC Scores

        Need to enter X_valid, y_valid, y_pred, and model
    """
    ac_val = accuracy_score(y, y_pred)
    f1_val = f1_score(y, y_pred)
    au_val = auc(X, y, model)
    rc_val = recall_score(y, y_pred)
    pr_val = precision_score(y, y_pred)
    aps_val = aps(X, y, model)

    print('Accuracy Score: ', ac_val)
    print('F1 Score: ', f1_val)
    print('ROC-AUC Score: ', au_val)
    print('Recall Score: ', rc_val)
    print('Precision Score: ', pr_val)
    print('PR-AUC Score: ', aps_val)


def run_resampling(X_train, y_train, X_valid, y_valid, resampling_method, model):
    """
        Function to run resampling method on training set to produce balanced dataset, 
        to show the count of the majority and minority class of resampled data,
        to train provided model on training data and evaluate metrics on validation data

        Need to enter X_train, y_train, X_valid, y_valid, resampling_method, and model
    """
    X_train_resampled, y_train_resampled = resampling_method.fit_resample(X_train, y_train)
    print("Training Count: ", Counter(y_train_resampled))
    new_model = model.fit(X_train_resampled, y_train_resampled)
    y_pred = new_model.predict(X_valid)
    get_metrics(X_valid, y_valid, y_pred, new_model)


def group_list(lst, size=100):
    """
    Generate batches of 100 ids in each
    Returns list of strings with , seperated ids
    """
    new_list =[]
    idx = 0
    while idx < len(lst):        
        new_list.append(
            ','.join([str(item) for item in lst[idx:idx+size]])
        )
        idx += size
    return new_list


def tweets_request(tweets_ids):
    """
    Make a request to Tweeter API
    """
    df_lst = []
    
    for batch in tqdm(tweets_ids):
        url = "https://api.twitter.com/2/tweets?ids={}&&tweet.fields=created_at,entities,geo,id,public_metrics,text&user.fields=description,entities,id,location,name,public_metrics,username".format(batch)
        payload={}
        headers = {'Authorization': 'Bearer ' + keys['bearer_token'],
        'Cookie': 'personalization_id="v1_hzpv7qXpjB6CteyAHDWYQQ=="; guest_id=v1%3A161498381400435837'}
        r = requests.request("GET", url, headers=headers, data=payload)
        data = r.json()
        if 'data' in data.keys():
            df_lst.append(pd.DataFrame(data['data']))
    
    return pd.concat(df_lst)



def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def train_test_metrics(y_train, y_test, y_train_pred, y_test_pred):
	print('Training R^2 Score: ', round(r2_score(y_train, y_train_pred), 4))
	print('Training RMSE: %d' % rmse(y_train, y_train_pred))
	print('Testing R^2 Score: ', round(r2_score(y_test, y_test_pred), 4))
	print('Testing RMSE: %d' % rmse(y_test, y_test_pred))
	return

def get_metrics_confusion(X, y, y_pred, model):
    """
        Function to get accuracy, F1, ROC-AUC, recall, precision, PR-AUC scores followed by confusion matrix
        where X is feature dataset, y is target dataset, and model is instantiated model variable
    """
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = auc(X, y, model)
    rec = recall_score(y, y_pred)
    prec = precision_score(y, y_pred)
    pr_auc = aps(X, y, model)

    print('Accuracy: ', acc)
    print('F1 Score: ', f1)
    print('ROC-AUC: ', roc_auc)
    print('Recall: ', rec)
    print('Precision: ', prec)
    print('PR-AUC: ', pr_auc)
    
    cnf = confusion_matrix(y, y_pred)
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf.flatten()/np.sum(cnf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cnf, annot=labels, fmt='', cmap='Blues', annot_kws={'size':14}, cbar=False, xticklabels=False, yticklabels=False)

def aps2(X, y, model):
    """
        Function to calculate PR-AUC Score based on decision_function(X)
        where X is feature values, y is target values, and model is instantiated model variable
    """
    probs = model.decision_function(X)
    return average_precision_score(y, probs)

def get_metrics_2(X, y, y_pred, model):
    """
        Function to get training and validation F1, recall, precision, PR AUC scores
        Instantiate model and pass the model into function
        Pass X_train, y_train, X_val, Y_val datasets
        Pass in calculated model.predict(X) for y_pred
    """    
    ac = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    rc = recall_score(y, y_pred)
    pr = precision_score(y, y_pred)
    prauc = aps2(X, y, model)
    
    print('Accuracy: ', ac)
    print('F1: ', f1)
    print('Recall: ', rc)
    print('Precision: ', pr)
    print('PR-AUC: ', prauc)

def get_confusion(y, y_pred):
    cnf = confusion_matrix(y, y_pred)
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cnf.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cnf.flatten()/np.sum(cnf)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cnf, annot=labels, fmt='', cmap='Blues', annot_kws={'size':14}, cbar=False, xticklabels=False, yticklabels=False)