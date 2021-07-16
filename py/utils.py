from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, average_precision_score, roc_auc_score
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, auc, average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import re
import pandas as pd
import requests
from config import keys
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


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

def remove_users(df, col):  
    df[col] = df[col].apply(lambda x: re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(x))) # remove re-tweet
    df[col] = df[col].apply(lambda x: re.sub(r'(@[A-Za-z0-9-_]+)', '', str(x))) # remove tweeted at

def remove_special_characters(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'&[\S]+?;', '', str(x))) # remove character references
    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', r'', str(x))) # remove punctuation

def remove_hash(df, col):
	df[col] = df[col].apply(lambda x: re.sub(r'#', ' ', str(x)))

def remove_links(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'http\S+', '', str(x)))  # remove http links
    df[col] = df[col].apply(lambda x: re.sub(r'bit.ly/\S+', '', str(x)))  # remove bit.ly links  

def remove_numerics(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'\w*\d\w*', r'', str(x)))

def remove_whitespaces(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'\s\s+', ' ', str(x))) 
    df[col] = df[col].apply(lambda x: re.sub(r'(\A\s+|\s+\Z)', '', str(x)))

def lemmatize(token):
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(tweet):
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:  # drops stopwords and words with <3 characters
            result.append(lemmatize(token))
    return result

def tokenize_and_lemmatize(df, col):
    df[col] = df[col].apply(lambda x: tokenize(x))

def preprocess_tweets(df, col):
    remove_users(df, col)
    remove_special_characters(df, col)
    remove_hash(df, col)
    remove_links(df, col)
    remove_numerics(df, col)
    remove_whitespaces(df, col)
    tokenize_and_lemmatize(df, col)
