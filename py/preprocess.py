import re
import sys
sys.path.append("../py")
import gensim
from nltk.stem import WordNetLemmatizer


def remove_users(df, col):  
    df[col] = df[col].apply(lambda x: re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(x))) # remove re-tweet
    df[col] = df[col].apply(lambda x: re.sub(r'(@[A-Za-z0-9-_]+)', '', str(x))) # remove tweeted at
    return df

def remove_special_characters(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'&[\S]+?;', '', str(x))) # remove character references
    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', r'', str(x))) # remove punctuation
    df[col] = df[col].apply(lambda x: re.sub(r'#', ' ', str(x)))
    return df

def remove_links(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'http\S+', '', str(x)))  # remove http links
    df[col] = df[col].apply(lambda x: re.sub(r'bit.ly/\S+', '', str(x)))  # remove bit.ly links
    return df

def remove_numerics(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'\w*\d\w*', r'', str(x)))
    return df

def remove_whitespaces(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'\s\s+', ' ', str(x))) 
    df[col] = df[col].apply(lambda x: re.sub(r'(\A\s+|\s+\Z)', '', str(x)))
    return df

def lemmatize(token):
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(tweet):
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:  # drops stopwords and words with <3 characters
            result.append(lemmatize(token))
    res = ' '.join(result)
    return res

def tokenize_and_lemmatize(df, col):
    df[col] = df[col].apply(lambda x: tokenize(x))
    return df

def preprocess_tweets(df, col):
    remove_users(df, col)
    remove_special_characters(df, col)
    remove_links(df, col)
    remove_numerics(df, col)
    remove_whitespaces(df, col)
    tokenize_and_lemmatize(df, col)
    return df

def preprocess(tweet):
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'#', ' ', result)
    result = re.sub(r'[^\w\s]', r'', result)    
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)
    res = tokenize(result)
    return res 