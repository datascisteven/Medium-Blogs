from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, auc, average_precision_score, roc_auc_score
from collections import Counter

def auc(X, y, model):
    probs = model.predict_proba(X)[:,1] 
    return roc_auc_score(y, probs)

def aps(X, y, model):
    probs = model.predict_proba(X)[:,1]
    return average_precision_score(y, probs)

def get_metric(X, y, y_pred, model):
    """
        Function to calculate the following metrics for evaluating the model:
        Accuracy, F1, ROC-AUC, Recall, Precision, and PR-AUC Scores

        Need to enter X_valid, y_valid, y_pred, and model
    """
    ac_val= accuracy_score(y, y_pred)
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

def sampling(X_train, y_train, X_valid, y_valid, resampling_method, model):
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
    get_metric(X_valid, y_valid, y_pred, new_model)
