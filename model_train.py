#!/usr/local/anaconda/bin/python
import sys
import argparse
import random
import json

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from xgboost import Booster

from datetime import datetime

def load_data(filename):
    print("Loading dataset... ", end="", flush=True)
    dataDF = pd.read_csv(filename)
    print("done.")
    print("The dataset contains {:,} observations and {} attributes.".format(dataDF.shape[0], dataDF.shape[1]))
    return dataDF

def get_train_test(dataDF, smote=True, seed=1234, test_size=0.2):
    
    trainDF, testDF = train_test_split(dataDF, test_size=test_size, random_state=seed, stratify=dataDF[["Class"]])
    
    print("Processing training set...")
    X_train, y_train = feature_eng(trainDF, smote, seed)
        
    print("Processing test set...")
    X_test, y_test = feature_eng(testDF, smote=False, seed=seed)

    return X_train, y_train, X_test, y_test


def feature_eng(dataDF, smote=True, seed=1234):

    data_fe_DF = dataDF.copy()
    
    # Convert Time to hours
    data_fe_DF["Hour"] = dataDF["Time"].apply(datetime.fromtimestamp).dt.hour
    data_fe_DF = data_fe_DF.drop(["Time"], axis=1)
    
    # Zero-mean center 
    data_fe_DF["Amount"] = data_fe_DF["Amount"].subtract(data_fe_DF["Amount"].mean())
    data_fe_DF["Hour"] = data_fe_DF["Hour"].subtract(data_fe_DF["Hour"].mean())
    
    X = data_fe_DF.iloc[:, data_fe_DF.columns != "Class"]
    y = data_fe_DF.iloc[:, data_fe_DF.columns == "Class"]
    
    if smote:
        # Oversampling
        print("Oversampling...")
        X, y = SMOTE(random_state=seed).fit_resample(X, y)
        
    value_counts = y["Class"].value_counts()
    print("Fraudulent transactions are {:.2f}% of the set.".format(value_counts[1] * 100 / (value_counts[0] + value_counts[1])))
    
    return X, y

def xgboost_search(X, y, search_verbose=1):
    
    params = {
        "gamma":[0.5, 1, 1.5, 2, 5],
        "max_depth":[3,4,5,6],
        "min_child_weight": [100],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "learning_rate": [0.1, 0.01, 0.001]
    }
    
    xgb = XGBClassifier(objective="binary:logistic", eval_metric="auc", use_label_encoder=False)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1234)
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=params, scoring="roc_auc", n_jobs=1, cv=skf.split(X,y), verbose=search_verbose)

    grid_search.fit(X, y)

    print("Best estimator: ")
    print(grid_search.best_estimator_)
    print("Parameters: ", grid_search.best_params_) 
    print("Highest AUC: %.2f" % grid_search.best_score_)
    
    return grid_search.best_params_

def grid_search(X, y, search_verbose, sample_size, seed=1234):
    
    if sample_size is not None:
        if sample_size > X.shape[0]:
            raise ValueError("Number of samples specified for the grid search can't be larger the the size of the dataset. nrows_search = {}. Dataset size = {}".format(sample_size, X.shape[0]))
        
        random.seed(seed)
        rows = random.sample(np.arange(0,len(X.index)).tolist(), sample_size)
        X = X.iloc[rows,]
        y = y.iloc[rows,]

    model_params = xgboost_search(X, y, search_verbose)
    
    return model_params

def train_model(X, y, params):
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", use_label_encoder=False)
    model.set_params(**params)
    model.fit(X, y)
    return model

def roc_curve(fp_r, tp_r, auc, roc_curve_filename="./model/roc_curve.png"):

    plt.figure(figsize=(8, 6))

    plt.plot(fp_r, tp_r, label = "AUC = {:.2f}".format(auc))
    plt.plot([0,1],[0,1],"r--")

    plt.ylabel("TP rate")
    plt.xlabel("FP rate")

    plt.legend(loc=4)
    plt.title("ROC Curve")
    plt.savefig(roc_curve_filename)
    
def conf_matrix(y_pred, y_test, threshold, filename="./model/conf_matrix.png"):
    
    plt.figure(figsize=(5,5))
    y_pred_int = (y_pred > threshold).astype(int)
    c_matrix = metrics.confusion_matrix(y_test, y_pred_int)
    sns.heatmap(c_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.suptitle("T={:.1f}".format(threshold))
    plt.savefig(filename)
    
    
def evaluate(model, X_test, y_test):
    
    y_pred = model.predict_proba(X_test)[:,1]
    fp_r, tp_r, t = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fp_r, tp_r)
    print("Final model AUC: {:.2f}".format(auc))
    
    print("Plotting ROC curve")
    roc_curve(fp_r, tp_r, auc)
        
    t_opt_idx = np.argmax(tp_r - fp_r)
    t_opt = t[t_opt_idx]
    print("Optimal threshold value is: {:.2f}".format(t_opt))
    
    print("Plotting confusion matrix...")
    conf_matrix(y_pred, y_test, t_opt)
    
    return auc
    
def main(args): 
 
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier using the Credit Card Fraud Data Set. \
                                     This work is licensed \
                                     under the Creative Commons Attribution \
                                     4.0 International License.")
    
    parser.add_argument("--nrows_search", help="Number of samples used in the grid search. If not set the entire training set is used.", 
                        required=False, default=5000, type=int)
    parser.add_argument("--gridsearch_verbose", help="Verbosity of the gridsearch.", required=False, 
                        default=1, type=int, choices=[0,1,2])
    
    args = parser.parse_args()
    
    dataDF = load_data("./dataset/creditcard.csv")
    X_train, y_train, X_test, y_test = get_train_test(dataDF)
    best_params = grid_search(X_train, y_train, args.gridsearch_verbose, args.nrows_search)
    
    print("Training final model...")
    model = train_model(X_train, y_train, best_params)
    #model = XGBClassifier()
    #booster = Booster()
    #booster.load_model('./model/smote_fraud.xgb.bak')
    #model._Booster = booster
    
    auc = evaluate(model, X_test, y_test)

    print("Saving the model...")
    model.save_model("./model/smote_fraud.xgb")
    
    with open('dominostats.json', 'w') as f:
      f.write(json.dumps({"AUC score": auc}))
 
if __name__=='__main__':
    main(sys.argv)