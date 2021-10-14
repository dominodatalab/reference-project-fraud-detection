# Pre-canned solution: Credit Card Fraud Detection

Credit card fraud represents a significant problem for financial institutions, and reliable fraud detection is generally challenging.
This project can be used as a template, facilitating the training of a machine learning model on a real-world credit card fraud dataset.
It also employs techniques like oversampling and threshold moving to address class imbalance.

The dataset used in this project has been collected as part of a research collaboration between Worldline and the Machine Learning Group of Université Libre de Bruxelles, and the raw data can be freely downloaded from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).


The assets included in the project are:

* **FraudDetection.ipynb** - a notebook that performs exploratory data analysis, data wrangling, hyperparameter optimisation, model training and evaluation. The notebook introduces the usecases and discusses the key techniques needed for implementing a classification model (e.g. oversampling, threshold moving etc.)

* **model_train.py** - a training script that can be operationalised and retrain the model on demand or on schedule. The script can be used as a template. The key elements that need to be customized for other datasets are:

    * *load_data* - data ingestion function
    * *feature_eng* - data wrangling
    * *xgboost_search* - more specifically, the values in *params*, which define the grid search scope
    
* **model_api.py** - a scoring function that exposes the persisted model as Model API. The *score* function accepts as arguments all independent parameters of the dataset and uses the model to compute the fraud probability for the individual transaction.

**Note:** You need to unzip the *dataset/creditcard.csv.zip* file before running any of the above.

## Dockerfile

This project uses a compute environment based on dominodatalab/base:Ubuntu18_DAD_Py3.7_R3.6_20200508

Add the following entries to the Dockerfile:

```
RUN echo "ubuntu    ALL=NOPASSWD: ALL" >> /etc/sudoers
RUN pip install --upgrade pip
RUN pip install imblearn && pip install xgboost
```

## Model API

You can test the Model API using the following observation:

{
  "data": {
    "V1": -0.88, 
    "V2": 0.40, 
    "V3": 0.73, 
    "V4":-1.65, 
    "V5":2.73, 
    "V6":3.41, 
    "V7":0.23, 
    "V8":0.71, 
    "V9":-0.35, 
    "V10":-0.45,
    "V11":-0.16, 
    "V12":-0.36, 
    "V13":-0.10, 
    "V14":-0.06, 
    "V15":0.86, 
    "V16":0.83, 
    "V17":-1.28, 
    "V18":0.14, 
    "V19":-0.27, 
    "V20":0.10,
    "V21":-0.25, 
    "V22":-0.90, 
    "V23":-0.22, 
    "V24":0.98, 
    "V25":0.27,
    "V26":-0.001, 
    "V27":-0.29, 
    "V28":-0.14, 
    "Amount":-68.74, 
    "Hour":5.98
  }
}
