from xgboost import XGBClassifier
from xgboost import Booster

from datetime import datetime

import numpy as np

model = XGBClassifier()
booster = Booster()
booster.load_model('./model/smote_fraud.xgb.bak')
model._Booster = booster

def score(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12,
            V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23,
            V24, V25, V26, V27, V28, Amount, Hour):
    
    inp = np.array([V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, 
                    V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23,
                    V24, V25, V26, V27, V28, Amount, Hour]).reshape((1,-1))
    
    return model.predict_proba(inp).tolist()

    
if __name__=='__main__':
    
    x = score(-0.88, 0.40, 0.73, -1.65, 2.73, 3.41, 0.23, 0.71, -0.35, -0.45,
                -0.16, -0.36, -0.10, -0.06, 0.86, 0.83, -1.28, 0.14, -0.27, 0.10,
                -0.25, -0.90, -0.22, 0.98, 0.27,-0.001, -0.29, -0.14, -68.74, 5.98)
    
    print(x)