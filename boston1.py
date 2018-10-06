
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import datetime
import os
import subprocess
import sys
import pandas as pd
from sklearn.externals import joblib

boston = load_boston()

# Fill in your Cloud Storage bucket name
BUCKET_ID = "regression_ml"

dataset = pd.DataFrame(boston.data,columns =boston.feature_names)

boston.target[:5]  

dataset['PRICE']= boston.target

X= dataset.drop('PRICE', axis=1)
y = dataset['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Export the classifier to a file
model = 'model.joblib'
joblib.dump(regressor, model)
# [END train-and-save-model]

# [START upload-model]
# Upload the saved model file to Cloud Storage
model_path = os.path.join('gs://', BUCKET_ID, datetime.datetime.now().strftime(
    'boston_%Y%m%d_%H%M%S'), model)
subprocess.check_call(['gsutil', 'cp', model, model_path], stderr=sys.stdout)
# [END upload-model]

