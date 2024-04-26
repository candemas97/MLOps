import os
import requests
import pandas as pd
import numpy as np

## download the dataset
# Directory of the raw data files
_data_root = '../Diabetes'
# Path to the raw training data
_data_filepath = os.path.join(_data_root, 'Diabetes.csv')
# Download data
os.makedirs(_data_root, exist_ok=True)
if not os.path.isfile(_data_filepath):
    #https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
    url = 'https://docs.google.com/uc?export= \
    download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
    r = requests.get(url, allow_redirects=True, stream=True)
    open(_data_filepath, 'wb').write(r.content)

df = pd.read_csv("../Diabetes/Diabetes.csv")

train_split = 0.8973
val_split = 0.05135
df_provicional = df.sample(frac=1, random_state=42)
df_train, df_validation, df_test = np.split(df_provicional, [int(train_split * len(df_provicional)), int((train_split+val_split) * len(df_provicional))])

print(len(df_train))
df_train.to_csv("../Diabetes/diabetes_train.csv", index=False)

print(len(df_validation))
df_validation.to_csv("../Diabetes/diabetes_val.csv", index=False)

print(len(df_test))
df_test.to_csv("../Diabetes/diabetes_test.csv", index=False)