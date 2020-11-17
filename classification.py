import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv('income_evaluation.csv', header = 1, names = cols)

#Change values into negative and positive class
df['income'].replace({' <=50K':0, ' >50K':1}, inplace=True)

# Feature Matrix 
X = df.iloc[:, :-1].values 
# Data labels 
y = df.iloc[:, -1:].values 


X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2)

#Input pipeline
def input_pipeline(features_df, target_df, num_of_epochs=2, shuffle=True, batch_size = 20):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(features_df), target_df))
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat(num_of_epochs)
        return dataset
    return input_function
