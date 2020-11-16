import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import tensorflow as tf

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv('income_evaluation.csv')  

# Feature Matrix 
x = df.iloc[:, 1:-1].values 
# Data labels 
y = df.iloc[:, -1:].values 
print(y)
