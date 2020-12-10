import numpy as np
import pandas as pd
import sklearn 
import tensorflow as tf
from tensorflow.estimator import LinearClassifier
from pandas.api.types import CategoricalDtype
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split



cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv('income_evaluation.csv', header = 1, names = cols)
#Change values into negative and positive class
df['income'].replace({' <=50K':0, ' >50K':1}, inplace=True)
df_obj = df.select_dtypes(include=['object'])
df_int = df.select_dtypes(include=['int64'])

def fit(X, y=None):
    categories = dict()
    df_num = X.select_dtypes(include=['object'])
    for column in df_num.columns:
        categories[column] = df_num[column].value_counts().index.tolist()
    return categories


def transform(X):
    X_copy = X.copy()
    categories = fit(X)
    X_copy = X_copy.select_dtypes(include=['object'])
    for column in X_copy.columns:
        X_copy[column] = X_copy[column].astype({column: CategoricalDtype(categories[column])})
    return pd.get_dummies(X_copy, drop_first=True)

all_text = transform(df_obj)

all_data = df_int.merge(all_text, left_index = True, right_index = True)


# Feature Matrix 
X = all_data.drop('income', axis =1)
# Data labels 
y = all_data['income']

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

train_input = input_pipeline(X_train, y_train)
train_input_testing = input_pipeline(X_train, y_train, num_of_epochs=1, shuffle=False)
test_input = input_pipeline(X_test,y_test, num_of_epochs=1, shuffle=False)

#Model training
feature_columns_numeric = [tf.feature_column.numeric_column(m) for m in X_train.columns]
logistic_model = LinearClassifier(feature_columns=feature_columns_numeric)
logistic_model.train(train_input)

#Predictions
train_predictions = logistic_model.predict(train_input_testing)
test_predictions = logistic_model.predict(test_input)
train_predictions_series = pd.Series([p['classes'][0].decode("utf-8") for p in train_predictions])
test_predictions_series = pd.Series([p['classes'][0].decode("utf-8") for p in test_predictions])

train_predictions_df = pd.DataFrame(train_predictions_series, columns=['predictions'])
test_predictions_df = pd.DataFrame(test_predictions_series, columns=['predictions'])

y_train.reset_index(drop=True, inplace=True)

#Validation
def calculate_binary_class_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred.astype('int64'))
    precision = precision_score(y_true, y_pred.astype('int64'))
    recall = recall_score(y_true, y_pred.astype('int64'))
    return accuracy, precision, recall

train_accuracy_score, train_precision_score, train_recall_score = calculate_binary_class_scores(y_train, train_predictions_series)
test_accuracy_score, test_precision_score, test_recall_score = calculate_binary_class_scores(y_test, test_predictions_series)

print('Training Data Accuracy (%) = ', round(train_accuracy_score*100,2))
print('Training Data Precision (%) = ', round(train_precision_score*100,2))
print('Training Data Recall (%) = ', round(train_recall_score*100,2))

print('#'*50)

print('Test Data Accuracy (%) = ', round(test_accuracy_score*100,2))
print('Test Data Precision (%) = ', round(test_precision_score*100,2))
print('Test Data Recall (%) = ', round(test_recall_score*100,2))


