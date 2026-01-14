# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Download the Latest Version of Dataset 
import kagglehub
path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")
print("path to dataset files:" , path)



# Loading Data and Seperating Features
def load_har_csv(dataset_path):
  # Construct file paths
  train_path = os.path.join(dataset_path, 'train.csv')
  test_path = os.path.join(dataset_path, 'test.csv')

  # Load data
  train_df = pd.read_csv(train_path)
  test_df = pd.read_csv(test_path)

  # SEPERATE FEATURES (X) AND TARGET (Y)
  # Drop 'subject' and 'Activity' to get X
  X_train = train_df.drop(['subject', 'Activity'], axis=1)
  y_train = train_df['Activity']

  X_test = test_df.drop(['subject', 'Activity'], axis=1)
  y_test = test_df['Activity']

  return X_train, y_train, X_test, y_test

try:
    X_train, y_train, X_test, y_test = load_har_csv(path)
except KeyError as e:
    print(f"Error: Column not found - {e}. The CSV headers might differ.")
    exit()

# Display Shape and Basic Info
print(f"\nTraining Shape: {X_train.shape}")
print(f"Testing Shape: {X_test.shape}")
print(f"Number of Features: {X_train.shape[1]}")
print(f"Number of Classes: {y_train.nunique()}")

print("\nFirst 5 Rows of Training Features:")
print(X_train.head())



# Normalization Comparison
# min-max scaling
min_max_scaler = MinMaxScaler()
X_train_mm = min_max_scaler.fit_transform(X_train)
X_test_mm = min_max_scaler.transform(X_test)

# z-score (standard) scaling
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

# quick comparison using a base linear svm
print("comparing scaling methods with base linear svm")
svm_mm = SVC(kernel = 'linear')
svm_mm.fit(X_train_mm, y_train)
acc_mm = accuracy_score(y_test, svm_mm.predict(X_test_std))

svm_std = SVC(kernel = 'linear')
svm_std.fit(X_train_std, y_train)
acc_std = accuracy_score(y_test, svm_std.predict(X_test_std))

print(f"accuracy with min max scaler: {acc_mm:.4f}")
print(f"accuracy with standard scaler:  {acc_std:.4f}")

# select the better scaler
if acc_std > acc_mm:
  print("proceeding with standard scaler")
  X_train_final = X_train_std
  X_test_final = X_test_std
else:
  print("proceeding with min max scaler")
  X_train_final = X_train_mm
  X_test_final = X_test_mm


param_grid_linear = {'C' : [0.1, 1, 10]}
param_grid_rbf = {'C' : [1,10], 'gamma': ['scale', 0.1]}
param_grid_poly = {'C': [1,10], 'degree': [3]}

kernels = [('Linear', 'linear', param_grid_linear),
           ('RBF', 'rbf', param_grid_rbf), ('Polynomial', 'poly', param_grid_poly)]

results = {}

X_train_final = X_train_std
X_test_final = X_test_std



# SVM Training and Tuning
for name, kernel_type, param_grid in kernels:
  grid = GridSearchCV(SVC(kernel = kernel_type), param_grid, cv = 3, n_jobs = -1)

  start_time = time.time()
  grid.fit(X_train_final, y_train)
  end_time = time.time() - start_time

  # best model evaluation
  best_model = grid.best_estimator_
  y_pred = best_model.predict(X_test_final)
  acc = accuracy_score(y_test, y_pred)

  results[name] = {
        'Best Params': grid.best_params_,
        'Accuracy': acc,
        'Time': end_time,
        'Model': best_model
    }

print(f"  Best Params: {grid.best_params_}")
print(f"  Accuracy: {acc:.4f}")


