# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Loading data from repository
df = pd.read_csv("tourism_package_prediction/data/tourism.csv")

# Define the target variable
target_col = 'ProdTaken'

# Removing unique column Customer Id
df.drop(columns=["CustomerID"], inplace=True)

# Handling missing values for numerical columns
print("Handling missing values for numerical columns")
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].median())

# Handling missing values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fixing gender inconsistency for data - 'Fe Male'
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Fixing marital status inconsistency for 'Unmarried and Single'
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

X = df.drop(columns=[target_col])
y = df[target_col]

# Splitting the dataset into training and test sets 80:20
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train data set size: ", Xtrain.shape[0])
print(f"Test data set size: ", Xtest.shape[0])

# Saving the train and test data set
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
