import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# loading data from repository
df = pd.read_csv("tourism_package_prediction/data/tourism.csv")

# Defining the target variable
target_col = 'ProdTaken'

# removing unique column Customer Id
df.drop(columns=["CustomerID"], inplace=True)

# Handling missing values for numerical columns
print("Handling missing values for numerical columns")
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
  if df[col].isnull().sum() > 0:
    df[col].fillna(df[col].median(), inplace=True)

# Handling missing values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
  if df[col].isnull().sum() > 0:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fixing gender inconsistency for 'Fe Male'
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Fixing marital status inconsistency for 'Unmarried and Single'
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

label_encoder = LabelEncoder()

# Encoding categorical columns using label encoder
for col in categorical_cols:
  df[col] = label_encoder.fit_transform(df[col].astype(str))

X = df.drop(columns=[target_col])
y = df[target_col]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train data set size: ", Xtrain.shape[0])
print(f"Test data set size: ", Xtest.shape[0])

# saving the train test data set
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
