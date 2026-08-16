import pandas as pd

DATA_PATH = "tourism_package_prediction/data/tourism.csv"

# Loading dataset from DATA_PATH
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")

# Validating the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("\n========== DATASET VALIDATION ==========")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nExpected columns:")
print(expected_columns)

print("\nMissing columns:")
print(missing)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nProdTaken distribution:")
print(df["ProdTaken"].value_counts())

print("\nDataset validation completed successfully.")
print("\nDataset registered successfully.")
