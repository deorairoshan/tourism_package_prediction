import pandas as pd

DATA_PATH = "tourism_package_prediction/data/tourism.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
