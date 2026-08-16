import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction")
st.write("Please enter the customer details:")

# Creating UI form to fetch customer data
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=30)
Occupation = st.selectbox("Occupation",["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting",min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups",min_value=0, max_value=20, value=3)
ProductPitched = st.selectbox("Product Pitched",["Basic", "Standard", "Deluxe", "Super Deluxe"])
PreferredPropertyStar = st.selectbox("Preferred Property Star",[1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status",["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips (per year)",min_value=0, max_value=50, value=2)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score",min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting",min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation",["Executive", "Manager", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income",min_value=5000, max_value=500000, value=50000)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

classification_threshold = 0.45

if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)
    st.metric("Purchase Probability",f"{prediction_proba:.2%}")
    if prediction == 1:
        st.success("The customer is likely to purchase the package.")
    else:
        st.warning("The customer is unlikely to purchase the package.")    
