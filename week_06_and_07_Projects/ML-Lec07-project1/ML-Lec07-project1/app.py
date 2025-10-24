import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Define categorical mappings
bsmt_qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
kitchen_qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}

# Load the saved neighborhood encoder
neighborhood_encoder = joblib.load("neighborhood_encoder.pkl")

def preprocess_input(overall_qual, gr_liv_area, total_bsmt_sf, year_built, neighborhood, 
                      garage_cars, bsmt_qual, kitchen_qual, full_bath, lot_area):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [np.log1p(gr_liv_area)],
        'TotalBsmtSF': [np.log1p(total_bsmt_sf)],
        'YearBuilt': [year_built],
        'GarageCars': [garage_cars],
        'BsmtQual': [bsmt_qual_mapping.get(bsmt_qual, 0)],  # Convert ordinal category
        'KitchenQual': [kitchen_qual_mapping.get(kitchen_qual, 3)],
        'FullBath': [full_bath],
        'LotArea': [np.log1p(lot_area)],
    })

    # One-hot encode Neighborhood using the saved encoder
    neighborhood_encoded = pd.DataFrame(neighborhood_encoder.transform([[neighborhood]]).toarray(), 
                                        columns=neighborhood_encoder.get_feature_names_out())

    # Concatenate with input data
    input_data = pd.concat([input_data, neighborhood_encoded], axis=1)
    return input_data

# Streamlit UI
st.title("House Price Prediction App")
st.write("Enter details of the house to predict its sale price.")

# Input fields
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=5000, value=1500)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
year_built = st.number_input("Year Built", min_value=1870, max_value=2024, value=2000)
neighborhood = st.selectbox("Neighborhood", neighborhood_encoder.categories_[0])  # Load from encoder
garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
bsmt_qual = st.selectbox("Basement Quality", list(bsmt_qual_mapping.keys()))
kitchen_qual = st.selectbox("Kitchen Quality", list(kitchen_qual_mapping.keys()))
full_bath = st.slider("Full Bathrooms", 0, 4, 2)
lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=50000, value=7000)

# Prediction button
if st.button("Predict Price"):
    input_features = preprocess_input(overall_qual, gr_liv_area, total_bsmt_sf, year_built, neighborhood, 
                                      garage_cars, bsmt_qual, kitchen_qual, full_bath, lot_area)
    # Predict
    log_price = model.predict(input_features)[0]
    predicted_price = np.expm1(log_price)  # Reverse log transformation

    st.success(f"Predicted Sale Price: ${predicted_price:,.2f}")