import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Load trained model
model = joblib.load("hdb_dt_model.pkl")

## Streamlit app
st.title("HDB Resale Price Prediction")

## Define the input options
towns = ['Bedok', 'Punggol', 'Tampines']
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM']
storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09']


## User inputs
town_selected = st.selectbox("Select Town", towns)
flat_type_selected = st.selectbox("Select Flat Type", flat_types)
storey_range_selected = st.selectbox("Select Storey", storey_ranges)
floor_area_selected = st.slider("Select Floor Area (sqm)", 
                                min_value=30, 
                                max_value=200, 
                                value=70)

## Predict button
if st.button("Predict HDB price"):

    ## Create dict for input features
    input_data = {
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area': floor_area_selected
    }

    ## Convert input data to a DataFrame
    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area': [floor_area_selected]
    })

    ## One-hot encoding
    df_input = pd.get_dummies(df_input, 
                              columns = ['town', 'flat_type', 'storey_range']
                             )
    
    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns = model.feature_names_in_,
                                fill_value=0)



    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Resale Price: ${y_unseen_pred:,.2f}")

## Page design
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://www.shutterstock.com/shutterstock/videos/1025418011/thumb/1.jpg");
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)