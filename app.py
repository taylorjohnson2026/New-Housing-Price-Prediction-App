# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_excel('AmesHousing.xlsx')
    return df

# Train the model
@st.cache_data
def train_model(df):
    # Select features and target
    features = df.drop('SalePrice', axis=1)
    target = df['SalePrice']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Mean Squared Error: {mse}")
    
    return model

# Streamlit app
def main():
    st.title("Housing Price Prediction App")
    st.write("This app predicts housing prices based on the Ames Housing Dataset.")
    
    # Load data
    df = load_data()
    
    # Display the dataset
    if st.checkbox("Show raw data"):
        st.write(df)
    
    # Train the model
    model = train_model(df)
    
    # User input for prediction
    st.sidebar.header("User Input Features")
    input_features = {}
    for column in df.columns:
        if column != 'SalePrice':
            input_features[column] = st.sidebar.number_input(f"Enter {column}", value=float(df[column].median()))
    
    # Predict
    if st.button("Predict Price"):
        input_df = pd.DataFrame([input_features])
        prediction = model.predict(input_df)
        st.success(f"Predicted Housing Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
