import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import joblib
import os
import time

# Step 1: Fetch Match Data (Web Scraping)
@st.cache_data  # Cache to avoid redundant scraping
def scrape_matches(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        
        matches = []
        for match in soup.find_all('div', class_='match-card'):  # Adjust class name as per website structure
            try:
                teams = match.find('div', class_='teams').text.strip()
                odds = float(match.find('span', class_='odds').text.strip())
                matches.append({'teams': teams, 'odds': odds})
            except AttributeError:
                continue  # Skip if any required data is missing
        
        return matches
    except requests.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
        return []

# Step 1.1: Fetch Match Data (API Alternative)
@st.cache_data
def fetch_data_from_api(api_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()  # Parse JSON response
    except requests.RequestException as e:
        st.error(f"Failed to fetch data from API: {e}")
        return None

# Step 2: Train or Load Machine Learning Model
@st.cache_resource  # Cache the model to avoid retraining
def train_model(data=None):
    if data is None and os.path.exists("trained_model.pkl"):
        # Load pre-trained model
        model = joblib.load("trained_model.pkl")
    else:
        # Train a new model
        if data is None:
            data = pd.read_csv("historical_matches.csv")
        
        # Ensure required columns are present
        required_columns = ['team_strength', 'home_advantage', 'past_performance', 'outcome']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Uploaded data must contain the following columns: {required_columns}")
            return None
        
        X = data[['team_strength', 'home_advantage', 'past_performance']]
        y = data['outcome']
        
        model = RandomForestClassifier()
        model.fit(X, y)
        
        # Save the trained model
        joblib.dump(model, "trained_model.pkl")
        st.success("Model trained and saved successfully!")
    
    return model

# Step 3: Filter Matches Based on User Criteria
def filter_matches(matches, min_odds, max_odds, num_matches):
    if not matches:
        return []
    
    filtered = [match for match in matches if min_odds <= match['odds'] <= max_odds]
    return filtered[:num_matches]

# Step 4: Generate Betting Tips
def generate_betting_tips(matches, model, min_odds, max_odds, num_matches):
    if not matches:
        return []
    
    for match in matches:
        try:
            # Replace with actual feature engineering logic
            team_strength = 1  # Example: Replace with actual team strength
            home_advantage = 1  # Example: Replace with actual home advantage
            features = [[match['odds'], team_strength, home_advantage]]
            match['prediction'] = model.predict(features)[0]
        except Exception as e:
            st.error(f"Prediction failed for match {match['teams']}: {e}")
            match['prediction'] = 'Unknown'
    
    return filter_matches(matches, min_odds, max_odds, num_matches)

# Step 5: Build a User Interface (Streamlit)
def main():
    st.title("Football Betting Tips Generator")
    
    # Input fields
    data_source = st.radio("Choose Data Source", ["Scraping", "API", "File Upload"])
    
    if data_source == "Scraping":
        url = st.text_input("Enter website URL for match data")
    
    elif data_source == "API":
        api_url = st.text_input("Enter API URL")
        api_key = st.text_input("Enter API Key", type="password")
    
    min_odds = st.number_input("Minimum Odds", value=1.5, min_value=1.0)
    max_odds = st.number_input("Maximum Odds", value=3.0, min_value=1.0)
    num_matches = st.number_input("Number of Matches", value=5, min_value=1)
    
    # File upload for historical data
    historical_data = st.file_uploader("Upload historical match data (CSV)", type=["csv"])
    if historical_data:
        data = pd.read_csv(historical_data)
        st.write("Preview of uploaded data:")
        st.write(data.head())
        
        # Retrain model if new data is uploaded
        if st.button("Retrain Model"):
            with st.spinner("Training model..."):
                model = train_model(data)
                time.sleep(1)  # Simulate delay
            st.success("Model retrained successfully!")
    
    if st.button("Generate Tips"):
        if data_source == "Scraping" and url:
            # Fetch match data via scraping
            with st.spinner("Scraping match data..."):
                matches = scrape_matches(url)
                time.sleep(1)  # Simulate delay
        elif data_source == "API" and api_url and api_key:
            # Fetch match data via API
            with st.spinner("Fetching match data from API..."):
                matches = fetch_data_from_api(api_url, api_key)
                matches = [{'teams': m['teams'], 'odds': m['odds']} for m in matches] if matches else []
                time.sleep(1)  # Simulate delay
        else:
            st.error("Please provide a valid data source (URL/API).")
            return
        
        if not matches:
            st.error("No matches found. Please check the source and try again.")
            return
        
        # Train or load model
        model = train_model()
        if model is None:
            return
        
        # Generate betting tips
        with st.spinner("Generating betting tips..."):
            tips = generate_betting_tips(matches, model, min_odds, max_odds, num_matches)
            time.sleep(1)  # Simulate delay
        st.success("Tips generated!")
        
        # Display tips
        if tips:
            st.write("### Betting Tips:")
            for tip in tips:
                st.write(f"*Match: {tip['teams']} | **Odds: {tip['odds']} | **Prediction*: {tip['prediction']}")
        else:
            st.warning("No matches match your criteria.")

# Run the app
if __name__ == "_main_":
    main()