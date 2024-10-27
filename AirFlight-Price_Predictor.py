import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

# Page config
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {
        background-color: #1A202C;
        color: #FFFFFF;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 2em;
        background-color: #0066cc;
        color: white;
    }
    .prediction-box {
        background-color: #2D3748;
        padding: 2em;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #4A5568;
        margin-top: 1em;
    }
    .metrics-box {
        background-color: #2D3748;
        padding: 1.5em;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #4A5568;
        margin-top: 1em;
        color: #FFFFFF;
    }
    .input-label {
        font-size: 1.2em;
        color: #A0AEC0;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .custom-time-input {
        background-color: #2D3748;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #4A5568;
    }
    .time-header {
        color: #A0AEC0;
        font-size: 1.1em;
        margin-bottom: 15px;
    }
    .time-value {
        color: #FFFFFF;
        font-size: 1.8em;
        text-align: center;
        padding: 10px;
        background-color: #1A202C;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #2D3748;
        padding: 1.5em;
        border-radius: 10px;
        margin-top: 1em;
        border: 1px solid #4A5568;
        color: #FFFFFF;
    }
    .warning-box {
        background-color: #553C2C;
        color: #FFB38A;
        padding: 1em;
        border-radius: 10px;
        margin-top: 1em;
        border: 1px solid #805C44;
    }
    .section-title {
        color: #A0AEC0;
        font-size: 1.5em;
        font-weight: 600;
        margin-bottom: 1em;
    }
    .value-display {
        color: #FFFFFF;
        font-size: 1.1em;
    }
    .stSelectbox > div > div {
        background-color: #2D3748 !important;
        color: #FFFFFF !important;
    }
    .stNumberInput > div > div > input {
        background-color: #2D3748 !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #2D3748 !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and values
@st.cache_resource
def load_models():
    model = joblib.load("XGB_Regressor_pipeline.pkl")
    unique_values = joblib.load("unique_values.pkl")
    min_max_values = joblib.load("min_max_values.pkl")
    return model, unique_values, min_max_values

model, unique_values, min_max_values = load_models()

@st.cache_resource
def init_geocoder():
    geolocator = Nominatim(user_agent="my_agent")
    coords = {}
    unique_cities = set(unique_values['Source']) | set(unique_values['Destination'])
    
    for city in unique_cities:
        try:
            time.sleep(1)  # Add sleep to respect API limits
            location = geolocator.geocode(city + ", India")  # Adding ", India" for better accuracy
            if location:
                coords[city] = (location.latitude, location.longitude)
            else:
                st.warning(f"Could not find coordinates for {city}")
                coords[city] = None
        except Exception as e:
            st.error(f"Error getting coordinates for {city}: {str(e)}")
            coords[city] = None
    return coords

def calculate_distance(source, dest, coords):
    source_coords = coords.get(source)
    dest_coords = coords.get(dest)
    
    if source_coords and dest_coords:
        try:
            return round(geodesic(source_coords, dest_coords).kilometers, 2)
        except Exception as e:
            st.error(f"Error calculating distance: {str(e)}")
            return None
    return None

def categorize_time(hour):
    if 5 <= hour < 12: return "morning"
    elif 12 <= hour < 17: return "afternoon"
    elif 17 <= hour < 21: return "evening"
    else: return "night"

def calculate_arrival(dep_day, dep_month, dep_hour, dep_min, duration_mins):
    base = datetime(2024, dep_month, dep_day, dep_hour, dep_min)
    arrival = base + timedelta(minutes=duration_mins)
    return {
        'month': arrival.month,
        'day': arrival.day,
        'hour': arrival.hour,
        'minute': arrival.minute,
        'part_of_day': categorize_time(arrival.hour)
    }

def format_time(hour, minute):
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour if 1 <= hour <= 12 else abs(hour - 12) if hour != 0 else 12
    return f"{hour_12:02d}:{minute:02d} {period}"

def main():
    city_coordinates = init_geocoder()
    st.markdown('<h1>✈️ Flight Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="input-label">Predict your flight prices across India</p>', unsafe_allow_html=True)

    st.markdown("""
        <div class="warning-box">
            ⚠️ The model was trained on flights from March (3) to June (6). 
            For more reliable predictions, consider booking within these months.
        </div>
    """, unsafe_allow_html=True)

    col1, space, col2 = st.columns([1, 0.1, 1])
    
    with col1:
        st.markdown('<p class="section-title">📋 Flight Details</p>', unsafe_allow_html=True)
        
        airline = st.selectbox("Select Airline ✈️", unique_values['Airline'])
        source = st.selectbox("Select Source 🛫", unique_values['Source'])
        destination = st.selectbox("Select Destination 🛬", unique_values['Destination'])
        
        distance = calculate_distance(source, destination, city_coordinates)
        if distance:
            st.markdown(f"""
                <div class="info-box">
                    <p class="input-label">📏 Flight Distance</p>
                    <p class="value-display">{distance:.2f} km</p>
                </div>
                """, unsafe_allow_html=True)
        
        total_stops = st.selectbox("Number of Stops 🛑", 
                                 range(min_max_values['Total_Stops'][0], 
                                 min_max_values['Total_Stops'][1] + 1))
        
        additional_info = st.selectbox("Additional Information ℹ️", 
                                     unique_values['Additional_Info'])

    with col2:
        st.markdown('<p class="section-title">🕒 Time Details</p>', unsafe_allow_html=True)
        
        month_names = [datetime(2024, i, 1).strftime('%B') for i in range(1, 13)]
        dep_month = st.selectbox("Month 📅", 
                               range(1, 13), 
                               format_func=lambda x: month_names[x-1])
        
        dep_day = st.selectbox("Day 📆", range(1, 31))

        st.markdown('<div class="custom-time-input">', unsafe_allow_html=True)
        time_cols = st.columns(2)
        with time_cols[0]:
            dep_hour = st.selectbox("Hour", range(24), 
                                  format_func=lambda x: f"{x:02d}")
        with time_cols[1]:
            dep_min = st.selectbox("Minute", range(0, 60, 5), 
                                 format_func=lambda x: f"{x:02d}")
        
        st.markdown(f'<div class="time-value">{format_time(dep_hour, dep_min)}<br><span style="font-size: 0.7em; color: #A0AEC0;">({categorize_time(dep_hour)})</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-time-input">', unsafe_allow_html=True)
        duration_cols = st.columns(2)
        with duration_cols[0]:
            duration_hours = st.number_input("Hours", min_value=0, max_value=47, value=2)
        with duration_cols[1]:
            duration_mins = st.number_input("Minutes", min_value=0, max_value=59, value=30)
            
        total_duration = duration_hours * 60 + duration_mins
        
        if total_duration < min_max_values['Duration'][0] or total_duration > min_max_values['Duration'][1]:
            min_duration_hrs = min_max_values['Duration'][0] // 60
            min_duration_mins = min_max_values['Duration'][0] % 60
            max_duration_hrs = min_max_values['Duration'][1] // 60
            max_duration_mins = min_max_values['Duration'][1] % 60
            st.markdown(f"""
                <div class="warning-box">
                    ⚠️ For more accurate predictions, consider flight durations between 
                    {min_duration_hrs}h {min_duration_mins}m and {max_duration_hrs}h {max_duration_mins}m
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f'<div class="time-value">{duration_hours}h {duration_mins}m</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        arrival = calculate_arrival(dep_day, dep_month, dep_hour, dep_min, total_duration)
        st.markdown(f"""
            <div class="info-box">
                <p class="input-label">🛬 Arrival Details</p>
                <p class="value-display">Date: {month_names[arrival['month']-1]} {arrival['day']}</p>
                <p class="value-display">Time: {format_time(arrival['hour'], arrival['minute'])}</p>
                <p class="value-display">Period: {arrival['part_of_day'].title()}</p>
            </div>
            """, unsafe_allow_html=True)

    if st.button("Predict Price 💰", type="primary"):
        features = pd.DataFrame({
            'Airline': [airline],
            'Source': [source],
            'Destination': [destination],
            'Duration': [total_duration],
            'Total_Stops': [total_stops],
            'Additional_Info': [additional_info],
            'Dep_Day': [dep_day],
            'Dep_Month': [dep_month],
            'Distance_km': [distance],
            'Dep_Hour': [dep_hour],
            'Dep_Min': [dep_min],
            'Dep_Part_of_Day': [categorize_time(dep_hour)],
            'Arrival_Day': [arrival['day']],
            'Arrival_Month': [arrival['month']],
            'Arr_Hour': [arrival['hour']],
            'Arr_Min': [arrival['minute']],
            'Arr_Part_of_Day': [arrival['part_of_day']]
        })
        
        prediction = model.predict(features)
        
        result_cols = st.columns(2)
        with result_cols[0]:
            st.markdown(f"""
                <div class="prediction-box">
                    <p class="input-label">Predicted Price</p>
                    <h2 style='color: #FFFFFF; font-size: 2.5em;'>₹{prediction[0]:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with result_cols[1]:
            st.markdown("""
                <div class="metrics-box">
                    <p class="input-label">Model Details</p>
                    <p class="value-display">Model Used: XGBoost Regressor</p>
                    <p class="value-display">Accuracy (R² Score): 0.9333</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()