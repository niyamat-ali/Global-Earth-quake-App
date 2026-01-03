import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import gdown
import os

# Page configuration
st.set_page_config(
    page_title="Earthquake Magnitude Predictor",
    page_icon="🌍",
    layout="wide"
)

# Load model from Google Drive
@st.cache_resource
def load_model():
    try:
        model_path = 'models.pkl'
        
        # ⚠️ IMPORTANT: Replace 'YOUR_FILE_ID_HERE' with your actual Google Drive file ID
        # Example: If your link is https://drive.google.com/file/d/1ABC123XYZ/view
        # Then replace with: file_id = '1ABC123XYZ'
        file_id = '1gVfl1-jcXLpZMs1_dBpcAbzZkNvST3mT'
        
        # Check if model already exists locally
        if not os.path.exists(model_path):
            with st.spinner('🔄 Model download ho raha hai... Please wait (5-10 seconds)'):
                url = f'https://drive.google.com/uc?id={file_id}'
                output = gdown.download(url, model_path, quiet=False)
                
                if output is None:
                    st.error("❌ Model download fail. Check if Google Drive link is public.")
                    return None
                
                st.success("✅ Model downloaded successfully!")
        
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure Google Drive link is set to 'Anyone with link can view'")
        return None

# Load model
model = load_model()

# Title and description
st.title("🌍 Global Earthquake Magnitude Predictor")
st.markdown("### Random Forest Model (Accuracy: 84%)")
st.markdown("---")

# Check if model loaded successfully
if model is None:
    st.error("⚠️ Model load nahi hua. Please contact admin.")
    st.stop()

# Create tabs for different input methods
tab1, tab2 = st.tabs(["🔢 Manual Input", "📁 Batch Prediction"])

with tab1:
    st.markdown("### Enter Earthquake Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location Details")
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=35.0, step=0.1)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-120.0, step=0.1)
        depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0, step=0.5)
    
    with col2:
        st.subheader("📊 Seismic Parameters")
        nst = st.number_input("Number of Stations (nst)", min_value=0.0, value=50.0, step=1.0)
        gap = st.number_input("Azimuthal Gap (degrees)", min_value=0.0, max_value=360.0, value=100.0, step=1.0)
        dmin = st.number_input("Distance to nearest station", min_value=0.0, value=0.5, step=0.01)
        rms = st.number_input("RMS travel time residual", min_value=0.0, value=0.3, step=0.01)
    
    with col3:
        st.subheader("⚠️ Error Metrics")
        horizontalError = st.number_input("Horizontal Error (km)", min_value=0.0, value=0.5, step=0.1)
        depthError = st.number_input("Depth Error (km)", min_value=0.0, value=1.0, step=0.1)
        magError = st.number_input("Magnitude Error", min_value=0.0, value=0.1, step=0.01)
        magNst = st.number_input("Magnitude Stations Count", min_value=0.0, value=20.0, step=1.0)
    
    # Date and time inputs
    st.markdown("---")
    st.subheader("📅 Date & Time")
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        date_input = st.date_input("Date", value=datetime.now())
    with col_date2:
        time_input = st.time_input("Time", value=datetime.now().time())
    
    # Network selection
    st.subheader("🌐 Seismic Network")
    networks = ['ak', 'ci', 'hv', 'iscgem', 'nc', 'nn', 'ok', 'pr', 'se', 'tx', 'us', 'uu', 'uw']
    selected_network = st.selectbox("Select Network", networks, index=10)  # 'us' default
    
    st.markdown("---")
    
    # Create prediction button
    if st.button("🔮 Predict Magnitude", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            # Extract date/time features
            year = date_input.year
            month = date_input.month
            day = date_input.day
            hour = time_input.hour
            dayofweek = date_input.weekday()
            
            # Calculate derived features
            distance_from_origin = np.sqrt(latitude**2 + longitude**2)
            is_shallow = 1 if depth < 70 else 0
            is_deep = 1 if depth > 300 else 0
            mag_squared = 5.0**2  # Initial estimate
            depth_mag_interaction = depth * 5.0
            lat_long_interaction = latitude * longitude
            abs_latitude = abs(latitude)
            abs_longitude = abs(longitude)
            high_quality = 1 if horizontalError < 1.0 and depthError < 2.0 else 0
            error_ratio = horizontalError / (depthError + 0.001)
            is_winter = 1 if month in [12, 1, 2] else 0
            is_summer = 1 if month in [6, 7, 8] else 0
            is_night = 1 if hour >= 18 or hour <= 6 else 0
            gap_normalized = gap / 360.0
            nst_log = np.log1p(nst)
            distance_from_center = distance_from_origin
            
            # Create network boolean features
            net_features = {f'net_{net}': (1 if net == selected_network else 0) for net in networks}
            
            # Build feature array matching training data order
            features = [
                latitude, longitude, depth, 5.0,  # mag placeholder
                nst, gap, dmin, rms,
                horizontalError, depthError, magError, magNst,
                year, month, day, hour, dayofweek
            ]
            
            # Add network features in order
            for net in networks:
                features.append(net_features[f'net_{net}'])
            
            # Add remaining features
            features.extend([
                distance_from_origin,
                is_shallow, is_deep,
                mag_squared, depth_mag_interaction, lat_long_interaction,
                abs_latitude, abs_longitude,
                high_quality, error_ratio,
                is_winter, is_summer,
                is_night, gap_normalized, nst_log, distance_from_center
            ])
            
            # Create feature names matching your dataset
            feature_names = [
                'latitude', 'longitude', 'depth', 'mag', 'nst', 'gap', 'dmin', 'rms',
                'horizontalError', 'depthError', 'magError', 'magNst',
                'year', 'month', 'day', 'hour', 'dayofweek'
            ]
            feature_names.extend([f'net_{net}' for net in networks])
            feature_names.extend([
                'distance_from_origin', 'is_shallow', 'is_deep',
                'mag_squared', 'depth_mag_interaction', 'lat_long_interaction',
                'abs_latitude', 'abs_longitude', 'high_quality', 'error_ratio',
                'is_winter', 'is_summer', 'is_night', 'gap_normalized', 
                'nst_log', 'distance_from_center'
            ])
            
            # Create DataFrame
            input_df = pd.DataFrame([features], columns=feature_names)
            
            try:
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display result with animation
                st.balloons()
                st.success("✅ Prediction Complete!")
                
                # Results display
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric(
                        label="🎯 Predicted Magnitude",
                        value=f"{prediction:.2f}",
                        delta=None
                    )
                
                with result_col2:
                    if prediction < 4.0:
                        severity = "Light 🟢"
                        color = "green"
                    elif prediction < 6.0:
                        severity = "Moderate 🟡"
                        color = "orange"
                    elif prediction < 7.0:
                        severity = "Strong 🟠"
                        color = "orange"
                    else:
                        severity = "Major 🔴"
                        color = "red"
                    
                    st.metric(label="⚡ Severity Level", value=severity)
                
                with result_col3:
                    st.metric(label="📈 Model Accuracy", value="84%")
                
                # Additional info
                st.markdown("---")
                st.info(f"""
                **Prediction Details:**
                - Location: ({latitude:.2f}, {longitude:.2f})
                - Depth: {depth:.1f} km
                - Network: {selected_network.upper()}
                - Time: {date_input} {time_input}
                """)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Please check if all input values are correct.")

with tab2:
    st.markdown("### 📁 Upload CSV for Batch Predictions")
    st.info("Upload a CSV file with the same columns as the training dataset")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Uploaded Data Preview:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.write(f"Total rows: {len(df)}")
            
            if st.button("🔮 Predict All", type="primary"):
                with st.spinner(f"Processing {len(df)} predictions..."):
                    try:
                        predictions = model.predict(df)
                        df['predicted_magnitude'] = predictions
                        
                        st.success(f"✅ {len(df)} Predictions Complete!")
                        st.dataframe(df, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Magnitude", f"{predictions.mean():.2f}")
                        with col2:
                            st.metric("Max Magnitude", f"{predictions.max():.2f}")
                        with col3:
                            st.metric("Min Magnitude", f"{predictions.min():.2f}")
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv,
                            file_name="earthquake_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.info("Make sure CSV has all required columns")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### 📖 About This Model")
col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Model Information:**
    - Algorithm: Random Forest
    - Accuracy: 84%
    - Dataset: 80,829 records
    - Features: 49 engineered features
    """)

with col2:
    st.info("""
    **Features Include:**
    - Geographic location
    - Seismic parameters
    - Error metrics
    - Temporal data
    - Network information
    """)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Data Source: Global Earthquake Database")