# 🌍 Earthquake Magnitude Predictor

A machine learning web application that predicts earthquake magnitudes using Random Forest algorithm.

## 📊 Model Performance
- **Algorithm**: Random Forest
- **Accuracy**: 84%
- **Dataset**: 80,829 global earthquake records
- **Features**: 49 engineered features

## 🚀 Features
- Real-time earthquake magnitude prediction
- Interactive web interface
- Batch prediction support (CSV upload)
- Visual severity classification
- 49 input parameters including:
  - Geographic location
  - Seismic parameters
  - Error metrics
  - Temporal data
  - Network information

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **ML Model**: Scikit-learn Random Forest
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud

## 📥 Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/earthquake-prediction-app.git
cd earthquake-prediction-app

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 🌐 Live Demo
[Add your Streamlit Cloud link here after deployment]

## 📝 Usage

### Manual Prediction
1. Enter earthquake parameters (location, depth, seismic data)
2. Select date, time, and network
3. Click "Predict Magnitude"
4. View results with severity classification

### Batch Prediction
1. Upload CSV file with required columns
2. Click "Predict All"
3. Download results as CSV

## 📋 Required Input Features
- Latitude, Longitude, Depth
- Number of Stations (nst)
- Azimuthal Gap
- Distance to nearest station (dmin)
- RMS travel time residual
- Horizontal Error, Depth Error, Magnitude Error
- Date, Time information
- Seismic Network

## 👨‍💻 Author
Niyamat Ali Murtaza

## 🙏 Acknowledgments
- Global Earthquake Database

- Streamlit team for the amazing framework
