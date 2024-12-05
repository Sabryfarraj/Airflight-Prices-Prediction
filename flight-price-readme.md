# Air Flight Price Prediction App ✈️

A machine learning application that predicts flight prices using XGBoost Regressor. The application is containerized using Docker and provides an interactive web interface built with Streamlit.

## 🚀 Quick Start

You have two options to access the application:

### Option 1: Web Browser (No Installation Required)
Visit the live application at:
```
https://airflight-prices-prediction-by-sabryfarraj.streamlit.app/
```
This option requires no setup - just click and use!

### Option 2: Run using Docker

If you prefer to run the application locally:

```bash
# Pull the image from Docker Hub
docker pull sabryfarraj/flight-price-predictor:latest

# Run the container
docker run -p 8501:8501 sabryfarraj/flight-price-predictor:latest
```

After running these commands, open your browser and visit:
```
http://localhost:8501
```

## 🛠️ Project Structure
```
Airflight-Prices-Prediction/
├── AirFlight-Price_Predictor.py     # Main Streamlit application
├── XGB_Regressor_pipeline.pkl       # Trained XGBoost model
├── min_max_values.pkl               # Feature scaling values
├── unique_values.pkl                # Categorical features mapping
├── requirements.txt                 # Python dependencies
```

## 📋 Prerequisites
For Docker option only:
- Docker installed on your machine ([Install Docker](https://docs.docker.com/get-docker/))

## 🔧 Local Development

If you want to build the Docker image locally:

```bash
# Clone the repository
git clone https://github.com/Sabryfarraj/Airflight-Prices-Prediction.git

# Navigate to project directory
cd Airflight-Prices-Prediction

# Build Docker image
docker build -t flight-price-predictor .

# Run container
docker run -p 8501:8501 flight-price-predictor
```

## 📊 Features
- Predicts flight prices based on various factors
- Interactive web interface
- Real-time predictions
- Available as web application and containerized application
- Pre-trained XGBoost model
- Handles various input features like:
  - Airlines
  - Source and Destination cities
  - Flight duration
  - Stops
  - Class type
  - Booking timing

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Authors
- Sabry Farraj

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments
- Built with Streamlit and XGBoost
- Uses geopy for distance calculations
- Hosted on Streamlit Cloud
