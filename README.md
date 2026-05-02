# Air Quality Prediction for Smog Mitigation
### Using Machine Learning in Urban Pakistan

> **DS Project by**  
> Eesha Waqar (22L-6780) · Reesha (22L-6666)

---

## Overview

F-aloodah is a machine learning-based air quality prediction system designed to forecast PM2.5 levels and AQI categories for major urban cities in Pakistan — primarily Lahore. The system enables proactive smog forecasting and supports early warning strategies for citizens and authorities.

Lahore is consistently ranked among the most polluted cities globally, with PM2.5 levels exceeding **300–500 µg/m³** during winter smog season (October–February) — over 20x the WHO safe limit of 15 µg/m³.

---

##  Objectives

- Collect and integrate historical air quality and weather data for Pakistani cities
- Analyze temporal and seasonal smog patterns through exploratory data analysis
- Develop and compare ML/DL models for AQI prediction
- Evaluate model performance using standard regression and classification metrics
- Provide an interactive early warning interface for smog management

---

## Project Structure

```
air-quality-pk/
│
├── app.py                                  # Streamlit web interface
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
├── lahore_air_quality_final_dataset.csv    # Final merged dataset
│
└── notebooks/
    ├── Data_pipeline_notebook.ipynb        # Data collection & preprocessing
    └── model_training.ipynb               # EDA, model training & evaluation
```

---

## Dataset

All data is sourced from publicly available, open-access platforms.

| Dataset | Source | Description |
|---|---|---|
| Air Quality Measurements | [OpenAQ](https://openaq.org) | Historical PM2.5 readings from monitoring stations |
| AQI Data | [AQICN / WAQI](https://aqicn.org) | Aggregated AQI from certified stations |
| Weather Data | [NASA POWER](https://power.larc.nasa.gov) | Temperature, humidity, wind speed via satellite |
| Climate Data | [NOAA GHCN](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) | Historical global weather records |

### Feature Description

| Column | Description |
|---|---|
| `pm25` | PM2.5 concentration (µg/m³) — **target variable** |
| `T2M` | Temperature at 2m height (°C) |
| `RH2M` | Relative humidity at 2m height (%) |
| `WS2M` | Wind speed at 2m height (m/s) |
| `pm25_lag_1` | Yesterday's PM2.5 value (engineered) |
| `pm25_lag_7` | PM2.5 from 7 days ago (engineered) |
| `month` | Month number 1–12 (engineered) |
| `smog_season` | 1 if Oct–Feb, 0 otherwise (engineered) |

---

## Methodology

### 1. Data Pre-processing & Feature Engineering

### 2. Exploratory Data Analysis

### 3. Models Training 
| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Random Forest | Ensemble |
| XGBoost | Gradient Boosting |
| LSTM | Deep Learning |

### 4. Evaluation of Models

---

## Results

| Model | MAE ↓ | RMSE ↓ | AQI Accuracy ↑ |
|---|---|---|---|
| Linear Regression | 32.89 | 51.10 | 0.614 |
| Random Forest | 33.15 | 50.14 | 0.598 |
| XGBoost | 37.67 | 54.28 | 0.530 |
| LSTM | 112.33 | 153.93 | N/A |

**Random Forest** achieved the best RMSE. **Linear Regression** achieved the best AQI category accuracy. LSTM underperformed due to limited dataset size (~730 daily records).

---

## Streamlit Interface

The app provides:
- **Live AQI Prediction** — input today's weather conditions and get an instant PM2.5 forecast
- **Historical Trends** — PM2.5 time series and monthly smog season boxplots
- **Model Comparison** — side-by-side actual vs predicted plots + MAE/RMSE bar charts
- **Feature Importance** — which variables drive PM2.5 predictions most

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Requirements

```
streamlit
pandas
numpy
matplotlib
scikit-learn
xgboost
tensorflow
seaborn
```

---

## Future Work

- Expand dataset to Faisalabad and Gujranwala
- Integrate real-time OpenAQ API for live data
- Improve LSTM with larger historical dataset
- Deploy on Streamlit Cloud for public access

---

<div align="center">
  Made with 🤍 at FAST NUCES Lahore · 2025
</div>
