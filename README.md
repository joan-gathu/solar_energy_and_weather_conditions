🌞 Renewable Energy Forecasting with Weather Data

This project focuses on forecasting solar energy production using weather conditions and machine learning. By combining exploratory analysis, anomaly detection, and predictive modeling, the work provides insights into the factors that most influence renewable energy generation.

The dataset includes solar energy output alongside meteorological variables such as Global Horizontal Irradiance (GHI), temperature, humidity, wind speed, cloud cover, rainfall, and snowfall.

🔎 Key Steps

Exploratory Data Analysis (EDA): Visualized trends, correlations, and seasonal patterns.

Anomaly Detection: Applied statistical methods (3-σ rule, IQR) and unsupervised learning (IsolationForest, OneClassSVM, DBSCAN) to detect unusual energy drops caused by extreme weather.

Seasonality & Trends: Modeled daily, monthly, and yearly patterns in solar generation.

Time Series Forecasting: Implemented ARIMA and Prophet models to predict future energy output, including weather regressors.

Supervised Learning: Trained and compared multiple models — Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, and LightGBM — to predict energy production from weather features.

📊 Results

XGBoost performed best with R² ≈ 0.94 and RMSE ≈ 262, closely followed by LightGBM and Random Forest.

GHI (solar irradiance) was consistently the strongest predictor, followed by DayLength, SunlightTime, Hour, and Humidity.

Tree-based models captured nonlinear effects (e.g., cloud thresholds, rainfall impact) that linear models could not.

Anomalies made up <1% of data — small enough to exclude for training but valuable for analyzing grid reliability during extreme events.

🚀 Applications

Enhancing solar power forecasting for energy providers.

Supporting grid stability and renewable energy integration.

Identifying weather-driven risks in energy production.

Guiding sensor deployment priorities (e.g., radiation and humidity sensors).

This project demonstrates how data science and machine learning can improve renewable energy forecasting and resilience under variable weather conditions.
