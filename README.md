# Transformer Health Dashboard

A Streamlit-based dashboard for monitoring transformer parameters using sensor data.

## Live Demo: https://transformer-health-dashboard-hljxktyzowy8yyezonvzsy.streamlit.app/

## Overview

Transformer operating conditions can be monitored through parameters such as temperature, current, voltage, humidity, and load.
This project provides a visual monitoring interface where these parameters can be analyzed together to identify abnormal operating conditions and generate basic alerts.
The current version focuses on the dashboard and monitoring layer, while real-time hardware integration and machine-learning-based prediction are planned as future extensions.

## Features

- Transformer health monitoring
- Temperature and ambient temperature tracking
- Current, voltage, humidity, and load visualization
- Health score calculation
- Rule-based alerts
- Temperature trend analysis
- Prototype temperature risk prediction
- CSV upload support
- Interactive Plotly charts

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly

## Project Structure

Transformer-health-dashboard/
│
├── dashboard.py
├── sample_transformer_data.csv
├── requirements.txt
├── README.md
├── .gitignore


## Installation & Setup

1. Clone the repository
git clone https://github.com/tanusingh-30/Transformer-health-dashboard.git
2. Open the project
cd Transformer-health-dashboard
3. Install dependencies
python -m pip install -r requirements.txt
4. Run the dashboard
python -m streamlit run dashboard.py

The dashboard will open in your browser.


Author

Tanu Singh

Computer Science & Engineering

GitHub: @tanusingh-30


