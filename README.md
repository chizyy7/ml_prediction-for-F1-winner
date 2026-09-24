# F1 Championship Predictor — Machine Learning-Based 2026 Formula 1 Winner Prediction System

An end-to-end machine learning system that uses historical Formula 1 data and current 2026-season data to estimate which driver has the highest probability of winning the 2026 Formula 1 Drivers' Championship.

## Project Overview

This professional portfolio project implements a sophisticated ML pipeline for predicting F1 championship outcomes. The system:

- Collects historical F1 data from the Jolpica F1 API (Ergast-compatible)
- Engineers meaningful features from driver, constructor, qualifying, race, and circuit data
- Implements time-aware validation to prevent data leakage
- Trains and compares multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, etc.)
- Uses Monte Carlo simulation to estimate championship-winning probabilities
- Provides explainability through SHAP values and feature importance
- Delivers an interactive Streamlit dashboard for exploration and prediction
- Automatically updates when new 2026 race data becomes available

## Key Features

✅ **Chronological Validation**: Walk-forward validation simulating real-world prediction scenarios
✅ **Feature Engineering**: 50+ features including recent form, teammate comparison, circuit performance
✅ **Model Comparison**: Multiple algorithms evaluated with proper statistical metrics
✅ **Uncertainty Quantification**: Monte Carlo simulations provide probability distributions
✅ **Explainable AI**: SHAP analysis shows why drivers receive specific championship probabilities
✅ **Live Updates**: Automated data refresh pipeline for ongoing 2026 season
✅ **Professional Dashboard**: Streamlit interface with F1-inspired design
✅ **Production-Ready**: Modular code, type hints, logging, error handling, and unit tests

## Technology Stack

- **Language**: Python 3.9+
- **Data Processing**: pandas, numpy
- **ML Libraries**: scikit-learn, xgboost, lightgbm, catboost
- **Explainability**: SHAP
- **Visualization**: matplotlib, plotly
- **Dashboard**: Streamlit
- **API Access**: requests
- **Utilities**: joblib, scipy, typing, logging

## Project Structure

```
f1-championship-predictor/
├── data/
│   ├── raw/                  # Raw data downloaded from API
│   └── processed/            # Cleaned and feature-engineered data
├── notebooks/                # Jupyter notebooks for exploration
├── src/
│   ├── data_loader.py        # API data ingestion
│   ├── data_cleaning.py      # Data validation and cleaning
│   ├── feature_engineering.py # Feature creation
│   ├── models.py             # Model training and prediction
│   ├── evaluation.py         # Model validation and metrics
│   ├── simulation.py         # Monte Carlo championship simulation
│   ├── explainability.py     # SHAP and feature importance
│   └── prediction.py         # Main prediction pipeline
├── app/
│   └── streamlit_app.py      # Streamlit dashboard
├── models/                   # Saved trained models
├── tests/                    # Unit tests
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/f1-championship-predictor.git
   cd f1-championship-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the data pipeline:
   ```bash
   python src/data_loader.py --init
   ```

4. Run the dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Usage

The system provides championship-winning probabilities for all eligible 2026 drivers based on historical performance and current season data available up to the most recent completed race.

**Important**: Predictions are estimates based on available data and should not be interpreted as guaranteed outcomes. The model explicitly quantifies uncertainty through probability distributions and confidence intervals.

## Results

[To be populated after model training and validation]

## Limitations

- Small sample size (~20 races per season)
- Changing regulations and technical specifications between seasons
- Driver transfers and team changes
- Unpredictable events (weather, safety cars, mechanical failures)
- Concept drift in performance patterns

## Future Improvements

- Incorporate qualifying session times and sector data
- Add weather and track condition features
- Implement driver-specific circuit preferences
- Add tire degradation and stint analysis
- Incorporate betting market odds as features
- Real-time lap time processing during races

## License

MIT License

## Acknowledgements

Data sourced from the Jolpica F1 API (Ergast-compatible), providing historical Formula 1 statistics.
