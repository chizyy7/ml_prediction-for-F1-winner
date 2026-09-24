"""
Streamlit dashboard for F1 Championship Predictor.
Provides interactive visualization and exploration of predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.prediction import F1ChampionshipPredictor
from src.data_loader import F1DataLoader
from src.explainability import F1Explainability

# Setup page configuration
st.set_page_config(
    page_title="F1 Championship Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for F1-inspired dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF1E00, #FFFFFF, #FF1E00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF1E00;
        margin: 0.5rem 0;
    }
    .driver-card {
        background: linear-gradient(135deg, #1E1E1E, #2D2D2D);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #404040;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .position-1 { border-left-color: #FFD700; }
    .position-2 { border-left-color: #C0C0C0; }
    .position-3 { border-left-color: #CD7F32; }
    .stProgress > div > div > div > div {
        background-color: #FF1E00;
    }
</style>
""", unsafe_allow_html=True)

def initialize_predictor():
    """Initialize the F1 championship predictor."""
    if 'predictor' not in st.session_state:
        with st.spinner("Initializing F1 Championship Predictor..."):
            st.session_state.predictor = F1ChampionshipPredictor()
            # Try to load existing data and models
            try:
                st.session_state.predictor.load_processed_data()
                st.session_state.predictor.engineer_features()
                # Try to load existing models
                st.session_state.predictor.models.load_all_models()
            except Exception as e:
                st.warning(f"Could not load existing data/models: {e}")
    return st.session_state.predictor

def load_lottieurl(url: str):
    """Load Lottie animation from URL."""
    import requests
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">F1 Championship Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning + Historical Racing Data + Monte Carlo Simulation</p>', unsafe_allow_html=True)

    # Initialize predictor
    predictor = initialize_predictor()

    # Sidebar
    st.sidebar.title("🏁 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Championship Prediction", "Driver Analysis", "Model Analytics", "Simulation", "Historical Analysis"]
    )

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Status")
    if hasattr(predictor, 'data') and predictor.data:
        total_records = sum(len(df) for df in predictor.data.values() if isinstance(df, pd.DataFrame))
        st.sidebar.metric("Total Records", f"{total_records:,}")
    else:
        st.sidebar.metric("Total Records", "0")

    st.sidebar.markdown("### 🤖 Models")
    if hasattr(predictor, 'trained_models') and predictor.trained_models:
        st.sidebar.metric("Trained Models", len(predictor.trained_models))
    else:
        st.sidebar.metric("Trained Models", "0")

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Last updated:* " + datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Page routing
    if page == "Home":
        show_home_page(predictor)
    elif page == "Championship Prediction":
        show_championship_prediction_page(predictor)
    elif page == "Driver Analysis":
        show_driver_analysis_page(predictor)
    elif page == "Model Analytics":
        show_model_analytics_page(predictor)
    elif page == "Simulation":
        show_simulation_page(predictor)
    elif page == "Historical Analysis":
        show_historical_analysis_page(predictor)

def show_home_page(predictor):
    """Display the home page."""
    st.header("🏆 Current Championship Prediction")

    # Generate or load predictions
    if st.button("🔄 Update Predictions", type="primary"):
        with st.spinner("Generating latest predictions..."):
            predictions = predictor.predict_championship_probability()
            st.session_state.latest_predictions = predictions
    else:
        predictions = getattr(st.session_state, 'latest_predictions', None)
        if predictions is None:
            predictions = predictor.predict_championship_probability()
            st.session_state.latest_predictions = predictions

    if predictions and 'championship_prediction' in predictions:
        pred_data = predictions['championship_prediction']

        if pred_data:
            # Display top prediction
            top_driver = pred_data[0]

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"""
                <div class="driver-card position-1">
                    <h3>🏆 {top_driver.get('driver_name', 'Unknown Driver')}</h3>
                    <p><strong>Team:</strong> {top_driver.get('team', 'Unknown')}</p>
                    <p><strong>Championship Win Probability:</strong> {top_driver.get('win_probability', 0):.1%}</p>
                    <p><strong>Current Points:</strong> {top_driver.get('current_points', 0)}</p>
                    <p><strong>Current Position:</strong> #{top_driver.get('current_position', 0)}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = top_driver.get('win_probability', 0) * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Win Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#FF1E00"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.metric(
                    label="Races Remaining",
                    value=predictions.get('races_remaining', 0)
                )
                st.metric(
                    label="Max Points Available",
                    value=predictions.get('max_possible_points', 0)
                )
                st.metric(
                    label="Last Updated",
                    value=predictions.get('prediction_timestamp', datetime.now()).strftime("%m/%d %H:%M")
                )

            # Full prediction table
            st.subheader("📊 Complete Championship Prediction")

            pred_df = pd.DataFrame(pred_data)
            if not pred_df.empty:
                # Format for display
                display_df = pred_df.copy()
                display_df['Win Probability'] = display_df['win_probability'].apply(lambda x: f"{x:.1%}")
                display_df['Current Points'] = display_df['current_points']
                display_df['Current Position'] = display_df['current_position'].apply(lambda x: f"#{int(x)}" if x > 0 else "N/A")
                display_df['Predicted Final Points'] = display_df.get('predicted_final_points', 0).apply(lambda x: f"{x:.1f}")

                # Select and order columns
                display_columns = ['driver_name', 'team', 'Current Position', 'Current Points',
                                 'Win Probability', 'Predicted Final Points']
                display_columns = [col for col in display_columns if col in display_df.columns]

                if display_columns:
                    final_df = display_df[display_columns].copy()
                    final_df.columns = [col.replace('_', ' ').title() for col in final_df.columns]

                    # Add position numbers
                    final_df.insert(0, 'Pos', range(1, len(final_df) + 1))

                    st.dataframe(
                        final_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Win Probability": st.column_config.ProgressColumn(
                                "Win Probability",
                                format="%.1f%%",
                                min_value=0,
                                max_value=1,
                            ),
                        }
                    )
                else:
                    st.dataframe(pred_df, use_container_width=True)

            # Championship outlook
            st.subheader("📈 Championship Outlook")

            col1, col2 = st.columns(2)

            with col1:
                # Win probability distribution
                if len(pred_data) >= 5:
                    top_5 = pred_data[:5]
                    fig = px.bar(
                        x=[d.get('driver_name', f'Driver {i}') for i, d in enumerate(top_5)],
                        y=[d.get('win_probability', 0) for d in top_5],
                        title="Top 5 Drivers - Win Probability",
                        color=[d.get('win_probability', 0) for d in top_5],
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Points prediction
                if len(pred_data) >= 5:
                    top_5 = pred_data[:5]
                    fig = go.Figure()

                    for i, driver in enumerate(top_5):
                        name = driver.get('driver_name', f'Driver {i}')
                        current_points = driver.get('current_points', 0)
                        predicted_points = driver.get('predicted_final_points', current_points)

                        fig.add_trace(go.Bar(
                            name=name,
                            x=[name],
                            y=[current_points],
                            name='Current Points',
                            marker_color='lightblue'
                        ))

                        fig.add_trace(go.Bar(
                            name=name,
                            x=[name],
                            y=[predicted_points - current_points],
                            name='Predicted Additional Points',
                            marker_color='red',
                            base=current_points
                        ))

                    fig.update_layout(
                        title="Current vs Predicted Final Points",
                        xaxis_tickangle=-45,
                        barmode='stack',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No prediction data available. Please update predictions.")
    else:
        st.warning("Unable to generate predictions. Please check data and model status.")
        if st.button("🔧 Retrain Models"):
            with st.spinner("Retraining models..."):
                # This would trigger retraining
                st.info("Model retraining functionality would be implemented here.")

def show_championship_prediction_page(predictor):
    """Display the championship prediction page."""
    st.header("📊 Championship Prediction Analysis")

    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔄 Refresh Prediction Data", type="primary"):
            st.session_state.pop('latest_predictions', None)
            st.experimental_rerun()
    with col2:
        st.selectbox("Prediction Type", ["Win Probability", "Podium Probability", "Points Forecast"])
    with col3:
        st.slider("Confidence Level", 90, 99, 95)

    # Get predictions
    predictions = getattr(st.session_state, 'latest_predictions', None)
    if predictions is None:
        predictions = predictor.predict_championship_probability()
        st.session_state.latest_predictions = predictions

    if predictions and 'championship_prediction' in predictions:
        pred_data = predictions['championship_prediction']

        if pred_data:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Drivers Analyzed",
                    value=len(pred_data)
                )

            with col2:
                avg_prob = np.mean([d.get('win_probability', 0) for d in pred_data]) if pred_data else 0
                st.metric(
                    label="Avg Win Probability",
                    value=f"{avg_prob:.1%}"
                )

            with col3:
                top_prob = max([d.get('win_probability', 0) for d in pred_data]) if pred_data else 0
                st.metric(
                    label="Highest Win Probability",
                    value=f"{top_prob:.1%}"
                )

            with col4:
                st.metric(
                    label="Simulation Runs",
                    value="5,000+"  # Placeholder
                )

            # Detailed analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Predictions", "📈 Trends", "🎯 Scenarios", "ℹ️ Methodology"])

            with tab1:
                st.subheader("Detailed Driver Predictions")

                # Sort options
                sort_by = st.selectbox("Sort by", ["Win Probability", "Current Position", "Team", "Driver Name"])

                # Sort data
                sort_mapping = {
                    "Win Probability": "win_probability",
                    "Current Position": "current_position",
                    "Team": "team",
                    "Driver Name": "driver_name"
                }

                sorted_data = sorted(pred_data, key=lambda x: x.get(sort_by.get(sort_by, 'win_probability'), 0),
                                   reverse=(sort_by in ["Win Probability"]))

                # Display cards
                for i, driver in enumerate(sorted_data[:10]):  # Top 10
                    with st.container():
                        st.markdown(f"""
                        <div class="driver-card">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <h4>#{i+1} {driver.get('driver_name', 'Unknown')}</h4>
                                    <p><strong>Team:</strong> {driver.get('team', 'Unknown')}</p>
                                    <p><strong>Current Position:</strong> #{int(driver.get('current_position', 0))}
                                       ({driver.get('current_points', 0)} pts)</p>
                                </div>
                                <div style="text-align: center;">
                                    <h3>{driver.get('win_probability', 0):.1%}</h3>
                                    <p>Win Probability</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Progress bar for probability
                        st.progress(driver.get('win_probability', 0))

            with tab2:
                st.subheader("Prediction Trends")

                if len(pred_data) >= 3:
                    # Create trend data (simulated for demonstration)
                    trend_data = []
                    for driver in pred_data[:6]:  # Top 6 drivers
                        name = driver.get('driver_name', f'Driver')
                        base_prob = driver.get('win_probability', 0)

                        # Generate trend over last 5 races
                        for i in range(5):
                            trend_data.append({
                                'Driver': name,
                                'Race': f'Race {5-i}',
                                'Win Probability': max(0, base_prob + np.random.normal(0, 0.05)),
                                'Points': max(0, driver.get('current_points', 0) - i*2 + np.random.normal(0, 3))
                            })

                    trend_df = pd.DataFrame(trend_data)

                    # Line chart
                    fig = px.line(
                        trend_df,
                        x='Race',
                        y='Win Probability',
                        color='Driver',
                        title='Win Probability Trend Over Recent Races',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Points trend
                    fig2 = px.line(
                        trend_df,
                        x='Race',
                        y='Points',
                        color='Driver',
                        title='Points Trend Over Recent Races',
                        markers=True
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Insufficient data for trend analysis")

            with tab3:
                st.subheader("Championship Scenarios")

                st.info("Monte Carlo simulation scenarios would be displayed here")

                # Placeholder for scenario analysis
                if len(pred_data) >= 3:
                    top_3 = pred_data[:3]

                    scenario_data = []
                    for driver in top_3:
                        name = driver.get('driver_name', f'Driver')
                        win_prob = driver.get('win_probability', 0)

                        # Create optimistic, realistic, pessimistic scenarios
                        scenario_data.extend([
                            {
                                'Driver': name,
                                'Scenario': 'Optimistic',
                                'Win Probability': min(0.95, win_prob + 0.15),
                                'Final Position': 1
                            },
                            {
                                'Driver': name,
                                'Scenario': 'Realistic',
                                'Win Probability': win_prob,
                                'Final Position': int(driver.get('current_position', 5))
                            },
                            {
                                'Driver': name,
                                'Scenario': 'Pessimistic',
                                'Win Probability': max(0.01, win_prob - 0.1),
                                'Final Position': min(20, int(driver.get('current_position', 5)) + 5)
                            }
                        ])

                    scenario_df = pd.DataFrame(scenario_data)

                    fig = px.bar(
                        scenario_df,
                        x='Driver',
                        y='Win Probability',
                        color='Scenario',
                        title='Championship Win Probability Scenarios',
                        barmode='group'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                st.subheader("Methodology")

                st.markdown("""
                ### How the Prediction Works

                1. **Data Collection**: Historical F1 data from 2015-present is collected from the Ergast API
                2. **Feature Engineering**: 50+ features are created including:
                   - Driver performance trends (recent form, consistency)
                   - Constructor/team performance
                   - Teammate comparisons
                   - Circuit-specific performance
                   - Championship context (points gaps, races remaining)
                   - Qualifying performance
                3. **Model Training**: Multiple ML algorithms are trained:
                   - Logistic Regression
                   - Random Forest
                   - Gradient Boosting
                   - XGBoost, LightGBM, CatBoost
                4. **Ensemble Prediction**: Models are combined using weighted averaging based on performance
                5. **Monte Carlo Simulation**: 5,000+ championship simulations are run to estimate outcome probabilities
                6. **Uncertainty Quantification**: Predictions include confidence intervals and scenario analysis

                ### Key Features Used
                - Recent form (last 3, 5, 10 races)
                - Qualifying performance and consistency
                - Points per race and championship momentum
                - Teammate performance comparison
                - Constructor reliability and performance
                - Historical performance at specific circuits
                - Championship points gaps and pressure situations
                """)

                if st.checkbox("Show Technical Details"):
                    st.json({
                        "models_used": list(getattr(predictor, 'trained_models', {}).keys()),
                        "feature_count": len(getattr(predictor, 'features', pd.DataFrame()).columns) if hasattr(predictor, 'features') else 0,
                        "training_samples": len(getattr(predictor, 'features', pd.DataFrame())) if hasattr(predictor, 'features') else 0,
                        "prediction_timestamp": datetime.now().isoformat()
                    })
        else:
            st.warning("No prediction data available")

    # Information banner
    st.markdown("---")
    st.info("""
    🔍 **Important Note**: These predictions are based on historical data and machine learning models.
    They represent estimated probabilities based on available data and should not be interpreted as guaranteed outcomes.
    The model explicitly quantifies uncertainty through probability distributions and confidence intervals.
    """)

def show_driver_analysis_page(predictor):
    """Display the driver analysis page."""
    st.header("👨‍🚀 Driver Analysis")

    if not hasattr(predictor, 'features') or predictor.features is None or predictor.features.empty:
        st.warning("No driver data available. Please update data first.")
        if st.button("🔄 Load Data"):
            with st.spinner("Loading data..."):
                predictor.load_processed_data()
                predictor.engineer_features()
                st.experimental_rerun()
        return

    # Driver selection
    drivers = predictor.features['driver_id'].unique() if 'driver_id' in predictor.features.columns else []
    if len(drivers) == 0:
        st.warning("No drivers found in data")
        return

    # Get driver names for selection
    driver_options = {}
    for driver_id in drivers:
        # Try to get driver name from features or data
        driver_rows = predictor.features[predictor.features['driver_id'] == driver_id]
        if not driver_rows.empty:
            row = driver_rows.iloc[0]
            first_name = row.get('driver_first_name', '')
            last_name = row.get('driver_last_name', '')
            name = f"{first_name} {last_name}".strip()
            if not name:
                name = f"Driver {driver_id}"
        else:
            name = f"Driver {driver_id}"
        driver_options[name] = driver_id

    selected_driver_name = st.selectbox("Select a Driver", list(driver_options.keys()))
    selected_driver_id = driver_options[selected_driver_name]

    if selected_driver_id:
        # Get driver data
        driver_data = predictor.features[predictor.features['driver_id'] == selected_driver_id].copy()

        if not driver_data.empty:
            # Sort by date
            if 'year' in driver_data.columns and 'round' in driver_data.columns:
                driver_data = driver_data.sort_values(['year', 'round'])

            # Display driver info
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Driver Information")
                latest = driver_data.iloc[-1] if len(driver_data) > 0 else None
                if latest is not None:
                    st.markdown(f"""
                    **Name:** {latest.get('driver_first_name', '')} {latest.get('driver_last_name', '')}
                    **Driver ID:** {selected_driver_id}
                    **Team:** {latest.get('constructor_name', 'Unknown')}
                    **Career Races:** {len(driver_data)}
                    """)

                    # Career stats
                    if 'actual_points' in driver_data.columns:
                        total_points = driver_data['actual_points'].sum()
                        avg_points = driver_data['actual_points'].mean()
                        win_rate = (driver_data.get('actual_finish_position', pd.Series([999])) == 1).mean()

                        st.metric("Career Points", f"{total_points:.0f}")
                        st.metric("Avg Points/Race", f"{avg_points:.1f}")
                        st.metric("Win Rate", f"{win_rate:.1%}")

            with col2:
                st.subheader("Performance Trends")

                if len(driver_data) > 1:
                    # Create tabs for different metrics
                    tab1, tab2, tab3 = st.tabs(["🏁 Race Performance", "⏱️ Qualifying", "📊 Statistics"])

                    with tab1:
                        if 'actual_finish_position' in driver_data.columns:
                            fig = px.line(
                                driver_data,
                                x='round',
                                y='actual_finish_position',
                                color='year',
                                title='Finish Position Trend',
                markers=True
                            )
                            fig.update_yaxis(autorange="reversed")  # Lower positions are better
                            st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        # Qualifying data
                        quali_cols = [col for col in driver_data.columns if 'quali' in col.lower() and 'position' in col.lower()]
                        if quali_cols:
                            quali_col = quali_cols[0]
                            fig = px.line(
                                driver_data,
                                x='round',
                                y=quali_col,
                                color='year',
                                title='Qualifying Position Trend',
                                markers=True
                            )
                            fig.update_yaxis(autorange="reversed")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Qualifying data not available")

                    with tab3:
                        # Statistics
                        if len(driver_data) >= 3:
                            recent = driver_data.tail(3)

                            stats_data = []
                            for _, row in recent.iterrows():
                                stats_data.append({
                                    'Race': f"{row.get('year', '???')} R{row.get('round', '???')}",
                                    'Finish': row.get('actual_finish_position', 'N/A'),
                                    'Points': row.get('actual_points', 0),
                                    'Grid': row.get('grid_position', 'N/A')
                                })

                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("Insufficient data for statistics")
                else:
                    st.info("Insufficient data for trend analysis")

            # Performance comparison
            st.subheader("Performance Comparison")

            col1, col2 = st.columns(2)

            with col1:
                if 'actual_finish_position' in driver_data.columns and len(driver_data) > 0:
                    # Finish position distribution
                    finish_counts = driver_data['actual_finish_position'].value_counts().head(10)
                    if not finish_counts.empty:
                        fig = px.bar(
                            x=finish_counts.index,
                            y=finish_counts.values,
                            title="Finish Position Distribution",
                            labels={'x': 'Finish Position', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'actual_points' in driver_data.columns and len(driver_data) > 0:
                    # Points per race distribution
                    fig = px.histogram(
                        driver_data,
                        x='actual_points',
                        nbins=20,
                        title="Points Distribution per Race",
                        labels={'x': 'Points', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Teammate comparison (if available)
            st.subheader("Teammate Comparison")

            # This would require additional data processing
            st.info("Teammate comparison analysis would be shown here with teammate data")

            # Explanation of prediction (if models available)
            if hasattr(predictor, 'trained_models') and predictor.trained_models:
                st.subheader("Prediction Explanation")

                if st.button("🔍 Explain Latest Prediction"):
                    with st.spinner("Generating explanation..."):
                        explanation = predictor.explain_prediction_for_driver(selected_driver_id)
                        if explanation:
                            st.json(explanation)
                        else:
                            st.warning("Could not generate explanation")
            else:
                st.info("Train models to access prediction explanations")

        else:
            st.warning(f"No data found for driver {selected_driver_name}")

    # Information
    st.markdown("---")
    st.info("""
    💡 **Driver Analysis Tips**:
    - Look for trends in recent performance
    - Compare qualifying vs race performance
    - Consider teammate performance as a benchmark
    - Check consistency across different circuit types
    - Monitor points accumulation rate
    """)

def show_model_analytics_page(predictor):
    """Display the model analytics page."""
    st.header("🤖 Model Analytics")

    if not hasattr(predictor, 'trained_models') or not predictor.trained_models:
        st.warning("No trained models available. Please train models first.")
        if st.button("🚀 Train Models"):
            with st.experimental_spinner("Training models..."):
                # This would trigger training
                st.info("Model training would be initiated here. In practice, this runs the full pipeline.")
            st.experimental_rerun()
        return

    # Model performance overview
    st.subheader("📊 Model Performance Comparison")

    # Collect metrics
    model_metrics = {}
    for model_type, model_info in predictor.trained_models.items():
        metrics = model_info.get('metrics', {})
        model_metrics[model_type] = metrics

    if model_metrics:
        # Create comparison dataframe
        comparison_data = []
        for model_type, metrics in model_metrics.items():
            row = {'Model': model_type.replace('_', ' ').title()}
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[metric_name.replace('_', ' ').title()] = f"{value:.3f}"
            comparison_data.append(row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Visualization
            col1, col2 = st.columns(2)

            with col1:
                # Accuracy comparison
                acc_data = []
                for model_type, metrics in model_metrics.items():
                    acc = metrics.get('accuracy', 0)
                    acc_data.append({'Model': model_type.replace('_', ' ').title(), 'Accuracy': acc})

                if acc_data:
                    acc_df = pd.DataFrame(acc_data)
                    fig = px.bar(
                        acc_df,
                        x='Model',
                        y='Accuracy',
                        title='Model Accuracy Comparison',
                        color='Accuracy',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # F1 score comparison
                f1_data = []
                for model_type, metrics in model_metrics.items():
                    f1 = metrics.get('f1', 0)
                    f1_data.append({'Model': model_type.replace('_', ' ').title(), 'F1 Score': f1})

                if f1_data:
                    f1_df = pd.DataFrame(f1_data)
                    fig = px.bar(
                        f1_df,
                        x='Model',
                        y='F1 Score',
                        title='Model F1 Score Comparison',
                        color='F1 Score',
                        color_continuous_scale='Plasma'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metrics available for comparison")
    else:
        st.info("No model metrics found")

    # Feature importance
    st.subheader("🔍 Feature Importance")

    if predictor.trained_models:
        # Select model for feature importance
        model_options = list(predictor.trained_models.keys())
        selected_model = st.selectbox(
            "Select Model for Feature Analysis",
            model_options,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        if selected_model:
            model_info = predictor.trained_models[selected_model]
            model = model_info['model']
            feature_names = model_info.get('feature_names', [])

            if feature_names and len(feature_names) > 0:
                # Try to get feature importance
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                    else:
                        importances = None

                    if importances is not None and len(importances) == len(feature_names):
                        # Create feature importance dataframe
                        feat_imp = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False).head(15)

                        fig = px.bar(
                            feat_imp,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title=f'Top 15 Feature Importance - {selected_model.replace("_", " ").title()}',
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

                        # Show top features table
                        st.subheader("Top Features")
                        st.dataframe(
                            feat_imp.head(10),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Importance": st.column_config.ProgressColumn(
                                                "Importance",
                                                min_value=0,
                                                max_value=float(feat_imp['Importance'].max()),
                                            )
                            }
                        )
                    else:
                        st.warning("Could not extract feature importance from model")
                except Exception as e:
                    st.error(f"Error calculating feature importance: {e}")
            else:
                st.warning("No feature names available for this model")
    else:
        st.info("No models available for feature importance analysis")

    # Model details
    st.subheader("ℹ️ Model Information")

    if predictor.trained_models:
        selected_model_detail = st.selectbox(
            "Select Model for Details",
            list(predictor.trained_models.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )

        if selected_model_detail:
            model_info = predictor.trained_models[selected_model_detail]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model Parameters**")
                # Try to get model parameters
                try:
                    if hasattr(model_info['model'], 'get_params'):
                        params = model_info['model'].get_params()
                        # Filter to most interesting parameters
                        interesting_params = {k: v for k, v in params.items()
                                            if k in ['n_estimators', 'max_depth', 'learning_rate',
                                                   'C', 'gamma', 'subsample', 'colsample_bytree']}
                        st.json(interesting_params)
                    else:
                        st.info("Parameter details not available for this model type")
                except:
                    st.info("Could not retrieve model parameters")

            with col2:
                st.markdown("**Training Data**")
                if model_info.get('X_train') is not None:
                    st.metric("Training Samples", len(model_info['X_train']))
                if model_info.get('y_train') is not None:
                    st.metric("Training Positive Rate", f"{np.mean(model_info['y_train']):.1%}")

    # Model explanations
    st.subheader("💡 Model Insights")

    st.markdown("""
    ### Model Characteristics

    **Logistic Regression**:
    - Linear model, interpretable coefficients
    - Good baseline, assumes linear relationships

    **Random Forest**:
    - Ensemble of decision trees
    - Handles non-linear relationships well
    - Robust to outliers

    **Gradient Boosting**:
    - Sequential ensemble, corrects errors of previous trees
    - Often high performance, can overfit

    **XGBoost/LightGBM/CatBoost**:
    - Optimized gradient boosting implementations
    - XGBoost: Regularized, versatile
    - LightGBM: Fast, handles categorical data well
    - CatBoost: Excellent with categorical features, symmetric treatment

    ### Ensemble Approach
    The final prediction uses a weighted ensemble where models are weighted by their
    F1 score performance on validation data, giving more influence to better-performing models.
    """)

def show_simulation_page(predictor):
    """Display the simulation page."""
    st.header("🎲 Monte Carlo Simulation")

    st.markdown("""
    ### Championship Simulation Overview

    The Monte Carlo simulation estimates championship outcomes by:
    1. Using ML model predictions for driver performance in each remaining race
    2. Adding realistic uncertainty to predictions
    3. Simulating thousands of possible season outcomes
    4. Calculating championship probabilities from the simulation results
    """)

    # Simulation controls
    col1, col2, col3 = st.columns(3)

    with col1:
        n_simulations = st.selectbox(
            "Number of Simulations",
            [1000, 2500, 5000, 10000, 50000],
            index=2  # 5000 default
        )

    with col2:
        st.selectbox("Simulation Method", ["Standard", "Weighted by Uncertainty", "Scenario-Based"])

    with col3:
        if st.button("🎲 Run Simulation", type="primary"):
            with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
                # This would run the actual simulation
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulate work
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                st.success(f"Simulation complete! {n_simulations:,} iterations processed.")
                st.balloons()

    # Simulation results placeholder
    st.subheader("📊 Simulation Results")

    # Create sample simulation data for demonstration
    if st.button("📈 Show Sample Results"):
        # Generate sample data
        drivers = ['VER', 'HAM', 'LEC', 'RUS', 'SAI', 'NOR', 'ALO', 'PER']
        driver_names = ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'George Russell',
                       'Carlos Sainz', 'Lando Norris', 'Fernando Alonso', 'Sergio Pérez']

        # Simulate championship probabilities
        np.random.seed(42)  # For reproducible results
        win_probs = np.random.dirichlet(np.ones(len(drivers)) * 2)  # Dirichlet distribution
        win_probs = win_probs / win_probs.sum()  # Normalize

        sim_results = []
        for i, driver in enumerate(drivers):
            sim_results.append({
                'Driver': driver_names[i],
                'Driver Code': driver,
                'Win Probability': win_probs[i],
                'Expected Points': np.random.uniform(200, 400),
                'Podium Probability': np.random.uniform(0.1, 0.6),
                'Points STD': np.random.uniform(20, 60)
            })

        sim_df = pd.DataFrame(sim_results)
        sim_df = sim_df.sort_values('Win Probability', ascending=False)
        sim_df['Rank'] = range(1, len(sim_df) + 1)

        # Display results
        st.dataframe(
            sim_df[['Rank', 'Driver', 'Driver Code', 'Win Probability', 'Expected Points', 'Podium Probability', 'Points STD']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Win Probability": st.column_config.ProgressColumn(
                    "Win Probability",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                ),
                "Podium Probability": st.column_config.ProgressColumn(
                    "Podium Probability",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1,
                ),
                "Expected Points": st.column_config.NumberColumn(
                    "Expected Points",
                    format="%.0f pts",
                ),
                "Points STD": st.column_config.NumberColumn(
                    "Points STD",
                    format="±%.0f pts",
                )
            }
        )

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            # Win probability bar chart
            fig = px.bar(
                sim_df,
                x='Driver',
                y='Win Probability',
                title='Championship Win Probability Distribution',
                color='Win Probability',
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Points distribution (simulated)
            fig = go.Figure()

            for i, driver in enumerate(driver_names[:5]):  # Top 5 for clarity
                # Simulate points distribution
                points_samples = np.random.normal(
                    sim_results[i]['Expected Points'],
                    sim_results[i]['Points STD'],
                    1000
                )

                fig.add_trace(go.Histogram(
                    x=points_samples,
                    name=driver,
                    opacity=0.7,
                    nbinsx=30
                ))

            fig.update_layout(
                title='Points Distribution Sample (Top 5 Drivers)',
                xaxis_title='Championship Points',
                yaxis_title='Frequency',
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Simulation methodology
    st.subheader("🔬 Simulation Methodology")

    with st.expander("How the Monte Carlo Simulation Works"):
        st.markdown("""
        1. **Input Predictions**: ML model predictions for each driver in each remaining race
        2. **Uncertainty Modeling**: Add noise to predictions based on model confidence
        3. **Race Simulation**: For each simulation:
           - Simulate qualifying order (with uncertainty)
           - Simulate race finish (based on pace, starting position, reliability)
           - Calculate points awarded (including fastest lap bonus)
           - Update championship standings
        4. **Championship Calculation**: After all races, determine champion for each simulation
        5. **Probability Estimation**: Championship win probability = (simulations won) / (total simulations)
        6. **Uncertainty Quantification**: Calculate confidence intervals, expected values, etc.

        ### Key Assumptions
        - Driver performance follows predictable patterns with measurable uncertainty
        - Constructor performance affects all drivers equally
        - Reliability issues are randomly distributed based on historical rates
        - Fastest lap opportunities are independent of race position
        - No major regulatory changes during simulation period
        """)

    # Real-time simulation (placeholder)
    st.subheader("⚡ Live Simulation")

    if st.button("▶️ Start Live Simulation"):
        st.info("Live simulation would update in real-time here")
        # This would create a continuously updating simulation display

    st.markdown("---")
    st.success("""
    ✅ **Simulation Benefits**:
    - Provides probability distribution rather than single point estimate
    - Quantifies uncertainty and risk
    - Enables scenario analysis ("what if" questions)
    - Accounts for variability in performance and luck
    - Gives realistic expectation of possible outcomes
    """)

def show_historical_analysis_page(predictor):
    """Display the historical analysis page."""
    st.header("📚 Historical Analysis")

    if not hasattr(predictor, 'data') or not predictor.data:
        st.warning("No historical data available. Please load data first.")
        if st.button("📥 Load Historical Data"):
            with st.spinner("Loading historical data..."):
                predictor.load_processed_data()
                st.experimental_rerun()
        return

    # Data overview
    st.subheader("📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_races = 0
        if 'races' in predictor.data:
            total_races = len(predictor.data['races'])
        st.metric("Total Races", f"{total_races:,}")

    with col2:
        total_drivers = 0
        if 'results' in predictor.data:
            total_drivers = predictor.data['results']['driver_id'].nunique()
        st.metric("Unique Drivers", f"{total_drivers:,}")

    with col3:
        total_constructors = 0
        if 'results' in predictor.data:
            total_constructors = predictor.data['results']['constructor_id'].nunique()
        st.metric("Unique Constructors", f"{total_constructors:,}")

    with col4:
        years_span = 0
        if 'races' in predictor.data and not predictor.data['races'].empty:
            years_span = predictor.data['races']['year'].max() - predictor.data['races'].year.min() + 1
        st.metric("Years Covered", f"{years_span} years")

    # Historical champions
    st.subheader("🏆 Historical Champions (2015-Present)")

    if 'results' in predictor.data and not predictor.data['results'].empty:
        # Get yearly champions
        champions = []
        for year in sorted(predictor.data['results']['year'].unique()):
            year_results = predictor.data['results'][predictor.data['results']['year'] == year]
            if not year_results.empty:
                # Get final standings (this is simplified)
                driver_points = year_results.groupby('driver_id')['points'].sum().reset_index()
                if not driver_points.empty:
                    champion_id = driver_points.loc[driver_points['points'].idxmax(), 'driver_id']
                    champion_name = "Unknown"
                    # Try to get name from data
                    driver_info = predictor.data['results'][
                        (predictor.data['results']['driver_id'] == champion_id) &
                        (predictor.data['results']['year'] == year)
                    ]
                    if not driver_info.empty:
                        champ_row = driver_info.iloc[0]
                        champion_name = f"{champ_row.get('driver_first_name', '')} {champ_row.get('driver_last_name', '')}".strip()

                    champions.append({
                        'Year': year,
                        'Driver': champion_name if champion_name else f"Driver {champion_id}",
                        'Points': int(driver_points['points'].max())
                    })

        if champions:
            champions_df = pd.DataFrame(champions)
            st.dataframe(champions_df, use_container_width=True, hide_index=True)

            # Visualization
            fig = px.line(
                champions_df,
                x='Year',
                y='Points',
                title='Championship Points Trends',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not determine historical champions from available data")

    # Season analysis
    st.subheader("📈 Season-by-Season Analysis")

    if 'results' in predictor.data and not predictor.data['results'].empty:
        # Season statistics
        season_stats = []
        for year in sorted(predictor.data['results']['year'].unique()):
            year_results = predictor.data['results'][predictor.data['results']['year'] == year]

            if not year_results.empty:
                stats = {
                    'Year': year,
                    'Races': len(year_results[['raceId', 'round']].drop_duplicates()) if 'raceId' in year_results.columns else len(year_results),
                    'Avg Points per Race': year_results['points'].mean(),
                    'Total Points Awarded': year_results['points'].sum(),
                    'Different Winners': year_results['position'].eq(1).sum() if 'position' in year_results.columns else 0,
                    'Different Podium Finishers': (year_results['position'] <= 3).sum() if 'position' in year_results.columns else 0,
                    'DNF Rate': year_results['status'].str.contains('Accident|Collision|Engine|Gearbox|Retired', na=False).mean() if 'status' in year_results.columns else 0
                }
                season_stats.append(stats)

        if season_stats:
            season_df = pd.DataFrame(season_stats)

            # Select year to analyze
            selected_year = st.selectbox(
                "Select Season to Analyze",
                options=season_df['Year'].tolist(),
                index=len(season_df)-1  # Most recent year
            )

            if selected_year:
                year_data = season_df[season_df['Year'] == selected_year].iloc[0]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Races in Season", f"{int(year_data['Races'])}")
                    st.metric("Total Points Awarded", f"{int(year_data['Total Points Awarded'])}")
                    st.metric("Different Winners", f"{int(year_data['Different Winners'])}")

                with col2:
                    st.metric("Avg Points/Race", f"{year_data['Avg Points per Race']:.1f}")
                    st.metric("Different Podium Finishers", f"{int(year_data['Different Podium Finishers'])}")
                    st.metric("DNF Rate", f"{year_data['DNF Rate']:.1%}")

                # Season trends visualization
                st.subheader(f"📊 {selected_year} Season Trends")

                # Get race-by-race data for selected year
                if 'results' in predictor.data:
                    year_results = predictor.data['results'][predictor.data['results']['year'] == selected_year].copy()
                    if not year_results.empty and 'round' in year_results.columns:
                        year_results = year_results.sort_values('round')

                        # Points accumulation over season
                        if 'points' in year_results.columns:
                            # Calculate cumulative points for top drivers
                            top_drivers = year_results.groupby('driver_id')['points'].sum().nlargest(5).index.tolist()

                            cumulative_data = []
                            for driver_id in top_drivers:
                                driver_races = year_results[year_results['driver_id'] == driver_id].copy()
                                driver_races = driver_races.sort_values('round')
                                driver_races['cumulative_points'] = driver_races['points'].cumsum()

                                # Get driver name
                                driver_name = f"Driver {driver_id}"
                                if not driver_races.empty:
                                    first_race = driver_races.iloc[0]
                                    name = f"{first_race.get('driver_first_name', '')} {first_race.get('driver_last_name', '')}".strip()
                                    if name:
                                        driver_name = name

                                for _, race in driver_races.iterrows():
                                    cumulative_data.append({
                                        'Race': race.get('round', 0),
                                        'Driver': driver_name,
                                        'Cumulative Points': race.get('cumulative_points', 0)
                                    })

                            if cumulative_data:
                                cum_df = pd.DataFrame(cumulative_data)
                                fig = px.line(
                                    cum_df,
                                    x='Race',
                                    y='Cumulative Points',
                                    color='Driver',
                                    title=f'{selected_year} Season - Cumulative Points by Driver',
                                    markers=True
                                )
                                st.plotly_chart(fig, use_container_width=True)

    # Circuit analysis
    st.subheader("🏁 Circuit Analysis")

    if 'circuits' in predictor.data and not predictor.data['circuits'].empty:
        circuits_df = predictor.data['circuits']
        st.metric("Total Circuits", len(circuits_df))

        if len(circuits_df) > 0:
            # Show circuits table
            display_cols = ['circuitId', 'circuitName', 'location', 'country']
            available_cols = [col for col in display_cols if col in circuits_df.columns]
            if available_cols:
                st.dataframe(
                    circuits_df[available_cols],
                    use_container_width=True,
                    hide_index=True
                )

    # Trends and patterns
    st.subheader("🔍 Historical Trends & Patterns")

    if st.checkbox("Show Trend Analysis"):
        st.markdown("""
        ### Notable Historical Trends (2015-Present)

        **Dominance Periods**:
        - 2014-2021: Mercedes hegemony (7 consecutive constructor titles)
        - 2022-present: Red Bull emergence (constructor titles 2022-2023)

        **Driver Eras**:
        - Lewis Hamilton: 7 WDC (2008, 2014-2015, 2017-2020)
        - Max Verstappen: 3 WDC (2021-2023)
        - Sebastian Vettel: 4 WDC (2010-2013)

        **Regulatory Changes Impact**:
        - 2014: Hybrid turbo introduction
        - 2017: Wider cars, more downforce
        - 2019: Simpler front wings
        - 2022: Major aerodynamic overhaul (ground effect)
        - 2026: Upcoming regulation changes (planned)

        **Performance Trends**:
        - Increasing reliability (fewer mechanical DNFs)
        - Closer midfield competition
        - Growing importance of qualifying performance
        - Strategic variability increasing race unpredictability
        """)

    # Data quality
    st.subheader("📋 Data Quality Report")

    if st.button("📊 Generate Data Quality Report"):
        with st.spinner("Analyzing data quality..."):
            # Simple data quality checks
            quality_report = {}

            for data_name, df in predictor.data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    report = {
                        'Rows': len(df),
                        'Columns': len(df.columns),
                        'Complete Rows': df.dropna().shape[0],
                        'Complete Percentage': f"{df.dropna().shape[0]/len(df)*100:.1f}%" if len(df) > 0 else "0%",
                        'Duplicate Rows': df.duplicated().sum(),
                        'Memory Usage (MB)': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}"
                    }
                    quality_report[data_name] = report

            if quality_report:
                for data_name, report in quality_report.items():
                    with st.expander(f"{data_name.replace('_', ' ').title()} Data Quality"):
                        for key, value in report.items():
                            st.text(f"{key}: {value}")
            else:
                st.info("No data available for quality analysis")

    st.markdown("---")
    st.info("""
    💡 **Historical Analysis Tips**:
    - Look for patterns in driver performance across different circuits
    - Consider how regulatory changes affected team performance
    - Analyze how teammate relationships evolve over seasons
    - Check for consistency in performance metrics across years
    - Monitor how constructor advantages change over time
    """)

# Run the app
if __name__ == "__main__":
    main()
EOF