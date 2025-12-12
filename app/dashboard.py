"""
NYC Crash Risk Prediction Dashboard - Enhanced Version

Multi-page Streamlit application with:
1. Risk Prediction with confidence intervals
2. SHAP Explainability
3. 24-Hour Forecast
4. Model Performance
5. Hotspot Analysis
6. Scenario Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import h3
import pydeck as pdk
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="NYC Crash Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = "models/final_model.joblib"
DATA_PATH = "data/processed/test.csv"
LOCATION_PATH = "data/processed/h3_location_mapping.csv"

# Feature columns
FEATURE_COLS = [
    'temperature', 'precipitation', 'wind_speed', 'snow_depth',
    'hour_of_day', 'day_of_week', 'month', 'year',
    'accidents_1h_ago', 'accidents_24h_ago', 'rolling_mean_7d',
    'is_holiday', 'is_weekend'
]


# ===================== CACHING FUNCTIONS =====================

@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    """Load reference data for hexagons and baseline features."""
    df = pd.read_csv(DATA_PATH, parse_dates=['hour'])
    return df


@st.cache_data
def load_location_mapping():
    """Load H3 to location name mapping."""
    try:
        mapping = pd.read_csv(LOCATION_PATH)
        return dict(zip(mapping['h3_index'], mapping['location_name']))
    except Exception:
        return {}


@st.cache_resource
def load_shap_explainer(_model, _X_background):
    """Load or create SHAP explainer."""
    try:
        from explainability import SHAPExplainer
        explainer = SHAPExplainer(_model, FEATURE_COLS, model_type='tree')
        explainer.fit(_X_background, max_samples=500)
        return explainer
    except Exception as e:
        st.warning(f"SHAP explainer not available: {e}")
        return None


@st.cache_resource
def load_conformal_predictor(_model, _X_cal, _y_cal):
    """Load or create conformal predictor for uncertainty."""
    try:
        from uncertainty import ConformalPredictor
        predictor = ConformalPredictor(_model, confidence_level=0.90)
        predictor.calibrate(_X_cal, _y_cal)
        return predictor
    except Exception as e:
        st.warning(f"Conformal predictor not available: {e}")
        return None


# ===================== HELPER FUNCTIONS =====================

def get_hexagon_center(h3_index):
    """Get the center coordinates of an H3 hexagon."""
    lat, lng = h3.cell_to_latlng(h3_index)
    return lat, lng


def get_location_name(h3_index, location_map):
    """Get location name for H3 index, fallback to short index if not found."""
    if location_map and h3_index in location_map:
        return location_map[h3_index]
    return h3_index[:12] + '...'


def prepare_prediction_data(df, selected_datetime, simulation_params):
    """Prepare data for prediction based on selected datetime and simulation parameters."""
    hexagons = df['h3_index'].unique()
    
    hour_of_day = selected_datetime.hour
    day_of_week = selected_datetime.weekday()
    month = selected_datetime.month
    year = selected_datetime.year
    is_weekend = 1 if day_of_week >= 5 else 0
    is_holiday = 0
    
    similar_hours = df[df['hour'].dt.hour == hour_of_day]
    if len(similar_hours) > 0:
        base_temp = similar_hours['temperature'].mean()
        base_precip = similar_hours['precipitation'].mean()
        base_wind = similar_hours['wind_speed'].mean()
        base_snow = similar_hours['snow_depth'].mean()
    else:
        base_temp, base_precip, base_wind, base_snow = 15.0, 0.0, 10.0, 0.0
    
    temperature = base_temp * (1 + simulation_params.get('temp_change', 0) / 100)
    precipitation = base_precip * (1 + simulation_params.get('rain_change', 0) / 100)
    wind_speed = base_wind * (1 + simulation_params.get('wind_change', 0) / 100)
    snow_depth = base_snow * (1 + simulation_params.get('snow_change', 0) / 100)
    
    hex_stats = df.groupby('h3_index').agg({
        'accident_count': 'mean',
        'rolling_mean_7d': 'mean'
    }).reset_index()
    hex_stats.columns = ['h3_index', 'avg_accidents', 'avg_rolling']
    
    pred_data = []
    for hex_id in hexagons:
        hex_stat = hex_stats[hex_stats['h3_index'] == hex_id]
        avg_acc = hex_stat['avg_accidents'].values[0] if len(hex_stat) > 0 else 0.0
        rolling = hex_stat['avg_rolling'].values[0] if len(hex_stat) > 0 else 0.0
        
        pred_data.append({
            'h3_index': hex_id,
            'temperature': temperature,
            'precipitation': precipitation,
            'wind_speed': wind_speed,
            'snow_depth': snow_depth,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'month': month,
            'year': year,
            'is_holiday': is_holiday,
            'is_weekend': is_weekend,
            'accidents_1h_ago': avg_acc,
            'accidents_24h_ago': avg_acc,
            'rolling_mean_7d': rolling
        })
    
    return pd.DataFrame(pred_data)


def run_predictions(model, pred_df, conformal_predictor=None):
    """Run model predictions with optional uncertainty."""
    X = pred_df[FEATURE_COLS].fillna(0)
    predictions = model.predict(X)
    predictions = np.clip(predictions, 0, None)
    
    pred_df['predicted_risk'] = predictions
    pred_df['risk_level'] = pd.cut(
        predictions,
        bins=[-0.001, 0.05, 0.15, 0.3, np.inf],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # Add confidence intervals if available
    if conformal_predictor is not None:
        try:
            _, lower, upper = conformal_predictor.predict_intervals(X)
            pred_df['lower_bound'] = np.maximum(lower, 0)
            pred_df['upper_bound'] = upper
            pred_df['interval_width'] = pred_df['upper_bound'] - pred_df['lower_bound']
        except Exception:
            pass
    
    return pred_df


def create_risk_map(pred_df, show_intervals=False, location_map=None):
    """Create PyDeck map with hexagons colored by risk."""
    # Create a copy to avoid modifying original
    pred_df = pred_df.copy()
    
    map_data = []
    for _, row in pred_df.iterrows():
        lat, lng = get_hexagon_center(row['h3_index'])
        risk = row['predicted_risk']
        location_name = get_location_name(row['h3_index'], location_map)
        
        # Determine color based on risk level
        if risk < 0.05:
            r, g, b, a = 0, 180, 0, 160
        elif risk < 0.15:
            r, g, b, a = 255, 200, 0, 160
        elif risk < 0.3:
            r, g, b, a = 255, 120, 0, 160
        else:
            r, g, b, a = 255, 0, 0, 180
        
        map_data.append({
            'h3_index': row['h3_index'],
            'location': location_name,
            'lat': lat,
            'lng': lng,
            'risk': float(risk),
            'lower': float(row.get('lower_bound', risk)),
            'upper': float(row.get('upper_bound', risk)),
            'r': r,
            'g': g,
            'b': b,
            'a': a
        })
    
    map_df = pd.DataFrame(map_data)
    
    tooltip_text = "Location: {location}<br/>Risk: {risk:.4f}"
    if show_intervals and 'lower_bound' in pred_df.columns:
        tooltip_text = "Location: {location}<br/>Risk: {risk:.4f}<br/>90% CI: [{lower:.4f}, {upper:.4f}]"
    
    layer = pdk.Layer(
        "H3HexagonLayer",
        map_df,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_hexagon="h3_index",
        get_fill_color="[r, g, b, a]",
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1
    )
    
    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10,
        pitch=0
    )
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": tooltip_text}
    )
    
    return deck


# ===================== PAGE FUNCTIONS =====================

def page_risk_prediction():
    """Main risk prediction page with confidence intervals."""
    st.header("🗺️ Risk Prediction Map")
    st.markdown("Predict accident risk across NYC with confidence intervals")
    
    model = load_model()
    ref_data = load_reference_data()
    location_map = load_location_mapping()
    
    # Prepare calibration data for conformal predictor
    X_cal = ref_data[FEATURE_COLS].fillna(0).sample(min(5000, len(ref_data)), random_state=42)
    y_cal = ref_data.loc[X_cal.index, 'accident_count']
    conformal = load_conformal_predictor(model, X_cal, y_cal)
    
    # Sidebar controls
    st.sidebar.header("📅 Select Date & Time")
    
    min_date = ref_data['hour'].min().date()
    max_date = ref_data['hour'].max().date()
    selected_date = st.sidebar.date_input("Date", value=max_date, min_value=min_date, max_value=max_date)
    selected_hour = st.sidebar.slider("Hour of Day", 0, 23, 12, format="%d:00")
    selected_datetime = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🌤️ Weather Simulation")
    
    rain_change = st.sidebar.slider("Rain Intensity (%)", -100, 200, 0, 10)
    temp_change = st.sidebar.slider("Temperature Change (%)", -50, 50, 0, 5)
    wind_change = st.sidebar.slider("Wind Speed Change (%)", -50, 100, 0, 10)
    snow_change = st.sidebar.slider("Snow Depth Change (%)", -100, 500, 0, 25)
    
    simulation_params = {
        'rain_change': rain_change,
        'temp_change': temp_change,
        'wind_change': wind_change,
        'snow_change': snow_change
    }
    
    show_intervals = st.sidebar.checkbox("Show Confidence Intervals", value=True)
    
    if st.sidebar.button("🔮 Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running predictions..."):
            pred_df = prepare_prediction_data(ref_data, selected_datetime, simulation_params)
            results = run_predictions(model, pred_df, conformal if show_intervals else None)
            st.session_state['results'] = results
            st.session_state['datetime'] = selected_datetime
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🚨 Total Predicted Accidents", f"{results['predicted_risk'].sum():.1f}")
        with col2:
            st.metric("📊 Average Risk", f"{results['predicted_risk'].mean():.4f}")
        with col3:
            high_risk = len(results[results['risk_level'].isin(['High', 'Critical'])])
            st.metric("⚠️ High Risk Areas", f"{high_risk}")
        with col4:
            if 'interval_width' in results.columns:
                st.metric("📏 Avg Uncertainty", f"±{results['interval_width'].mean()/2:.4f}")
        
        st.markdown(f"**Predictions for:** {st.session_state['datetime'].strftime('%B %d, %Y at %H:00')}")
        
        # Map
        map_col, stats_col = st.columns([2, 1])
        
        with map_col:
            deck = create_risk_map(results, show_intervals, location_map)
            st.pydeck_chart(deck)
            st.markdown("""
            **Risk Legend:** 🟢 Low (<0.05) | 🟡 Medium (0.05-0.15) | 🟠 High (0.15-0.3) | 🔴 Critical (>0.3)
            """)
        
        with stats_col:
            st.subheader("📈 Top 10 Riskiest Areas")
            top_10 = results.nlargest(10, 'predicted_risk')[['h3_index', 'predicted_risk', 'risk_level']]
            
            if 'lower_bound' in results.columns:
                top_10 = results.nlargest(10, 'predicted_risk')[
                    ['h3_index', 'predicted_risk', 'lower_bound', 'upper_bound', 'risk_level']
                ]
                top_10['CI'] = top_10.apply(lambda r: f"[{r['lower_bound']:.3f}, {r['upper_bound']:.3f}]", axis=1)
                display_df = top_10[['h3_index', 'predicted_risk', 'CI', 'risk_level']].copy()
                display_df.columns = ['Location', 'Risk', '90% CI', 'Level']
            else:
                display_df = top_10.copy()
                display_df.columns = ['Location', 'Risk', 'Level']
            
            display_df['Location'] = display_df['Location'].apply(lambda h: get_location_name(h, location_map))
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # Risk distribution
            st.subheader("📊 Risk Distribution")
            risk_counts = results['risk_level'].value_counts()
            st.bar_chart(risk_counts)
    else:
        st.info("👈 Select date, time, and weather parameters, then click **Run Prediction**")


def page_explainability():
    """SHAP explainability page."""
    st.header("🔍 Model Explainability")
    st.markdown("Understand what drives crash risk predictions using SHAP values")
    
    model = load_model()
    ref_data = load_reference_data()
    location_map = load_location_mapping()
    
    X_background = ref_data[FEATURE_COLS].fillna(0).sample(min(1000, len(ref_data)), random_state=42)
    explainer = load_shap_explainer(model, X_background)
    
    if explainer is None:
        st.error("SHAP explainability not available. Please ensure the explainability module is properly installed.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Global Importance", "Local Explanation", "What-If Analysis"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        st.markdown("Which features matter most across all predictions?")
        
        with st.spinner("Computing SHAP values..."):
            importance = explainer.get_global_importance(X_background)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            importance_sorted = importance.sort_values('importance')
            ax.barh(importance_sorted['feature'], importance_sorted['importance'], color='steelblue')
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Feature Importance (SHAP)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.dataframe(
                importance[['feature', 'importance_pct']].rename(columns={
                    'feature': 'Feature',
                    'importance_pct': 'Importance %'
                }).round(2),
                hide_index=True,
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Local Explanation")
        st.markdown("Explain why a specific location has high/low risk")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            hexagon_options = results.nlargest(20, 'predicted_risk')['h3_index'].tolist()
            display_options = [get_location_name(h, location_map) for h in hexagon_options]
            hex_to_display = dict(zip(hexagon_options, display_options))
            display_to_hex = dict(zip(display_options, hexagon_options))
            
            selected_display = st.selectbox("Select a location to explain", display_options)
            selected_hex = display_to_hex[selected_display]
            
            if st.button("Explain This Prediction"):
                hex_data = results[results['h3_index'] == selected_hex]
                X_single = hex_data[FEATURE_COLS].fillna(0)
                
                explanation = explainer.explain_single(X_single)
                
                st.metric("Predicted Risk", f"{explanation['prediction']:.4f}")
                st.metric("Base Value", f"{explanation['base_value']:.4f}")
                
                st.markdown("**Top factors increasing risk:**")
                for _, row in explanation['top_positive'].head(3).iterrows():
                    st.write(f"- **{row['feature']}**: +{row['shap_value']:.4f} (value: {row['value']:.2f})")
                
                st.markdown("**Top factors decreasing risk:**")
                for _, row in explanation['top_negative'].head(3).iterrows():
                    st.write(f"- **{row['feature']}**: {row['shap_value']:.4f} (value: {row['value']:.2f})")
        else:
            st.info("Run a prediction first to see local explanations")
    
    with tab3:
        st.subheader("What-If Analysis")
        st.markdown("See how changing features affects the prediction")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            hexagon_options = results['h3_index'].unique()[:50].tolist()
            display_options = [get_location_name(h, location_map) for h in hexagon_options]
            display_to_hex = dict(zip(display_options, hexagon_options))
            
            selected_display = st.selectbox("Select location", display_options, key='whatif_hex')
            selected_hex = display_to_hex[selected_display]
            
            hex_data = results[results['h3_index'] == selected_hex]
            X_original = hex_data[FEATURE_COLS].fillna(0)
            original_pred = model.predict(X_original)[0]
            
            st.markdown(f"**Original prediction:** {original_pred:.4f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_temp = st.slider("Temperature", -20.0, 40.0, float(X_original['temperature'].values[0]))
                new_precip = st.slider("Precipitation", 0.0, 50.0, float(X_original['precipitation'].values[0]))
            
            with col2:
                new_wind = st.slider("Wind Speed", 0.0, 50.0, float(X_original['wind_speed'].values[0]))
                new_hour = st.slider("Hour of Day", 0, 23, int(X_original['hour_of_day'].values[0]))
            
            X_modified = X_original.copy()
            X_modified['temperature'] = new_temp
            X_modified['precipitation'] = new_precip
            X_modified['wind_speed'] = new_wind
            X_modified['hour_of_day'] = new_hour
            
            new_pred = model.predict(X_modified)[0]
            change = new_pred - original_pred
            
            st.metric(
                "Modified Prediction",
                f"{new_pred:.4f}",
                delta=f"{change:+.4f}",
                delta_color="inverse"
            )
        else:
            st.info("Run a prediction first to use What-If analysis")


def page_forecast():
    """24-hour forecast page."""
    st.header("📈 24-Hour Risk Forecast")
    st.markdown("Predict how crash risk will evolve over the next 24 hours")
    
    model = load_model()
    ref_data = load_reference_data()
    location_map = load_location_mapping()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_date = st.date_input("Start Date", value=ref_data['hour'].max().date())
        start_hour = st.slider("Start Hour", 0, 23, 8)
        
        hexagon_options = ref_data.groupby('h3_index')['accident_count'].mean().nlargest(50).index.tolist()
        display_options = [get_location_name(h, location_map) for h in hexagon_options]
        display_to_hex = dict(zip(display_options, hexagon_options))
        
        selected_displays = st.multiselect(
            "Select locations to forecast",
            display_options,
            default=display_options[:3]
        )
        selected_hexagons = [display_to_hex[d] for d in selected_displays]
    
    if st.button("Generate 24-Hour Forecast", type="primary"):
        if not selected_hexagons:
            st.warning("Please select at least one location")
            return
        
        with st.spinner("Generating forecast..."):
            start_time = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=start_hour)
            
            forecast_data = []
            
            for hour_offset in range(24):
                forecast_time = start_time + timedelta(hours=hour_offset)
                
                for hex_id in selected_hexagons:
                    hex_history = ref_data[ref_data['h3_index'] == hex_id]
                    
                    features = {
                        'h3_index': hex_id,
                        'temperature': hex_history['temperature'].mean(),
                        'precipitation': hex_history['precipitation'].mean(),
                        'wind_speed': hex_history['wind_speed'].mean(),
                        'snow_depth': hex_history['snow_depth'].mean(),
                        'hour_of_day': forecast_time.hour,
                        'day_of_week': forecast_time.weekday(),
                        'month': forecast_time.month,
                        'year': forecast_time.year,
                        'is_holiday': 0,
                        'is_weekend': 1 if forecast_time.weekday() >= 5 else 0,
                        'accidents_1h_ago': hex_history['accident_count'].mean(),
                        'accidents_24h_ago': hex_history['accident_count'].mean(),
                        'rolling_mean_7d': hex_history['rolling_mean_7d'].mean()
                    }
                    
                    X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)
                    pred = model.predict(X)[0]
                    
                    forecast_data.append({
                        'time': forecast_time,
                        'hour': hour_offset,
                        'location': get_location_name(hex_id, location_map),
                        'risk': max(0, pred)
                    })
            
            forecast_df = pd.DataFrame(forecast_data)
            st.session_state['forecast'] = forecast_df
    
    with col2:
        if 'forecast' in st.session_state:
            forecast_df = st.session_state['forecast']
            
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for location in forecast_df['location'].unique():
                loc_data = forecast_df[forecast_df['location'] == location]
                ax.plot(loc_data['hour'], loc_data['risk'], 'o-', label=location, linewidth=2, markersize=4)
            
            ax.set_xlabel('Hours from Start')
            ax.set_ylabel('Predicted Risk')
            ax.set_title('24-Hour Risk Forecast')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Summary stats
            st.subheader("Forecast Summary")
            summary = forecast_df.groupby('location').agg({
                'risk': ['mean', 'max', 'min', 'std']
            }).round(4)
            summary.columns = ['Mean Risk', 'Max Risk', 'Min Risk', 'Volatility']
            st.dataframe(summary, use_container_width=True)
            
            # Peak risk times
            st.subheader("Peak Risk Times")
            peak_times = forecast_df.loc[forecast_df.groupby('location')['risk'].idxmax()]
            st.dataframe(
                peak_times[['location', 'hour', 'risk']].rename(columns={
                    'location': 'Location',
                    'hour': 'Peak Hour',
                    'risk': 'Peak Risk'
                }),
                hide_index=True
            )


def page_model_performance():
    """Model performance dashboard."""
    st.header("📊 Model Performance Dashboard")
    st.markdown("Evaluate model accuracy and reliability")
    
    model = load_model()
    ref_data = load_reference_data()
    
    # Sample for evaluation
    sample = ref_data.sample(min(10000, len(ref_data)), random_state=42)
    X = sample[FEATURE_COLS].fillna(0)
    y_true = sample['accident_count']
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, None)
    
    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMSE", f"{rmse:.4f}")
    with col2:
        st.metric("MAE", f"{mae:.4f}")
    with col3:
        st.metric("R² Score", f"{r2:.4f}")
    with col4:
        baseline_mae = mean_absolute_error(y_true, np.full_like(y_true, y_true.mean()))
        improvement = (baseline_mae - mae) / baseline_mae * 100
        st.metric("vs Baseline", f"{improvement:.1f}%")
    
    tab1, tab2, tab3 = st.tabs(["Predictions vs Actual", "Residual Analysis", "By Time Period"])
    
    with tab1:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of predictions
        axes[1].hist(y_pred, bins=50, alpha=0.7, label='Predicted', color='blue')
        axes[1].hist(y_true, bins=50, alpha=0.5, label='Actual', color='orange')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution Comparison')
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Residual')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(residuals, bins=50, alpha=0.7, color='green')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residual Distribution (Mean: {residuals.mean():.4f})')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        sample['prediction'] = y_pred
        sample['error'] = np.abs(y_true - y_pred)
        
        hourly_performance = sample.groupby('hour_of_day').agg({
            'error': 'mean',
            'prediction': 'mean',
            'accident_count': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(hourly_performance['hour_of_day'], hourly_performance['error'], alpha=0.7)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Prediction Error by Hour')
        ax.set_xticks(range(24))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def page_hotspot_analysis():
    """Hotspot clustering analysis page."""
    st.header("🎯 Hotspot Cluster Analysis")
    st.markdown("Identify patterns in high-risk areas through clustering")
    
    ref_data = load_reference_data()
    
    try:
        from clustering import HexagonClusterer, TemporalPatternExtractor
        
        clusterer = HexagonClusterer()
        hex_df = clusterer.prepare_hexagon_features(ref_data, 'h3_index', 'accident_count')
        
        n_clusters = st.slider("Number of Clusters", 3, 10, 5)
        
        if st.button("Run Clustering Analysis", type="primary"):
            with st.spinner("Clustering hexagons..."):
                hex_df = clusterer.fit_kmeans(hex_df, n_clusters)
                interpretations = clusterer.get_cluster_interpretation()
                
                st.session_state['hex_clusters'] = hex_df
                st.session_state['cluster_interpretations'] = interpretations
        
        if 'hex_clusters' in st.session_state:
            hex_df = st.session_state['hex_clusters']
            interpretations = st.session_state['cluster_interpretations']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Cluster Map")
                
                map_data = []
                colors = [[31, 119, 180], [255, 127, 14], [44, 160, 44], 
                         [214, 39, 40], [148, 103, 189], [140, 86, 75],
                         [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207]]
                
                for _, row in hex_df.iterrows():
                    h3_idx = row['h3_index']
                    cluster = int(row['cluster'])
                    lat, lng = get_hexagon_center(h3_idx)
                    
                    map_data.append({
                        'h3_index': h3_idx,
                        'lat': lat,
                        'lng': lng,
                        'cluster': cluster,
                        'color': colors[cluster % len(colors)] + [160]
                    })
                
                layer = pdk.Layer(
                    "H3HexagonLayer",
                    pd.DataFrame(map_data),
                    pickable=True,
                    stroked=True,
                    filled=True,
                    get_hexagon="h3_index",
                    get_fill_color="color",
                    get_line_color=[255, 255, 255]
                )
                
                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=10),
                    tooltip={"text": "Cluster: {cluster}"}
                )
                st.pydeck_chart(deck)
            
            with col2:
                st.subheader("Cluster Profiles")
                
                for cluster_id, desc in interpretations.items():
                    cluster_count = (hex_df['cluster'] == cluster_id).sum()
                    st.markdown(f"**Cluster {cluster_id}** ({cluster_count} hexagons)")
                    st.write(desc)
                    st.markdown("---")
                
                st.subheader("Cluster Statistics")
                cluster_stats = hex_df.groupby('cluster').agg({
                    'accident_count_mean': 'mean',
                    'accident_count_max': 'max'
                }).round(4)
                cluster_stats.columns = ['Avg Risk', 'Max Risk']
                st.dataframe(cluster_stats)
    
    except Exception as e:
        st.error(f"Clustering analysis not available: {e}")
        st.info("Make sure the clustering module is properly installed.")


def page_scenario_comparison():
    """Scenario comparison page."""
    st.header("🔄 Scenario Comparison")
    st.markdown("Compare predictions under different weather conditions")
    
    model = load_model()
    ref_data = load_reference_data()
    location_map = load_location_mapping()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Scenario A (Baseline)")
        rain_a = st.slider("Rain Intensity (%)", -100, 300, 0, key='rain_a')
        temp_a = st.slider("Temperature (%)", -50, 50, 0, key='temp_a')
        wind_a = st.slider("Wind Speed (%)", -50, 100, 0, key='wind_a')
        scenario_a = {'rain_change': rain_a, 'temp_change': temp_a, 'wind_change': wind_a, 'snow_change': 0}
    
    with col2:
        st.subheader("📋 Scenario B (Modified)")
        rain_b = st.slider("Rain Intensity (%)", -100, 300, 200, key='rain_b')
        temp_b = st.slider("Temperature (%)", -50, 50, -30, key='temp_b')
        wind_b = st.slider("Wind Speed (%)", -50, 100, 50, key='wind_b')
        scenario_b = {'rain_change': rain_b, 'temp_change': temp_b, 'wind_change': wind_b, 'snow_change': 0}
    
    selected_time = datetime.combine(ref_data['hour'].max().date(), datetime.min.time()) + timedelta(hours=12)
    
    if st.button("🔄 Compare Scenarios", type="primary", use_container_width=True):
        with st.spinner("Computing scenarios..."):
            # Generate predictions for Scenario A
            pred_a = prepare_prediction_data(ref_data, selected_time, scenario_a)
            results_a = run_predictions(model, pred_a.copy())
            
            # Generate predictions for Scenario B
            pred_b = prepare_prediction_data(ref_data, selected_time, scenario_b)
            results_b = run_predictions(model, pred_b.copy())
            
            # Store in session with deep copies to ensure independence
            st.session_state['scenario_a'] = results_a.copy()
            st.session_state['scenario_b'] = results_b.copy()
            st.session_state['scenario_params_a'] = scenario_a.copy()
            st.session_state['scenario_params_b'] = scenario_b.copy()
    
    if 'scenario_a' in st.session_state:
        results_a = st.session_state['scenario_a']
        results_b = st.session_state['scenario_b']
        params_a = st.session_state.get('scenario_params_a', {})
        params_b = st.session_state.get('scenario_params_b', {})
        
        # Calculate totals
        total_a = results_a['predicted_risk'].sum()
        total_b = results_b['predicted_risk'].sum()
        diff = total_b - total_a
        pct_change = (diff / total_a * 100) if total_a > 0 else 0
        
        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Scenario A Total", f"{total_a:.2f}")
        with col2:
            st.metric("Scenario B Total", f"{total_b:.2f}")
        with col3:
            st.metric("Absolute Change", f"{diff:+.2f}")
        with col4:
            st.metric("% Change", f"{pct_change:+.1f}%")
        
        # Show actual weather values used
        with st.expander("🌤️ Weather Values Used"):
            weather_col1, weather_col2 = st.columns(2)
            with weather_col1:
                st.markdown("**Scenario A:**")
                if len(results_a) > 0:
                    st.write(f"- Temperature: {results_a['temperature'].iloc[0]:.1f}°C")
                    st.write(f"- Precipitation: {results_a['precipitation'].iloc[0]:.2f} mm")
                    st.write(f"- Wind: {results_a['wind_speed'].iloc[0]:.1f} km/h")
            with weather_col2:
                st.markdown("**Scenario B:**")
                if len(results_b) > 0:
                    st.write(f"- Temperature: {results_b['temperature'].iloc[0]:.1f}°C")
                    st.write(f"- Precipitation: {results_b['precipitation'].iloc[0]:.2f} mm")
                    st.write(f"- Wind: {results_b['wind_speed'].iloc[0]:.1f} km/h")
        
        st.markdown("---")
        
        # Create comparison dataframe
        comparison = pd.merge(
            results_a[['h3_index', 'predicted_risk']].rename(columns={'predicted_risk': 'risk_a'}),
            results_b[['h3_index', 'predicted_risk']].rename(columns={'predicted_risk': 'risk_b'}),
            on='h3_index'
        )
        comparison['change'] = comparison['risk_b'] - comparison['risk_a']
        comparison['change_abs'] = comparison['change'].abs()
        
        # Side-by-side comparison maps
        st.subheader("📊 Side-by-Side Risk Maps")
        
        map_col1, map_col2 = st.columns(2)
        
        with map_col1:
            st.markdown(f"**Scenario A** (Total: {total_a:.2f})")
            # Create map data for A
            map_data_a = create_scenario_map_data(results_a, location_map)
            st.pydeck_chart(build_hex_map(map_data_a, "Scenario A"), key="map_scenario_a")
        
        with map_col2:
            st.markdown(f"**Scenario B** (Total: {total_b:.2f})")
            # Create map data for B
            map_data_b = create_scenario_map_data(results_b, location_map)
            st.pydeck_chart(build_hex_map(map_data_b, "Scenario B"), key="map_scenario_b")
        
        # Difference analysis
        st.markdown("---")
        st.subheader("📈 Detailed Changes")
        
        # Top increases and decreases
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔺 Top 5 Risk Increases:**")
            top_inc = comparison.nlargest(5, 'change')[['h3_index', 'risk_a', 'risk_b', 'change']]
            top_inc['Location'] = top_inc['h3_index'].apply(lambda h: get_location_name(h, location_map))
            top_inc = top_inc[['Location', 'risk_a', 'risk_b', 'change']]
            top_inc.columns = ['Location', 'Risk A', 'Risk B', 'Change']
            st.dataframe(top_inc.round(4), hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**🔻 Top 5 Risk Decreases:**")
            top_dec = comparison.nsmallest(5, 'change')[['h3_index', 'risk_a', 'risk_b', 'change']]
            top_dec['Location'] = top_dec['h3_index'].apply(lambda h: get_location_name(h, location_map))
            top_dec = top_dec[['Location', 'risk_a', 'risk_b', 'change']]
            top_dec.columns = ['Location', 'Risk A', 'Risk B', 'Change']
            st.dataframe(top_dec.round(4), hide_index=True, use_container_width=True)
        
        # Statistics
        st.markdown("---")
        st.markdown("**Summary Statistics:**")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.write(f"Locations with increased risk: {(comparison['change'] > 0).sum()}")
        with stats_col2:
            st.write(f"Locations with decreased risk: {(comparison['change'] < 0).sum()}")
        with stats_col3:
            st.write(f"Average change per location: {comparison['change'].mean():.4f}")


def create_scenario_map_data(results_df, location_map):
    """Create map data for a scenario."""
    map_data = []
    for _, row in results_df.iterrows():
        lat, lng = get_hexagon_center(row['h3_index'])
        risk = float(row['predicted_risk'])
        location_name = get_location_name(row['h3_index'], location_map)
        
        # Color based on risk
        if risk < 0.05:
            r, g, b = 0, 180, 0
        elif risk < 0.15:
            r, g, b = 255, 200, 0
        elif risk < 0.3:
            r, g, b = 255, 120, 0
        else:
            r, g, b = 255, 0, 0
        
        map_data.append({
            'h3_index': row['h3_index'],
            'location': location_name,
            'risk': risk,
            'r': r, 'g': g, 'b': b, 'a': 160
        })
    
    return pd.DataFrame(map_data)


def build_hex_map(map_df, title):
    """Build a PyDeck map from prepared data."""
    layer = pdk.Layer(
        "H3HexagonLayer",
        map_df,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_hexagon="h3_index",
        get_fill_color="[r, g, b, a]",
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1
    )
    
    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10,
        pitch=0
    )
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": f"<b>{title}</b><br/>Location: {{location}}<br/>Risk: {{risk:.4f}}"}
    )


# ===================== MAIN APP =====================

def main():
    st.sidebar.title("🚗 NYC Crash Risk")
    st.sidebar.markdown("Prediction Dashboard")
    st.sidebar.markdown("---")
    
    pages = {
        "🗺️ Risk Prediction": page_risk_prediction,
        "🔍 Explainability": page_explainability,
        "📈 24h Forecast": page_forecast,
        "📊 Model Performance": page_model_performance,
        "🎯 Hotspot Analysis": page_hotspot_analysis,
        "🔄 Scenario Comparison": page_scenario_comparison
    }
    
    selected_page = st.sidebar.radio("Navigate", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This dashboard predicts vehicle crash risk across NYC using:
    - **XGBoost** model with Poisson objective
    - **SHAP** for explainability
    - **Conformal Prediction** for uncertainty
    - **H3** hexagonal spatial binning
    """)
    
    try:
        pages[selected_page]()
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
