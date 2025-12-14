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
import plotly.express as px
import plotly.graph_objects as go

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="NYC Crash Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS STYLING =====================

CUSTOM_CSS = """
<style>
    /* ==================== CSS VARIABLES - COLOR PALETTE ==================== */
    :root {
        --primary-cyan: #00D4FF;
        --primary-purple: #BD00FF;
        --accent-pink: #FF006E;
        --bg-dark: #0A0E17;
        --bg-card: #131B2E;
        --bg-card-hover: #1A2340;
        --success-green: #00F5A0;
        --warning-amber: #FFB800;
        --danger-red: #FF3366;
        --text-primary: #FFFFFF;
        --text-secondary: #A0AEC0;
        --text-muted: #6B7280;
        --border-glow: rgba(0, 212, 255, 0.3);
        --shadow-cyan: 0 0 20px rgba(0, 212, 255, 0.3);
        --shadow-purple: 0 0 20px rgba(189, 0, 255, 0.3);
        --shadow-pink: 0 0 20px rgba(255, 0, 110, 0.3);
        --gradient-primary: linear-gradient(135deg, #00D4FF 0%, #BD00FF 100%);
        --gradient-accent: linear-gradient(135deg, #FF006E 0%, #BD00FF 100%);
        --gradient-success: linear-gradient(135deg, #00F5A0 0%, #00D4FF 100%);
        --gradient-danger: linear-gradient(135deg, #FF3366 0%, #FF006E 100%);
    }
    
    /* ==================== GLOBAL STYLES ==================== */
    .stApp {
        background: linear-gradient(180deg, #0A0E17 0%, #0F1629 50%, #0A0E17 100%);
    }
    
    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-cyan), var(--primary-purple));
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-cyan);
    }
    
    /* ==================== SIDEBAR STYLING ==================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1321 0%, #131B2E 50%, #0D1321 100%) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.2) !important;
        box-shadow: 4px 0 30px rgba(0, 212, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: var(--text-primary);
    }
    
    /* Sidebar title glow effect */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    /* Radio buttons styling */
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(19, 27, 46, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin: 4px 0 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        border-color: var(--primary-cyan) !important;
        box-shadow: var(--shadow-cyan) !important;
        transform: translateX(5px);
        background: rgba(0, 212, 255, 0.1) !important;
    }
    
    /* ==================== MAIN CONTENT HEADERS ==================== */
    .main h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(0, 212, 255, 0.4);
        padding-bottom: 10px;
        border-bottom: 2px solid;
        border-image: var(--gradient-primary) 1;
        margin-bottom: 1rem;
    }
    
    .main h2 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: var(--primary-cyan) !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .main h3 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    /* ==================== METRIC CARDS ==================== */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(19, 27, 46, 0.9), rgba(13, 19, 33, 0.9)) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(0, 212, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
        border-radius: 16px 16px 0 0;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px) scale(1.02) !important;
        border-color: var(--primary-cyan) !important;
        box-shadow: 
            0 8px 40px rgba(0, 0, 0, 0.5),
            0 0 60px rgba(0, 212, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    }
    
    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
    
    /* ==================== BUTTONS ==================== */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 
            0 8px 30px rgba(0, 212, 255, 0.5),
            0 0 50px rgba(0, 212, 255, 0.3) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02) !important;
    }
    
    /* ==================== SLIDERS ==================== */
    .stSlider [data-baseweb="slider"] {
        margin-top: 10px;
    }
    
    .stSlider [data-testid="stThumbValue"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 4px 10px !important;
        font-weight: 600;
    }
    
    /* Slider track styling - Neon blue */
    .stSlider [data-baseweb="slider"] > div > div {
        background: rgba(0, 212, 255, 0.2) !important;
    }
    
    .stSlider [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #00D4FF, #BD00FF) !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5) !important;
    }
    
    /* Slider thumb - Neon glow */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #00D4FF, #BD00FF) !important;
        border: 2px solid #00D4FF !important;
        box-shadow: 
            0 0 10px rgba(0, 212, 255, 0.6),
            0 0 20px rgba(0, 212, 255, 0.3) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {
        box-shadow: 
            0 0 15px rgba(0, 212, 255, 0.8),
            0 0 30px rgba(0, 212, 255, 0.5) !important;
        transform: scale(1.1);
    }
    
    /* Slider label */
    .stSlider label {
        color: var(--text-primary) !important;
    }
    
    /* ==================== DATA TABLES ==================== */
    [data-testid="stDataFrame"] {
        background: rgba(19, 27, 46, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] table {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stDataFrame"] th {
        background: linear-gradient(180deg, rgba(0, 212, 255, 0.2), rgba(0, 212, 255, 0.1)) !important;
        color: var(--primary-cyan) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stDataFrame"] td {
        border-color: rgba(0, 212, 255, 0.1) !important;
    }
    
    [data-testid="stDataFrame"] tr:hover td {
        background: rgba(0, 212, 255, 0.1) !important;
    }
    
    /* ==================== TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(19, 27, 46, 0.6) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.1) !important;
        color: var(--primary-cyan) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* ==================== EXPANDERS ==================== */
    .streamlit-expanderHeader {
        background: rgba(19, 27, 46, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--primary-cyan) !important;
        box-shadow: var(--shadow-cyan) !important;
    }
    
    /* ==================== SELECT BOXES ==================== */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(19, 27, 46, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: var(--primary-cyan) !important;
    }
    
    /* ==================== MULTISELECT ==================== */
    [data-testid="stMultiSelect"] > div > div {
        background: rgba(19, 27, 46, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: var(--gradient-primary) !important;
        border-radius: 6px !important;
    }
    
    /* ==================== CHECKBOXES ==================== */
    .stCheckbox label span {
        color: var(--text-primary) !important;
    }
    
    /* ==================== INFO/WARNING BOXES ==================== */
    .stAlert {
        background: rgba(19, 27, 46, 0.8) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
    }
    
    /* ==================== CHARTS CONTAINER ==================== */
    [data-testid="stPlotlyChart"],
    .stPlotlyChart {
        background: rgba(19, 27, 46, 0.4) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 16px !important;
        padding: 10px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* ==================== ANIMATIONS ==================== */
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.6); }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Animated gradient border effect */
    .glow-card {
        animation: glow-pulse 3s ease-in-out infinite;
    }
    
    /* ==================== SPINNER ==================== */
    .stSpinner > div {
        border-top-color: var(--primary-cyan) !important;
    }
    
    /* ==================== HORIZONTAL RULE ==================== */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, var(--primary-cyan), var(--primary-purple), transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* ==================== DATE INPUT ==================== */
    [data-testid="stDateInput"] > div > div {
        background: rgba(19, 27, 46, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
</style>
"""

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Paths - Use absolute paths relative to script location for deployment compatibility
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
LOCATION_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "h3_location_mapping.csv")

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


# ===================== CHART STYLING UTILITIES =====================

# Neon color palette for charts
NEON_COLORS = {
    'cyan': '#00D4FF',
    'purple': '#BD00FF',
    'pink': '#FF006E',
    'green': '#00F5A0',
    'amber': '#FFB800',
    'red': '#FF3366',
    'teal': '#00E5CC',
    'orange': '#FF8C00'
}

NEON_COLOR_SEQUENCE = [
    '#00D4FF', '#BD00FF', '#FF006E', '#00F5A0', 
    '#FFB800', '#FF3366', '#00E5CC', '#FF8C00'
]


def configure_plotly_theme():
    """Return Plotly layout configuration with dark neon theme."""
    return {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(19, 27, 46, 0.8)',
        'plot_bgcolor': 'rgba(10, 14, 23, 0.9)',
        'font': {'color': '#FFFFFF', 'family': 'Segoe UI, Roboto, sans-serif'},
        'title': {'font': {'size': 20, 'color': '#00D4FF'}},
        'xaxis': {
            'gridcolor': 'rgba(0, 212, 255, 0.1)',
            'linecolor': 'rgba(0, 212, 255, 0.3)',
            'tickfont': {'color': '#A0AEC0'}
        },
        'yaxis': {
            'gridcolor': 'rgba(0, 212, 255, 0.1)',
            'linecolor': 'rgba(0, 212, 255, 0.3)',
            'tickfont': {'color': '#A0AEC0'}
        },
        'legend': {'bgcolor': 'rgba(19, 27, 46, 0.8)', 'bordercolor': 'rgba(0, 212, 255, 0.3)'},
        'hoverlabel': {
            'bgcolor': 'rgba(19, 27, 46, 0.95)',
            'bordercolor': '#00D4FF',
            'font': {'color': '#FFFFFF', 'size': 13}
        }
    }


def style_matplotlib_chart(fig, ax, title='', xlabel='', ylabel=''):
    """Apply neon dark theme styling to matplotlib figure and axes."""
    # Dark background
    fig.patch.set_facecolor('#0A0E17')
    ax.set_facecolor('#0D1321')
    
    # Neon grid
    ax.grid(True, alpha=0.2, color='#00D4FF', linestyle='--', linewidth=0.5)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color('#00D4FF')
        spine.set_alpha(0.3)
        spine.set_linewidth(1)
    
    # Labels and title with neon colors
    if title:
        ax.set_title(title, color='#00D4FF', fontsize=16, fontweight='bold', pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, color='#A0AEC0', fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, color='#A0AEC0', fontsize=12)
    
    # Tick colors
    ax.tick_params(colors='#A0AEC0', which='both')
    
    return fig, ax


def create_neon_bar_chart(data, x, y, title='', color_col=None, horizontal=False):
    """Create a Plotly bar chart with neon styling."""
    if horizontal:
        fig = px.bar(data, x=y, y=x, orientation='h',
                     color=color_col if color_col else None,
                     color_discrete_sequence=NEON_COLOR_SEQUENCE)
    else:
        fig = px.bar(data, x=x, y=y,
                     color=color_col if color_col else None,
                     color_discrete_sequence=NEON_COLOR_SEQUENCE)
    
    theme = configure_plotly_theme()
    fig.update_layout(
        title=title,
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        xaxis=theme['xaxis'],
        yaxis=theme['yaxis'],
        hoverlabel=theme['hoverlabel'],
        showlegend=bool(color_col),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Add glow effect to bars
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='rgba(0, 212, 255, 0.5)')
        ),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>' if not horizontal 
                      else '<b>%{y}</b><br>Value: %{x:.4f}<extra></extra>'
    )
    
    return fig


def create_neon_line_chart(data, x, y, color=None, title=''):
    """Create a Plotly line chart with neon styling and glow effect."""
    fig = px.line(data, x=x, y=y, color=color,
                  color_discrete_sequence=NEON_COLOR_SEQUENCE,
                  markers=True)
    
    theme = configure_plotly_theme()
    fig.update_layout(
        title=title,
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        font=theme['font'],
        xaxis=theme['xaxis'],
        yaxis=theme['yaxis'],
        legend=theme['legend'],
        hoverlabel=theme['hoverlabel'],
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Style lines with glow
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8, line=dict(width=2, color='rgba(255,255,255,0.3)')),
        hovertemplate='<b>%{x}</b><br>Risk: %{y:.4f}<extra></extra>'
    )
    
    return fig


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
    """Create PyDeck 3D map with hexagons colored and extruded by risk."""
    # Create a copy to avoid modifying original
    pred_df = pred_df.copy()
    
    map_data = []
    for _, row in pred_df.iterrows():
        lat, lng = get_hexagon_center(row['h3_index'])
        risk = row['predicted_risk']
        location_name = get_location_name(row['h3_index'], location_map)
        
        # Enhanced neon color scheme based on risk level
        if risk < 0.05:
            # Low risk: Cyan/Teal glow
            r, g, b, a = 0, 212, 255, 200
        elif risk < 0.15:
            # Medium risk: Amber/Gold glow
            r, g, b, a = 255, 184, 0, 200
        elif risk < 0.3:
            # High risk: Orange/Pink glow
            r, g, b, a = 255, 100, 50, 210
        else:
            # Critical risk: Magenta/Red glow
            r, g, b, a = 255, 51, 102, 230
        
        # Calculate elevation for 3D effect (scale risk to visible height)
        elevation = max(50, risk * 3000)  # Base height of 50, scales up with risk
        
        map_data.append({
            'h3_index': row['h3_index'],
            'location': location_name,
            'lat': lat,
            'lng': lng,
            'risk': float(risk),
            'lower': float(row.get('lower_bound', risk)),
            'upper': float(row.get('upper_bound', risk)),
            'elevation': elevation,
            'r': r,
            'g': g,
            'b': b,
            'a': a
        })
    
    map_df = pd.DataFrame(map_data)
    
    # Enhanced HTML tooltip with dark theme styling
    tooltip_html = """
    <div style="
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.95), rgba(13, 19, 33, 0.95));
        border: 1px solid rgba(0, 212, 255, 0.5);
        border-radius: 12px;
        padding: 15px 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 212, 255, 0.2);
        font-family: 'Segoe UI', Roboto, sans-serif;
        min-width: 200px;
    ">
        <div style="
            font-size: 14px;
            font-weight: 600;
            color: #00D4FF;
            margin-bottom: 8px;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        ">📍 {location}</div>
        <div style="
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #00D4FF, #BD00FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        ">Risk: {risk:.4f}</div>
    """
    
    if show_intervals and 'lower_bound' in pred_df.columns:
        tooltip_html += """
        <div style="
            font-size: 12px;
            color: #A0AEC0;
            padding-top: 6px;
            border-top: 1px solid rgba(0, 212, 255, 0.2);
            margin-top: 6px;
        ">
            <span style="color: #00F5A0;">90% CI:</span> [{lower:.4f}, {upper:.4f}]
        </div>
        """
    
    tooltip_html += "</div>"
    
    layer = pdk.Layer(
        "H3HexagonLayer",
        map_df,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=True,  # Enable 3D extrusion
        get_hexagon="h3_index",
        get_fill_color="[r, g, b, a]",
        get_line_color=[0, 212, 255, 100],  # Cyan outline
        get_elevation="elevation",
        elevation_scale=1,
        line_width_min_pixels=1,
        coverage=0.9,  # Slight gap between hexagons for visual clarity
    )
    
    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10.5,
        pitch=45,  # 3D perspective
        bearing=0
    )
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": tooltip_html, "style": {"backgroundColor": "transparent", "border": "none"}},
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"  # Dark map style
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
            
            # Risk distribution - Styled Plotly chart
            st.subheader("📊 Risk Distribution")
            risk_counts = results['risk_level'].value_counts()
            
            # Create ordered dataframe for risk levels
            risk_order = ['Low', 'Medium', 'High', 'Critical']
            risk_df = pd.DataFrame({
                'level': [lvl for lvl in risk_order if lvl in risk_counts.index],
                'count': [risk_counts.get(lvl, 0) for lvl in risk_order if lvl in risk_counts.index]
            })
            
            # Neon colors matching risk levels
            risk_colors = {
                'Low': '#00D4FF',      # Cyan
                'Medium': '#FFB800',   # Amber
                'High': '#FF6432',     # Orange
                'Critical': '#FF3366'  # Magenta
            }
            
            fig = go.Figure()
            
            for _, row in risk_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row['level']],
                    y=[row['count']],
                    name=row['level'],
                    marker=dict(
                        color=risk_colors.get(row['level'], '#00D4FF'),
                        line=dict(width=2, color='rgba(255, 255, 255, 0.3)')
                    ),
                    hovertemplate=f"<b>{row['level']}</b><br>Count: {row['count']}<extra></extra>"
                ))
            
            theme = configure_plotly_theme()
            fig.update_layout(
                paper_bgcolor=theme['paper_bgcolor'],
                plot_bgcolor=theme['plot_bgcolor'],
                font=theme['font'],
                xaxis=dict(title='Risk Level', **theme['xaxis']),
                yaxis=dict(title='Number of Areas', **theme['yaxis']),
                hoverlabel=theme['hoverlabel'],
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=40),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
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
            # Interactive Plotly SHAP chart with neon styling
            importance_sorted = importance.sort_values('importance', ascending=True)
            
            fig = go.Figure()
            
            # Create gradient colors for bars (cyan to purple)
            n_features = len(importance_sorted)
            colors = []
            for i in range(n_features):
                ratio = i / max(n_features - 1, 1)
                r = int(0 + ratio * 189)  # 00 to BD
                g = int(212 - ratio * 212)  # D4 to 00
                b = int(255)  # FF
                colors.append(f'rgb({r}, {g}, {b})')
            
            fig.add_trace(go.Bar(
                x=importance_sorted['importance'],
                y=importance_sorted['feature'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(width=1, color='rgba(0, 212, 255, 0.5)')
                ),
                hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>'
            ))
            
            theme = configure_plotly_theme()
            fig.update_layout(
                title=dict(text='Feature Importance (SHAP)', font=dict(size=18, color='#00D4FF')),
                paper_bgcolor=theme['paper_bgcolor'],
                plot_bgcolor=theme['plot_bgcolor'],
                font=theme['font'],
                xaxis=dict(title='Mean |SHAP Value|', **theme['xaxis']),
                yaxis=theme['yaxis'],
                hoverlabel=theme['hoverlabel'],
                margin=dict(l=20, r=20, t=50, b=40),
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
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
            
            # Chart heading above the graph
            st.subheader("📈 24-Hour Risk Forecast")
            
            # Interactive Plotly forecast chart with neon styling
            fig = create_neon_line_chart(
                forecast_df, 
                x='hour', 
                y='risk', 
                color='location',
                title=''  # Title is shown as subheader above
            )
            
            # Add area fill for visual impact
            fig.update_traces(fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.05)')
            
            fig.update_layout(
                xaxis_title='Hours from Start',
                yaxis_title='Predicted Risk',
                height=450,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='left',
                    x=0,
                    bgcolor='rgba(19, 27, 46, 0.8)',
                    bordercolor='rgba(0, 212, 255, 0.3)',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot with Plotly
            fig = go.Figure()
            
            # Add scatter points with neon cyan color
            fig.add_trace(go.Scattergl(
                x=y_true,
                y=y_pred,
                mode='markers',
                marker=dict(
                    color='rgba(0, 212, 255, 0.4)',
                    size=5,
                    line=dict(width=0)
                ),
                name='Predictions',
                hovertemplate='<b>Actual:</b> %{x:.4f}<br><b>Predicted:</b> %{y:.4f}<extra></extra>'
            ))
            
            # Add diagonal line
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='#FF3366', width=2, dash='dash'),
                name='Perfect Prediction',
                hoverinfo='skip'
            ))
            
            theme = configure_plotly_theme()
            fig.update_layout(
                title=dict(text='Predicted vs Actual', font=dict(size=16, color='#00D4FF')),
                xaxis_title='Actual',
                yaxis_title='Predicted',
                paper_bgcolor=theme['paper_bgcolor'],
                plot_bgcolor=theme['plot_bgcolor'],
                font=theme['font'],
                xaxis=theme['xaxis'],
                yaxis=theme['yaxis'],
                hoverlabel=theme['hoverlabel'],
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(19, 27, 46, 0.8)',
                    bordercolor='rgba(0, 212, 255, 0.3)'
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution comparison with overlaid histograms
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=y_pred,
                name='Predicted',
                marker_color='rgba(0, 212, 255, 0.6)',
                marker_line=dict(width=1, color='#00D4FF'),
                nbinsx=50,
                opacity=0.7
            ))
            
            fig.add_trace(go.Histogram(
                x=y_true,
                name='Actual',
                marker_color='rgba(189, 0, 255, 0.5)',
                marker_line=dict(width=1, color='#BD00FF'),
                nbinsx=50,
                opacity=0.6
            ))
            
            fig.update_layout(
                title=dict(text='Distribution Comparison', font=dict(size=16, color='#00D4FF')),
                xaxis_title='Value',
                yaxis_title='Frequency',
                barmode='overlay',
                paper_bgcolor=theme['paper_bgcolor'],
                plot_bgcolor=theme['plot_bgcolor'],
                font=theme['font'],
                xaxis=theme['xaxis'],
                yaxis=theme['yaxis'],
                hoverlabel=theme['hoverlabel'],
                legend=dict(
                    bgcolor='rgba(19, 27, 46, 0.8)',
                    bordercolor='rgba(0, 212, 255, 0.3)'
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        residuals = y_true - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals vs Predicted scatter
            fig = go.Figure()
            
            fig.add_trace(go.Scattergl(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(
                    color='rgba(0, 245, 160, 0.4)',
                    size=5
                ),
                name='Residuals',
                hovertemplate='<b>Predicted:</b> %{x:.4f}<br><b>Residual:</b> %{y:.4f}<extra></extra>'
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="#FF3366", line_width=2)
            
            fig.update_layout(
                title=dict(text='Residuals vs Predicted', font=dict(size=16, color='#00D4FF')),
                xaxis_title='Predicted',
                yaxis_title='Residual',
                paper_bgcolor=theme['paper_bgcolor'],
                plot_bgcolor=theme['plot_bgcolor'],
                font=theme['font'],
                xaxis=theme['xaxis'],
                yaxis=theme['yaxis'],
                hoverlabel=theme['hoverlabel'],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residual distribution histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=residuals,
                marker_color='rgba(0, 212, 255, 0.6)',
                marker_line=dict(width=1, color='#00D4FF'),
                nbinsx=50
            ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="#FF3366", line_width=2)
            
            fig.update_layout(
                title=dict(
                    text=f'Residual Distribution (Mean: {residuals.mean():.4f})',
                    font=dict(size=16, color='#00D4FF')
                ),
                xaxis_title='Residual',
                yaxis_title='Frequency',
                paper_bgcolor=theme['paper_bgcolor'],
                plot_bgcolor=theme['plot_bgcolor'],
                font=theme['font'],
                xaxis=theme['xaxis'],
                yaxis=theme['yaxis'],
                hoverlabel=theme['hoverlabel'],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        sample['prediction'] = y_pred
        sample['error'] = np.abs(y_true - y_pred)
        
        hourly_performance = sample.groupby('hour_of_day').agg({
            'error': 'mean',
            'prediction': 'mean',
            'accident_count': 'mean'
        }).reset_index()
        
        # Create bar chart with gradient colors
        colors = []
        for i, err in enumerate(hourly_performance['error']):
            # Color from green (low error) to red (high error)
            max_err = hourly_performance['error'].max()
            min_err = hourly_performance['error'].min()
            ratio = (err - min_err) / (max_err - min_err) if max_err != min_err else 0
            colors.append(f'rgba({int(255 * ratio)}, {int(245 * (1-ratio))}, {int(160 * (1-ratio))}, 0.8)')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hourly_performance['hour_of_day'],
            y=hourly_performance['error'],
            marker=dict(
                color=colors,
                line=dict(width=1, color='rgba(0, 212, 255, 0.5)')
            ),
            hovertemplate='<b>Hour: %{x}:00</b><br>MAE: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text='Prediction Error by Hour', font=dict(size=16, color='#00D4FF')),
            xaxis_title='Hour of Day',
            yaxis_title='Mean Absolute Error',
            paper_bgcolor=theme['paper_bgcolor'],
            plot_bgcolor=theme['plot_bgcolor'],
            font=theme['font'],
            xaxis=dict(
                **theme['xaxis'],
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis=theme['yaxis'],
            hoverlabel=theme['hoverlabel'],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


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
                # Neon colors for clusters
                neon_cluster_colors = [
                    [0, 212, 255],    # Cyan
                    [189, 0, 255],    # Purple
                    [255, 0, 110],    # Pink
                    [0, 245, 160],    # Green
                    [255, 184, 0],    # Amber
                    [255, 51, 102],   # Red
                    [0, 229, 204],    # Teal
                    [255, 140, 0],    # Orange
                    [138, 43, 226],   # Blue Violet
                    [50, 205, 50]     # Lime
                ]
                
                for _, row in hex_df.iterrows():
                    h3_idx = row['h3_index']
                    cluster = int(row['cluster'])
                    lat, lng = get_hexagon_center(h3_idx)
                    
                    # Use accident count for elevation
                    elevation = max(100, row.get('accident_count_mean', 0) * 500)
                    
                    map_data.append({
                        'h3_index': h3_idx,
                        'lat': lat,
                        'lng': lng,
                        'cluster': cluster,
                        'elevation': elevation,
                        'color': neon_cluster_colors[cluster % len(neon_cluster_colors)] + [200]
                    })
                
                # Enhanced 3D layer
                layer = pdk.Layer(
                    "H3HexagonLayer",
                    pd.DataFrame(map_data),
                    pickable=True,
                    stroked=True,
                    filled=True,
                    extruded=True,
                    get_hexagon="h3_index",
                    get_fill_color="color",
                    get_line_color=[0, 212, 255, 100],
                    get_elevation="elevation",
                    elevation_scale=1,
                    coverage=0.9
                )
                
                # Enhanced tooltip
                cluster_tooltip = """
                <div style="
                    background: linear-gradient(135deg, rgba(19, 27, 46, 0.95), rgba(13, 19, 33, 0.95));
                    border: 1px solid rgba(0, 212, 255, 0.5);
                    border-radius: 10px;
                    padding: 12px 16px;
                    font-family: 'Segoe UI', sans-serif;
                ">
                    <div style="color: #00D4FF; font-size: 18px; font-weight: 600;">
                        Cluster {cluster}
                    </div>
                </div>
                """
                
                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(
                        latitude=40.7128, 
                        longitude=-74.0060, 
                        zoom=10.5,
                        pitch=45,
                        bearing=0
                    ),
                    tooltip={"html": cluster_tooltip, "style": {"backgroundColor": "transparent"}},
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
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
    """Create map data for a scenario with neon colors."""
    map_data = []
    for _, row in results_df.iterrows():
        lat, lng = get_hexagon_center(row['h3_index'])
        risk = float(row['predicted_risk'])
        location_name = get_location_name(row['h3_index'], location_map)
        
        # Neon color scheme based on risk (matching main risk map)
        if risk < 0.05:
            r, g, b = 0, 212, 255  # Cyan
        elif risk < 0.15:
            r, g, b = 255, 184, 0  # Amber
        elif risk < 0.3:
            r, g, b = 255, 100, 50  # Orange
        else:
            r, g, b = 255, 51, 102  # Magenta
        
        # Elevation for 3D effect
        elevation = max(50, risk * 2000)
        
        map_data.append({
            'h3_index': row['h3_index'],
            'location': location_name,
            'risk': risk,
            'elevation': elevation,
            'r': r, 'g': g, 'b': b, 'a': 200
        })
    
    return pd.DataFrame(map_data)


def build_hex_map(map_df, title):
    """Build a PyDeck 3D map with neon styling."""
    layer = pdk.Layer(
        "H3HexagonLayer",
        map_df,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=True,
        get_hexagon="h3_index",
        get_fill_color="[r, g, b, a]",
        get_line_color=[0, 212, 255, 80],
        get_elevation="elevation",
        elevation_scale=1,
        line_width_min_pixels=1,
        coverage=0.9
    )
    
    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10.5,
        pitch=40,
        bearing=0
    )
    
    # Enhanced tooltip with neon styling
    tooltip_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.95), rgba(13, 19, 33, 0.95));
        border: 1px solid rgba(0, 212, 255, 0.5);
        border-radius: 10px;
        padding: 12px 16px;
        font-family: 'Segoe UI', sans-serif;
    ">
        <div style="color: #BD00FF; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
            {title}
        </div>
        <div style="color: #00D4FF; font-size: 14px;">
            📍 {{location}}
        </div>
        <div style="
            font-size: 18px;
            font-weight: 700;
            background: linear-gradient(135deg, #00D4FF, #BD00FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">
            Risk: {{risk:.4f}}
        </div>
    </div>
    """
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": tooltip_html, "style": {"backgroundColor": "transparent"}},
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
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
