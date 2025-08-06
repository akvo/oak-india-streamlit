import streamlit as st
st.set_page_config(
    page_title="OAK India WASH Data Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import scipy.stats as stats
from scipy.stats import zscore
import requests
import altair as alt
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jinja2
import branca.colormap as cm
import warnings
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import base64
from difflib import SequenceMatcher
import tempfile
import webbrowser
from src.visualization import HexbinMapGenerator

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
BENGALI_FONT = 'NotoSansBengali'

# GPS column mapping from Bengali to English
GPS_COLUMN_MAPPING = {
    '_বাড়ির অবস্থান _latitude': 'house_latitude',
    '_বাড়ির অবস্থান _longitude': 'house_longitude',
    '_বাড়ির অবস্থান _altitude': 'house_altitude',
    '_বাড়ির অবস্থান _precision': 'house_precision',
    '_টিউবওয়েল/ট্যাপের অবস্থান_latitude': 'tubewell_latitude',
    '_টিউবওয়েল/ট্যাপের অবস্থান_longitude': 'tubewell_longitude',
    '_টিউবওয়েল/ট্যাপের অবস্থান_altitude': 'tubewell_altitude',
    '_টিউবওয়েল/ট্যাপের অবস্থান_precision': 'tubewell_precision',
    '_খালের অবস্থান _latitude': 'canal_latitude',
    '_খালের অবস্থান _longitude': 'canal_longitude',
    '_খালের অবস্থান _altitude': 'canal_altitude',
    '_খালের অবস্থান _precision': 'canal_precision'
}

# Load GeoJSON data
@st.cache_data
def load_geojson():
    return requests.get('https://raw.githubusercontent.com/akvo/oak-india/refs/heads/main/data/villages.geojson').json()

# Load water usage mapping config
@st.cache_data
def load_water_usage_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'water_usage_mapping.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def string_similarity(a, b):
    """
    Calculate string similarity using SequenceMatcher.
    Returns a value between 0 and 1, where 1 means identical strings.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_column_match(target_col, available_cols, threshold=0.8):
    """
    Find the best matching column from available columns using string similarity.
    Returns the best match if similarity is above threshold, otherwise returns None.
    """
    if not available_cols:
        return None
    
    # Calculate similarity scores for all available columns
    scores = [(col, string_similarity(target_col, col)) for col in available_cols]
    
    # Find the best match
    if not scores:
        return None
    
    best_match, best_score = max(scores, key=lambda x: x[1])
    
    # Convert threshold from percentage to ratio (e.g., 80% -> 0.8)
    threshold_ratio = threshold / 100.0
    
    return best_match if best_score >= threshold_ratio else None

def get_matching_columns(target_cols, available_cols, threshold=80):
    """
    Find best matches for a list of target columns from available columns.
    Returns a dictionary mapping target columns to their best matches.
    """
    matches = {}
    unmatched = []
    
    for target_col in target_cols:
        best_match = find_best_column_match(target_col, available_cols, threshold)
        if best_match:
            matches[target_col] = best_match
            # Remove matched column from available columns to prevent duplicate matches
            available_cols.remove(best_match)
        else:
            unmatched.append(target_col)
    
    return matches, unmatched

# Load CSV data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_csv_data(file_url):
    df = pd.read_csv(file_url, encoding='utf-8', low_memory=False)
    
    # Clean column names: remove colons and strip whitespace
    df.columns = df.columns.str.replace(':', '', regex=False).str.strip()
    
    # Create the 'loc-res-sub' column
    columns_to_concatenate = [
        'গ্রামের নাম',
        'পাড়ার নাম',
        'উত্তরদাতার নাম(দিদির নাম)',
        'তথ্য সংগ্রহকারীর নাম'
    ]
    
    df['loc-res-sub'] = df.apply(lambda row: '$'.join(
        str(row[col]) if col in row.index and pd.notna(row[col]) else '' for col in columns_to_concatenate
    ), axis=1)
    
    # Rename only GPS columns to English while preserving Bengali data
    df = df.rename(columns=GPS_COLUMN_MAPPING)
    
    # Add water usage columns based on config
    water_usage_config = load_water_usage_config()
    
    # Get dataset name from the file URL
    dataset_name = next((name for name, url in csv_files.items() if url == file_url), None)
    
    if not dataset_name:
        st.error(f"Could not find dataset name for URL: {file_url}")
        return df
        
    if dataset_name not in water_usage_config:
        st.warning(f"No water usage mapping found for dataset: {dataset_name}")
        return df
    
    # Get the specific mapping for this dataset
    dataset_config = water_usage_config[dataset_name]
    
    # Get available columns for fuzzy matching
    available_cols = df.columns.tolist()
    
    # Clean column names in config (strip whitespace)
    surface_cols = [col.strip() for col in dataset_config['surface_water_usage']['columns']]
    ground_cols = [col.strip() for col in dataset_config['ground_water_usage']['columns']]
    
    # Find best matches for surface water columns
    surface_matches, unmatched_surface = get_matching_columns(surface_cols, available_cols.copy())
    if unmatched_surface:
        st.warning(f"Could not find fuzzy matches for {len(unmatched_surface)} surface water columns")
    
    # Find best matches for ground water columns
    ground_matches, unmatched_ground = get_matching_columns(ground_cols, available_cols.copy())
    if unmatched_ground:
        st.warning(f"Could not find fuzzy matches for {len(unmatched_ground)} ground water columns")
    
    # Create surface water usage column
    available_surface_cols = list(surface_matches.values())
    if available_surface_cols:
        df['surface_water_usage'] = df[available_surface_cols].fillna(0).sum(axis=1)
    else:
        df['surface_water_usage'] = 0
        st.warning(f"No surface water usage columns found for dataset {dataset_name}")
    
    # Create ground water usage column
    available_ground_cols = list(ground_matches.values())
    if available_ground_cols:
        df['ground_water_usage'] = df[available_ground_cols].fillna(0).sum(axis=1)
    else:
        df['ground_water_usage'] = 0
        st.warning(f"No ground water usage columns found for dataset {dataset_name}")
    
    # Add total water usage column
    df['total_water_usage'] = df['surface_water_usage'] + df['ground_water_usage']
    
    # Log any completely unmatched columns
    if unmatched_surface or unmatched_ground:
        st.warning(f"Some water usage columns could not be matched in dataset {dataset_name}")
    
    return df

# Define CSV file URLs
csv_files = {
    'HH - Gosaba 1': 'https://raw.githubusercontent.com/akvo/oak-india/refs/heads/main/data/hh-gosaba-1-cleaned.csv',
    'HH - Gosaba 2': 'https://raw.githubusercontent.com/akvo/oak-india/refs/heads/main/data/hh-gosaba-2-cleaned.csv',
    'HH - Pathar': 'https://raw.githubusercontent.com/akvo/oak-india/refs/heads/main/data/hh-pathar.csv',
}

# Sidebar for data selection
st.sidebar.title("Data Selection")
selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    options=list(csv_files.keys())
)

# Load data
geojson_data = load_geojson()
df = load_csv_data(csv_files[selected_dataset])

# Debug logging for columns

# Find latitude and longitude columns
lat_cols = [col for col in df.columns if 'latitude' in col.lower()]
lon_cols = [col for col in df.columns if 'longitude' in col.lower()]

# Store coordinate columns in session state to prevent reloads
if 'lat_col' not in st.session_state or 'lon_col' not in st.session_state:
    if lat_cols and lon_cols:
        # Try to find house location columns first
        house_lat_cols = [col for col in lat_cols if 'house' in col.lower()]
        house_lon_cols = [col for col in lon_cols if 'house' in col.lower()]
        
        if house_lat_cols and house_lon_cols:
            st.session_state.lat_col = house_lat_cols[0]
            st.session_state.lon_col = house_lon_cols[0]
        else:
            # Fallback to any available coordinates
            st.session_state.lat_col = lat_cols[0]
            st.session_state.lon_col = lon_cols[0]
    else:
        st.error("Could not find any coordinate columns in the data")
        st.stop()

# Use session state variables
lat_col = st.session_state.lat_col
lon_col = st.session_state.lon_col

# Cache the map creation
@st.cache_data(ttl=3600)  # Cache for 1 hour
def create_map_data(df, lat_col, lon_col, selected_column):
    df_cleaned = df.dropna(subset=[selected_column, lat_col, lon_col])
    return df_cleaned

# Initialize session state for filtered dataframe if it doesn't exist
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None

# Initialize session state for outlier detection
if 'outlier_method' not in st.session_state:
    st.session_state.outlier_method = "None"

# Initialize session state for aggregation method
if 'agg_method' not in st.session_state:
    st.session_state.agg_method = "mean"

# Main app title
st.title("Household Data Analysis Dashboard")
st.write(f"Currently viewing: {selected_dataset}")
st.write(f"Number of records: {len(df)}")

# Add data filter in header in a single row
st.header("Filter Data")

# Get categorical columns, excluding specific columns
exclude_cols = ['start', 'end', '_uuid', '_id', '_submission_time', '_validation_status',
                '_notes', '_status', '_submitted_by', '__version__', '_tags', '_index']
cat_cols = [col for col in df.select_dtypes(include=['object', 'period[M]']).columns 
            if col not in exclude_cols]

# Create a single row with columns for the filter
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    selected_col = st.selectbox("Select Column to Filter:", cat_cols)

# Get value counts for selected column
value_counts = df[selected_col].value_counts()
sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

# Create multiselect with counts
options = [f"{val} (Count: {count})" for val, count in sorted_values]

with col2:
    selected_values = st.multiselect("Select Values:", options)

with col3:
    if selected_values:
        # Extract original values from the selected options
        filter_values = [val.split(" (Count: ")[0] for val in selected_values]
        st.session_state.filtered_df = df[df[selected_col].isin(filter_values)]
        st.write(f"Filtered: {len(st.session_state.filtered_df)} records")
    else:
        st.session_state.filtered_df = df
        st.write(f"Showing all {len(df)} records")
current_df = st.session_state.filtered_df if st.session_state.filtered_df is not None else df

# Add outlier detection options in sidebar
with st.sidebar.expander("Outlier Detection", expanded=True):
    st.session_state.agg_method = st.radio(
        "Aggregation Method",
        ["mean", "median", "sum"],
        format_func=lambda x: x.capitalize(),
        help="Choose how to aggregate the numeric values",
        key="agg_method_radio"
    )
    
    st.session_state.outlier_method = st.radio(
        "Outlier Detection Method",
        ["None", "Z-score", "IQR", "DBSCAN"],
        help="Choose a method to detect and remove outliers",
        key="outlier_method_radio"
    )
    
    if st.session_state.outlier_method == "Z-score":
        z_threshold = st.slider(
            "Z-score threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.05,
            help="Points with absolute Z-score above this threshold will be considered outliers"
        )
    
    elif st.session_state.outlier_method == "IQR":
        iqr_multiplier = st.slider(
            "IQR multiplier",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.05,
            help="Points beyond IQR * multiplier will be considered outliers"
        )
    
    elif st.session_state.outlier_method == "DBSCAN":
        eps = st.slider(
            "Epsilon (neighborhood size)",
            min_value=0.01,
            max_value=2.0,
            value=0.5,
            step=0.01,
            help="Maximum distance between points to be considered neighbors"
        )
        min_samples = st.slider(
            "Minimum samples",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Minimum number of points required to form a cluster"
        )

# Function to detect outliers using Z-score method
def detect_outliers_zscore(series, threshold=3.0):
    """
    Detect outliers in a pandas Series using Z-score method.
    
    Args:
        series: pandas Series to check for outliers
        threshold: Z-score threshold (default: 3.0)
    
    Returns:
        pandas Series: Boolean mask where True indicates outliers
    """
    if len(series) == 0:
        return pd.Series(False, index=series.index)
    
    # Check if data is nearly constant
    if (series.max() - series.min()) < 1e-10:
        return pd.Series(False, index=series.index)
    
    z_scores = np.abs(stats.zscore(series))
    return z_scores > threshold

# Function to detect outliers
def detect_outliers(df, x_col, y_col=None, method="Z-score", **kwargs):
    if method == "None":
        return df, pd.Series(False, index=df.index)
    
    df_outliers = df.copy()
    
    def is_near_constant(series, threshold=1e-10):
        """Check if a series is nearly constant (all values are very close to each other)"""
        return (series.max() - series.min()) < threshold
    
    if method == "Z-score":
        if y_col is None:
            # For bar chart (single dimension)
            if is_near_constant(df[x_col]):
                # If data is nearly constant, no outliers
                return df_outliers, pd.Series(False, index=df.index)
            z_scores = np.abs(stats.zscore(df[x_col]))
            outlier_mask = z_scores > kwargs.get('z_threshold', 3.0)
        else:
            # For scatter plot (two dimensions)
            if is_near_constant(df[x_col]) and is_near_constant(df[y_col]):
                # If both dimensions are nearly constant, no outliers
                return df_outliers, pd.Series(False, index=df.index)
            z_scores_x = np.abs(stats.zscore(df[x_col]))
            z_scores_y = np.abs(stats.zscore(df[y_col]))
            outlier_mask = (z_scores_x > kwargs.get('z_threshold', 3.0)) | (z_scores_y > kwargs.get('z_threshold', 3.0))
    
    elif method == "IQR":
        if y_col is None:
            # For bar chart (single dimension)
            if is_near_constant(df[x_col]):
                # If data is nearly constant, no outliers
                return df_outliers, pd.Series(False, index=df.index)
            Q1 = df[x_col].quantile(0.25)
            Q3 = df[x_col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:  # If IQR is 0, data is constant
                return df_outliers, pd.Series(False, index=df.index)
            outlier_mask = (
                (df[x_col] < (Q1 - kwargs.get('iqr_multiplier', 1.5) * IQR)) |
                (df[x_col] > (Q3 + kwargs.get('iqr_multiplier', 1.5) * IQR))
            )
        else:
            # For scatter plot (two dimensions)
            if is_near_constant(df[x_col]) and is_near_constant(df[y_col]):
                # If both dimensions are nearly constant, no outliers
                return df_outliers, pd.Series(False, index=df.index)
            
            # Check x dimension
            Q1_x = df[x_col].quantile(0.25)
            Q3_x = df[x_col].quantile(0.75)
            IQR_x = Q3_x - Q1_x
            
            # Check y dimension
            Q1_y = df[y_col].quantile(0.25)
            Q3_y = df[y_col].quantile(0.75)
            IQR_y = Q3_y - Q1_y
            
            # If either IQR is 0, that dimension is constant
            if IQR_x == 0 and IQR_y == 0:
                return df_outliers, pd.Series(False, index=df.index)
            
            outlier_mask = (
                (df[x_col] < (Q1_x - kwargs.get('iqr_multiplier', 1.5) * IQR_x)) |
                (df[x_col] > (Q3_x + kwargs.get('iqr_multiplier', 1.5) * IQR_x)) |
                (df[y_col] < (Q1_y - kwargs.get('iqr_multiplier', 1.5) * IQR_y)) |
                (df[y_col] > (Q3_y + kwargs.get('iqr_multiplier', 1.5) * IQR_y))
            )
    
    elif method == "DBSCAN":
        if y_col is None:
            # For bar chart, use only x dimension
            if is_near_constant(df[x_col]):
                # If data is nearly constant, no outliers
                return df_outliers, pd.Series(False, index=df.index)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[[x_col]])
            dbscan = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
            clusters = dbscan.fit_predict(scaled_data)
        else:
            # For scatter plot, use both dimensions
            if is_near_constant(df[x_col]) and is_near_constant(df[y_col]):
                # If both dimensions are nearly constant, no outliers
                return df_outliers, pd.Series(False, index=df.index)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[[x_col, y_col]])
            dbscan = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
            clusters = dbscan.fit_predict(scaled_data)
        
        outlier_mask = clusters == -1
    
    return df_outliers, outlier_mask

# Initialize the hexbin map generator
hexbin_map = HexbinMapGenerator('https://raw.githubusercontent.com/akvo/oak-india/refs/heads/main/data/villages.geojson')

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Bar Chart", "Scatter Plot", "Map", "Table View", "Hexbin Map"])

# Tab 1: Bar Chart
with tab1:
    st.header("Bar Chart Analysis")
    
    # Get categorical and numeric columns from filtered dataframe, excluding specific columns
    exclude_cols = [
        # System columns
        'start', 'end', '_uuid', '_id', '_submission_time', '_validation_status',
        '_notes', '_status', '_submitted_by', '__version__', '_tags', '_index',
        # GPS columns
        'house_latitude', 'house_longitude', 'house_altitude', 'house_precision',
        'tubewell_latitude', 'tubewell_longitude', 'tubewell_altitude', 'tubewell_precision',
        'canal_latitude', 'canal_longitude', 'canal_altitude', 'canal_precision',
        # Original Bengali GPS column names
        '_বাড়ির অবস্থান _latitude', '_বাড়ির অবস্থান _longitude', '_বাড়ির অবস্থান _altitude', '_বাড়ির অবস্থান _precision',
        '_টিউবওয়েল/ট্যাপের অবস্থান_latitude', '_টিউবওয়েল/ট্যাপের অবস্থান_longitude', '_টিউবওয়েল/ট্যাপের অবস্থান_altitude', '_টিউবওয়েল/ট্যাপের অবস্থান_precision',
        '_খালের অবস্থান _latitude', '_খালের অবস্থান _longitude', '_খালের অবস্থান _altitude', '_খালের অবস্থান _precision'
    ]
    
    # Define computed columns that should appear first
    computed_columns = ['surface_water_usage', 'ground_water_usage', 'total_water_usage']
    
    # Get categorical and numeric columns
    categorical_cols = [col for col in current_df.select_dtypes(exclude=['int64', 'float64']).columns 
                       if col not in exclude_cols]
    numeric_cols = [col for col in current_df.select_dtypes(include=['int64', 'float64']).columns 
                   if col not in exclude_cols]
    
    # Sort numeric columns to put computed columns first
    computed_numeric = [col for col in computed_columns if col in numeric_cols]
    other_numeric = [col for col in numeric_cols if col not in computed_columns]
    numeric_cols = computed_numeric + sorted(other_numeric)
    
    col1, col2 = st.columns(2)
    
    with col1:
        cat_col = st.selectbox("Group by:", categorical_cols)
    with col2:
        num_col = st.selectbox("Average of:", numeric_cols)
    
    if cat_col and num_col:
        # Create aggregation using filtered dataframe
        mask = ~(current_df[cat_col].isna() | current_df[num_col].isna())
        df_clean = current_df[mask]
        
        # Apply outlier detection
        if st.session_state.outlier_method != "None":
            # Detect outliers within each category
            df_normal = pd.DataFrame()
            df_outliers = pd.DataFrame()
            
            for category in df_clean[cat_col].unique():
                category_data = df_clean[df_clean[cat_col] == category]
                _, outlier_mask = detect_outliers(
                    category_data, 
                    num_col, 
                    method=st.session_state.outlier_method,
                    z_threshold=z_threshold if st.session_state.outlier_method == "Z-score" else None,
                    iqr_multiplier=iqr_multiplier if st.session_state.outlier_method == "IQR" else None,
                    eps=eps if st.session_state.outlier_method == "DBSCAN" else None,
                    min_samples=min_samples if st.session_state.outlier_method == "DBSCAN" else None
                )
                
                df_normal = pd.concat([df_normal, category_data[~outlier_mask]])
                df_outliers = pd.concat([df_outliers, category_data[outlier_mask]])
            
            # Create aggregations
            # 1. All data (including outliers)
            all_agg = df_clean.groupby(cat_col)[num_col].agg(['sum', 'mean', 'median', 'count']).reset_index()
            all_agg['mean'] = all_agg['mean'].round(2)
            all_agg['median'] = all_agg['median'].round(2)
            all_agg['sum'] = all_agg['sum'].round(2)
            
            # 2. Data without outliers
            normal_agg = df_normal.groupby(cat_col)[num_col].agg(['sum', 'mean', 'median', 'count']).reset_index()
            normal_agg['mean'] = normal_agg['mean'].round(2)
            normal_agg['median'] = normal_agg['median'].round(2)
            normal_agg['sum'] = normal_agg['sum'].round(2)
            
            # Ensure all categories are present in both aggregations
            all_categories = all_agg[cat_col].unique()
            normal_agg = normal_agg.set_index(cat_col).reindex(all_categories).reset_index()
            normal_agg = normal_agg.fillna(0)  # Fill missing values with 0
            
            # Filter and sort both aggregations
            all_agg = all_agg[all_agg['count'] >= 10].sort_values(st.session_state.agg_method, ascending=True)
            normal_agg = normal_agg[normal_agg.index.isin(all_agg.index)].sort_values(st.session_state.agg_method, ascending=True)
            
            # Create bar chart using Plotly
            fig = go.Figure()
            
            # Add bars for all data
            fig.add_trace(go.Bar(
                x=all_agg[st.session_state.agg_method],
                y=all_agg[cat_col],
                orientation='h',
                text=all_agg[st.session_state.agg_method].round(2),
                textposition='auto',
                name=f'{st.session_state.agg_method.capitalize()} (All Data)',
                marker_color='#1f77b4',
                customdata=np.column_stack((
                    all_agg['mean'].values,
                    all_agg['median'].values,
                    all_agg['sum'].values,
                    all_agg['count'].values
                )),
                hovertemplate=(
                    f"<b>{cat_col}:</b> %{{y}}<br>"
                    f"<b>{st.session_state.agg_method.capitalize()} {num_col} (All Data):</b> %{{x:.2f}}<br>"
                    f"<b>Mean:</b> %{{customdata[0]:.2f}}<br>"
                    f"<b>Median:</b> %{{customdata[1]:.2f}}<br>"
                    f"<b>Sum:</b> %{{customdata[2]:.2f}}<br>"
                    f"<b>Total Count:</b> %{{customdata[3]}}<br>"
                    "<extra></extra>"
                )
            ))
            
            # Add bars for data without outliers
            fig.add_trace(go.Bar(
                x=normal_agg[st.session_state.agg_method],
                y=normal_agg[cat_col],
                orientation='h',
                text=normal_agg[st.session_state.agg_method].round(2),
                textposition='auto',
                name=f'{st.session_state.agg_method.capitalize()} (Without Outliers)',
                marker_color='#2ca02c',  # Green color
                opacity=0.7,
                customdata=np.column_stack((
                    normal_agg['mean'].values,
                    normal_agg['median'].values,
                    normal_agg['sum'].values,
                    normal_agg['count'].values,
                    (all_agg['count'] - normal_agg['count']).values
                )),
                hovertemplate=(
                    f"<b>{cat_col}:</b> %{{y}}<br>"
                    f"<b>{st.session_state.agg_method.capitalize()} {num_col} (Without Outliers):</b> %{{x:.2f}}<br>"
                    f"<b>Mean:</b> %{{customdata[0]:.2f}}<br>"
                    f"<b>Median:</b> %{{customdata[1]:.2f}}<br>"
                    f"<b>Sum:</b> %{{customdata[2]:.2f}}<br>"
                    f"<b>Normal Count:</b> %{{customdata[3]}}<br>"
                    f"<b>Outlier Count:</b> %{{customdata[4]}}<br>"
                    "<extra></extra>"
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=f'{st.session_state.agg_method.capitalize()} {num_col} by {cat_col}',
                xaxis_title=f'{st.session_state.agg_method.capitalize()} {num_col}',
                yaxis_title=cat_col,
                height=max(400, len(all_agg) * 25),
                showlegend=True,
                hovermode='closest',
                margin=dict(l=20, r=20, t=40, b=20),
                barmode='group',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                )
            )
            
            # Display outlier statistics
            total_points = len(df_clean)
            outlier_points = len(df_outliers)
            st.sidebar.write(f"Bar Chart Statistics:")
            st.sidebar.write(f"Total points: {total_points}")
            st.sidebar.write(f"Outliers detected: {outlier_points}")
            st.sidebar.write(f"Outlier percentage: {(outlier_points / total_points * 100):.1f}%")
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            # No outlier detection, show all data as one bar
            df_agg = df_clean.groupby(cat_col)[num_col].agg(['sum', 'mean', 'median', 'count']).reset_index()
            df_agg['mean'] = df_agg['mean'].round(2)
            df_agg['median'] = df_agg['median'].round(2)
            df_agg['sum'] = df_agg['sum'].round(2)
            df_agg = df_agg[df_agg['count'] >= 10]
            df_agg = df_agg.sort_values(st.session_state.agg_method, ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_agg[st.session_state.agg_method],
                y=df_agg[cat_col],
                orientation='h',
                text=df_agg[st.session_state.agg_method].round(2),
                textposition='auto',
                name=f'{st.session_state.agg_method.capitalize()}',
                marker_color='#1f77b4',
                customdata=np.column_stack((
                    df_agg['mean'].values,
                    df_agg['median'].values,
                    df_agg['sum'].values,
                    df_agg['count'].values
                )),
                hovertemplate=(
                    f"<b>{cat_col}:</b> %{{y}}<br>"
                    f"<b>{st.session_state.agg_method.capitalize()} {num_col}:</b> %{{x:.2f}}<br>"
                    f"<b>Mean:</b> %{{customdata[0]:.2f}}<br>"
                    f"<b>Median:</b> %{{customdata[1]:.2f}}<br>"
                    f"<b>Sum:</b> %{{customdata[2]:.2f}}<br>"
                    f"<b>Count:</b> %{{customdata[3]}}<br>"
                    "<extra></extra>"
                )
            ))
            
            fig.update_layout(
                title=f'{st.session_state.agg_method.capitalize()} {num_col} by {cat_col}',
                xaxis_title=f'{st.session_state.agg_method.capitalize()} {num_col}',
                yaxis_title=cat_col,
                height=max(400, len(df_agg) * 25),
                showlegend=False,
                hovermode='closest',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Scatter Plot
with tab2:
    st.header("Scatter Plot Analysis")
    
    # Get numeric columns from filtered dataframe
    numeric_cols = [col for col in current_df.select_dtypes(include=['int64', 'float64']).columns 
                   if col not in exclude_cols]  # Use the same exclude_cols as bar chart
    
    # Sort numeric columns to put computed columns first
    computed_numeric = [col for col in computed_columns if col in numeric_cols]
    other_numeric = [col for col in numeric_cols if col not in computed_columns]
    numeric_cols = computed_numeric + sorted(other_numeric)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("X-axis:", numeric_cols, key='scatter_x')
    with col2:
        y_col = st.selectbox("Y-axis:", numeric_cols, key='scatter_y')
    
    if x_col and y_col:
        # Clean data using filtered dataframe and create a proper copy
        mask = ~(current_df[x_col].isna() | current_df[y_col].isna())
        df_clean = current_df[mask].copy().head(5000)  # Create explicit copy and limit to 5000 points
        
        # Add visualization options in sidebar
        with st.sidebar.expander("Scatter Plot Options", expanded=True):
            show_jitter = st.checkbox("Add jitter to overlapping points", value=True)
            size_by_frequency = st.checkbox("Size points by frequency", value=True)
            jitter_amount = st.slider(
                "Jitter amount",
                min_value=0.0,
                max_value=0.1,
                value=0.02,
                step=0.001,
                help="Amount of random offset to add to overlapping points"
            )
        
        # Create scatter plot using Plotly
        fig = go.Figure()
        
        # Apply outlier detection if selected
        if st.session_state.outlier_method != "None":
            # Detect outliers
            df_clean, outlier_mask = detect_outliers(
                df_clean,
                x_col,
                y_col,
                method=st.session_state.outlier_method,
                z_threshold=z_threshold if st.session_state.outlier_method == "Z-score" else None,
                iqr_multiplier=iqr_multiplier if st.session_state.outlier_method == "IQR" else None,
                eps=eps if st.session_state.outlier_method == "DBSCAN" else None,
                min_samples=min_samples if st.session_state.outlier_method == "DBSCAN" else None
            )
            
            # Split data into normal points and outliers using .loc
            df_normal = df_clean.loc[~outlier_mask].copy()
            df_outliers = df_clean.loc[outlier_mask].copy()
            
            # Calculate frequency for both normal and outlier points
            df_normal.loc[:, 'point_freq'] = df_normal.groupby([x_col, y_col])[x_col].transform('count')
            df_outliers.loc[:, 'point_freq'] = df_outliers.groupby([x_col, y_col])[x_col].transform('count')
            
            # Add jitter if enabled
            if show_jitter:
                x_range = df_clean[x_col].max() - df_clean[x_col].min()
                y_range = df_clean[y_col].max() - df_clean[y_col].min()
                
                # Add jitter to normal points using .loc
                x_jitter = np.random.normal(0, x_range * jitter_amount, len(df_normal))
                y_jitter = np.random.normal(0, y_range * jitter_amount, len(df_normal))
                df_normal.loc[:, 'x_jittered'] = df_normal[x_col] + x_jitter
                df_normal.loc[:, 'y_jittered'] = df_normal[y_col] + y_jitter
                
                # Add jitter to outlier points using .loc
                x_jitter = np.random.normal(0, x_range * jitter_amount, len(df_outliers))
                y_jitter = np.random.normal(0, y_range * jitter_amount, len(df_outliers))
                df_outliers.loc[:, 'x_jittered'] = df_outliers[x_col] + x_jitter
                df_outliers.loc[:, 'y_jittered'] = df_outliers[y_col] + y_jitter
            else:
                df_normal.loc[:, 'x_jittered'] = df_normal[x_col]
                df_normal.loc[:, 'y_jittered'] = df_normal[y_col]
                df_outliers.loc[:, 'x_jittered'] = df_outliers[x_col]
                df_outliers.loc[:, 'y_jittered'] = df_outliers[y_col]
            
            # Get unique villages for coloring
            villages = df_clean['গ্রামের নাম'].unique()
            colors = px.colors.qualitative.Set3
            
            # Add normal points
            for i, village in enumerate(villages):
                village_data = df_normal[df_normal['গ্রামের নাম'] == village]
                if len(village_data) > 0:
                    color = colors[i % len(colors)]
                    if size_by_frequency:
                        sizes = 5 + (village_data['point_freq'] * 2)  # Base size + frequency scaling
                        sizes = np.clip(sizes, 5, 20)  # Limit size between 5 and 20
                    else:
                        sizes = 8
                    
                    fig.add_trace(go.Scatter(
                        x=village_data['x_jittered'],
                        y=village_data['y_jittered'],
                        mode='markers',
                        name=village,
                        marker=dict(
                            color=color,
                            size=sizes,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        customdata=np.stack((
                            village_data[x_col],
                            village_data[y_col],
                            village_data['point_freq'],
                            village_data['loc-res-sub'],
                            village_data['_uuid']
                        ), axis=1),
                        hovertemplate=(
                            f"<b>Village:</b> {village}<br>"
                            f"<b>{x_col}:</b> %{{customdata[0]:.2f}}<br>"
                            f"<b>{y_col}:</b> %{{customdata[1]:.2f}}<br>"
                            f"<b>Number of overlapping points:</b> %{{customdata[2]}}<br>"
                            f"<b>Location-Respondent:</b> %{{customdata[3]}}<br>"
                            f"<b>UUID:</b> %{{customdata[4]}}<br>"
                            "<extra></extra>"
                        )
                    ))
            
            # Add outlier points
            for village in df_outliers['গ্রামের নাম'].unique():
                village_outliers = df_outliers[df_outliers['গ্রামের নাম'] == village]
                if len(village_outliers) > 0:
                    color = colors[list(villages).index(village) % len(colors)]
                    if size_by_frequency:
                        sizes = 5 + (village_outliers['point_freq'] * 2)
                        sizes = np.clip(sizes, 5, 20)
                    else:
                        sizes = 8
                    
                    fig.add_trace(go.Scatter(
                        x=village_outliers['x_jittered'],
                        y=village_outliers['y_jittered'],
                        mode='markers',
                        name=f"{village} (Outliers)",
                        marker=dict(
                            color=color,
                            size=sizes,
                            opacity=0.4,
                            line=dict(width=1, color='black')
                        ),
                        customdata=np.stack((
                            village_outliers[x_col],
                            village_outliers[y_col],
                            village_outliers['point_freq'],
                            village_outliers['loc-res-sub'],
                            village_outliers['_uuid']
                        ), axis=1),
                        hovertemplate=(
                            f"<b>Village:</b> {village} (Outlier)<br>"
                            f"<b>{x_col}:</b> %{{customdata[0]:.2f}}<br>"
                            f"<b>{y_col}:</b> %{{customdata[1]:.2f}}<br>"
                            f"<b>Number of overlapping points:</b> %{{customdata[2]}}<br>"
                            f"<b>Location-Respondent:</b> %{{customdata[3]}}<br>"
                            f"<b>UUID:</b> %{{customdata[4]}}<br>"
                            "<extra></extra>"
                        )
                    ))
            
            # Add trend line if there are enough points
            if len(df_normal) > 2:
                z = np.polyfit(df_normal[x_col], df_normal[y_col], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=df_normal[x_col],
                    y=p(df_normal[x_col]),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True
                ))
        else:
            # No outlier detection, show all points
            df_clean.loc[:, 'point_freq'] = df_clean.groupby([x_col, y_col])[x_col].transform('count')
            
            # Add jitter if enabled
            if show_jitter:
                x_range = df_clean[x_col].max() - df_clean[x_col].min()
                y_range = df_clean[y_col].max() - df_clean[y_col].min()
                x_jitter = np.random.normal(0, x_range * jitter_amount, len(df_clean))
                y_jitter = np.random.normal(0, y_range * jitter_amount, len(df_clean))
                df_clean.loc[:, 'x_jittered'] = df_clean[x_col] + x_jitter
                df_clean.loc[:, 'y_jittered'] = df_clean[y_col] + y_jitter
            else:
                df_clean.loc[:, 'x_jittered'] = df_clean[x_col]
                df_clean.loc[:, 'y_jittered'] = df_clean[y_col]
            
            # Get unique villages for coloring
            villages = df_clean['গ্রামের নাম'].unique()
            colors = px.colors.qualitative.Set3
            
            # Add points for each village
            for i, village in enumerate(villages):
                village_data = df_clean[df_clean['গ্রামের নাম'] == village]
                if len(village_data) > 0:
                    color = colors[i % len(colors)]
                    if size_by_frequency:
                        sizes = 5 + (village_data['point_freq'] * 2)
                        sizes = np.clip(sizes, 5, 20)
                    else:
                        sizes = 8
                    
                    fig.add_trace(go.Scatter(
                        x=village_data['x_jittered'],
                        y=village_data['y_jittered'],
                        mode='markers',
                        name=village,
                        marker=dict(
                            color=color,
                            size=sizes,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        customdata=np.stack((
                            village_data[x_col],
                            village_data[y_col],
                            village_data['point_freq'],
                            village_data['loc-res-sub'],
                            village_data['_uuid']
                        ), axis=1),
                        hovertemplate=(
                            f"<b>Village:</b> {village}<br>"
                            f"<b>{x_col}:</b> %{{customdata[0]:.2f}}<br>"
                            f"<b>{y_col}:</b> %{{customdata[1]:.2f}}<br>"
                            f"<b>Number of overlapping points:</b> %{{customdata[2]}}<br>"
                            f"<b>Location-Respondent:</b> %{{customdata[3]}}<br>"
                            f"<b>UUID:</b> %{{customdata[4]}}<br>"
                            "<extra></extra>"
                        )
                    ))
            
            # Add trend line if there are enough points
            if len(df_clean) > 2:
                z = np.polyfit(df_clean[x_col], df_clean[y_col], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=df_clean[x_col],
                    y=p(df_clean[x_col]),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True
                ))
        
        # Update layout
        fig.update_layout(
            title=f'{y_col} vs {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=600,
            showlegend=True,
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Map
with tab3:
    st.header("Geographic Analysis")
    
    # Define columns to exclude from mapping
    exclude_cols = [lat_col, lon_col, '_id', '_uuid', '_submission_time', '_validation_status',
                   '_notes', '_status', '_submitted_by', '__version__', '_tags', '_index', 'start', 'end']
    
    plot_columns = [col for col in current_df.columns if col not in exclude_cols]
    
    selected_column = st.selectbox("Color by:", plot_columns)
    
    if selected_column:
        # Use cached data with filtered dataframe
        df_cleaned = create_map_data(current_df, lat_col, lon_col, selected_column)
        st.write(f"Number of points with valid coordinates: {len(df_cleaned)}")
        
        # Create base map
        m = folium.Map(
            location=[df_cleaned[lat_col].mean(),
                     df_cleaned[lon_col].mean()],
            zoom_start=12,
            prefer_canvas=True
        )
        
        # Add GeoJSON layer
        style_function = lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0
        }
        
        tooltip = folium.GeoJsonTooltip(
            fields=['tv_name'],
            aliases=['Village:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
        
        geojson_layer = folium.GeoJson(
            geojson_data,
            name='Village Boundaries',
            style_function=style_function,
            tooltip=tooltip
        )
        geojson_layer.add_to(m)
        
        # Create color map based on data type
        if pd.api.types.is_numeric_dtype(df_cleaned[selected_column]):
            # For numeric data
            Q1 = df_cleaned[selected_column].quantile(0.25)
            Q3 = df_cleaned[selected_column].quantile(0.75)
            IQR = Q3 - Q1
            bounds = [
                df_cleaned[selected_column].min(),
                Q1 - 1.5 * IQR,
                Q1,
                df_cleaned[selected_column].median(),
                Q3,
                Q3 + 1.5 * IQR,
                df_cleaned[selected_column].max()
            ]
            bounds = sorted(list(set([round(b, 2) for b in bounds])))
            
            color_map = cm.StepColormap(
                colors=['#313695', '#4575b4', '#74add1', '#abd9e9', '#fdae61', '#f46d43', '#d73027'],
                vmin=bounds[0],
                vmax=bounds[-1],
                index=bounds
            )
            
            # Add points to map
            for idx, row in df_cleaned.iterrows():
                color = color_map(row[selected_column])
                popup_text = f"Location-Respondent-Submitter: {row['loc-res-sub']}<br>UUID: {row['_uuid']}<br>{selected_column}: {row[selected_column]}"
                
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=popup_text,
                    color=color,
                    fill=True
                ).add_to(m)
            
            # Add color map to map
            color_map.add_to(m)
            
        else:
            # For categorical data
            unique_values = df_cleaned[selected_column].value_counts()
            top_categories = unique_values.head(20)
            
            distinct_colors = [
                '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
                '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
                '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000'
            ]
            
            color_dict = dict(zip(top_categories.index, distinct_colors[:len(top_categories)]))
            if len(unique_values) > 20:
                color_dict['Others'] = '#808080'
            
            # Add points to map
            for idx, row in df_cleaned.iterrows():
                color = color_dict.get(row[selected_column], '#808080')
                popup_text = f"Location-Respondent-Submitter: {row['loc-res-sub']}<br>UUID: {row['_uuid']}<br>{selected_column}: {row[selected_column]}"
                
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=popup_text,
                    color=color,
                    fill=True
                ).add_to(m)
        
        # Add layer control
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.LayerControl().add_to(m)
        
        # Display the map
        st_folium(m, width=1200, height=600)

# Tab 4: Table View
with tab4:
    st.header("Data Table View")
    
    # Set pandas styling limit
    pd.set_option("styler.render.max_elements", 1000000)  # Increase the limit
    
    # Get numeric columns for outlier detection
    numeric_cols = [col for col in current_df.select_dtypes(include=['int64', 'float64']).columns 
                   if col not in exclude_cols]  # Use the same exclude_cols as other tabs
    
    # Sort numeric columns to put computed columns first
    computed_numeric = [col for col in computed_columns if col in numeric_cols]
    other_numeric = [col for col in numeric_cols if col not in computed_columns]
    numeric_cols = computed_numeric + sorted(other_numeric)
    
    # Add table view options
    with st.expander("Table View Options", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Column selection
            all_columns = current_df.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to display:",
                all_columns,
                default=all_columns[:10],  # Default to first 10 columns
                help="Choose which columns to display in the table"
            )
            
            # Page size selection
            page_size = st.selectbox(
                "Rows per page:",
                options=[10, 25, 50, 100, 250, 500],
                index=1,  # Default to 25
                help="Number of rows to display per page"
            )
        
        with col2:
            # Outlier detection options
            selected_numeric_cols = st.multiselect(
                "Select columns for outlier detection:",
                numeric_cols,
                default=computed_numeric if computed_numeric else numeric_cols[:2],
                help="Choose which numeric columns to check for outliers"
            )
            
            if selected_numeric_cols:
                outlier_threshold = st.slider(
                    "Z-score threshold for outlier detection",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                    help="Points with absolute Z-score above this threshold will be highlighted as outliers"
                )
    
    # Create a copy of the dataframe for display
    display_df = current_df[selected_columns].copy() if selected_columns else current_df.copy()
    
    # Add outlier column if columns are selected
    if selected_numeric_cols:
        # Initialize outlier column
        display_df['is_outlier'] = False
        
        # Detect outliers for each selected column
        for col in selected_numeric_cols:
            if col in display_df.columns:  # Only process if column is selected for display
                outlier_mask = detect_outliers_zscore(display_df[col], threshold=outlier_threshold)
                display_df['is_outlier'] |= outlier_mask
        
        # Add outlier count to the display
        outlier_count = display_df['is_outlier'].sum()
        total_count = len(display_df)
        st.info(f"Found {outlier_count} rows ({outlier_count/total_count*100:.1f}%) with outliers in selected columns")
    
    # Calculate total pages
    total_rows = len(display_df)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Add pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Page navigation
        if total_pages > 1:
            current_page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                help=f"Navigate through {total_pages} pages"
            )
        else:
            current_page = 1
    
    with col2:
        st.write(f"Showing rows {(current_page-1)*page_size + 1} to {min(current_page*page_size, total_rows)} of {total_rows}")
    
    with col3:
        # Jump to page
        if total_pages > 1:
            jump_to = st.number_input(
                "Jump to page",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                step=1,
                help="Quickly jump to a specific page"
            )
            if jump_to != current_page:
                current_page = jump_to
    
    # Get the current page of data
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    current_page_data = display_df.iloc[start_idx:end_idx]
    
    # Apply styling if outlier detection is active
    if selected_numeric_cols and 'is_outlier' in display_df.columns:
        def highlight_outliers(row):
            if row['is_outlier']:
                return ['background-color: #ffebee'] * len(row)  # Light red background
            return [''] * len(row)
        
        # Apply styling only to the current page
        styled_df = current_page_data.style.apply(highlight_outliers, axis=1)
        
        # Display the styled dataframe
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=600
        )
    else:
        # Display regular dataframe
        st.dataframe(
            current_page_data,
            use_container_width=True,
            height=600
        )
    
    # Add download button for the data
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Full Dataset",
        csv,
        f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
        key='download-csv'
    )
    
    # Add download button for current page
    if total_pages > 1:
        current_page_csv = current_page_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Current Page",
            current_page_csv,
            f"data_page_{current_page}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key='download-current-page-csv'
        )

# Tab 5: Hexbin Map
with tab5:
    st.header("Village Hexbin Map")
    
    # Get available villages
    available_villages = hexbin_map.get_available_villages(current_df)
    
    if not available_villages:
        st.error("No matching villages found between the dataset and GeoJSON data")
    else:
        # Village selection
        selected_village = st.selectbox(
            "Select Village",
            options=available_villages,
            help="Select a village to view its hexbin map"
        )
        
        # Indicator selection
        selected_indicator = st.selectbox(
            "Select Indicator",
            options=list(hexbin_map.indicators.keys()),
            format_func=lambda x: hexbin_map.indicators[x],
            help="Select the water usage indicator to display"
        )
        
        try:
            # Create and display map
            map_object = hexbin_map.create_hexbin_map(
                current_df, 
                selected_village, 
                selected_indicator
            )
            st_folium(map_object, width=1200, height=600)
            
            # Add some statistics about the selected village
            village_data = current_df[current_df['গ্রামের নাম'] == selected_village]
            st.write(f"Total points in {selected_village}: {len(village_data)}")
            
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")

# HTML template for the report
REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Village Water Usage Report</title>
    <style>
        :root {
            --primary-color: #2E86C1;
            --secondary-color: #27AE60;
            --accent-color: #E74C3C;
            --text-color: #2C3E50;
            --light-bg: #F8F9FA;
            --border-color: #DEE2E6;
        }
        
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: var(--text-color);
            background-color: white;
        }
        
        .village-section {
            page-break-after: always;
            margin-bottom: 30px;
        }
        
        .village-header {
            page-break-after: always;
        }
        
        .para-section {
            page-break-before: always;
            page-break-after: always;
            margin-top: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 10px;
            font-size: 28px;
            margin-bottom: 25px;
        }
        
        h2 {
            color: var(--text-color);
            margin-top: 25px;
            font-size: 22px;
            border-left: 4px solid var(--primary-color);
            padding-left: 15px;
        }
        
        h3 {
            color: var(--text-color);
            font-size: 18px;
            margin-top: 20px;
            border-left: 3px solid var(--secondary-color);
            padding-left: 12px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
            border: 1px solid var(--border-color);
        }
        
        th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
        }
        
        tr:nth-child(even) {
            background-color: var(--light-bg);
        }
        
        tr:hover {
            background-color: #E8F4F8;
        }
        
        .chart-block {
            page-break-inside: avoid;
            margin: 30px 0;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chart-block h2 {
            margin-top: 0;
            margin-bottom: 20px;
            border-left: none;
            padding-left: 0;
            text-align: center;
        }
        
        .chart-block img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            display: block;
            margin: 0 auto;
        }
        
        .summary-box {
            background-color: var(--light-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .summary-box h2 {
            margin-top: 0;
            color: var(--primary-color);
            border-left: none;
            padding-left: 0;
        }
        
        @media print {
            body {
                margin: 20px;
            }
            
            .village-section {
                page-break-after: always;
            }
            
            .village-header {
                page-break-after: always;
            }
            
            .para-section {
                page-break-before: always;
                page-break-after: always;
            }
            
            .chart-block {
                page-break-inside: avoid;
                box-shadow: none;
                border: 1px solid #ccc;
            }
            
            table {
                page-break-inside: avoid;
            }
            
            .summary-box {
                box-shadow: none;
                border: 1px solid #ccc;
            }
        }
    </style>
</head>
<body>
    {% for village in villages %}
    <div class="village-section">
        <div class="village-header">
            <h1>Village Profile: {{ village.name }}</h1>
            
            <div class="summary-box">
                <h2>Summary Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {% for stat in village.stats %}
                    <tr>
                        <td>{{ stat.label }}</td>
                        <td>{{ stat.value }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            {% for water_type in ['Surface', 'Ground'] %}
            <div class="chart-block">
                <h2>{{ water_type }} Water Usage</h2>
                <img src="data:image/png;base64,{{ village.charts[water_type] }}" alt="{{ water_type }} Water Usage Chart">
            </div>
            {% endfor %}

            <div class="para-section">
                <h2>List of Paras</h2>
                <table>
                    <tr>
                        <th>Para Name</th>
                        <th>Number of Households</th>
                    </tr>
                    {% for para in village.paras %}
                    <tr>
                        <td>{{ para.name }}</td>
                        <td>{{ para.count }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>

        {% for para in village.paras %}
        <div class="para-section">
            <h1>{{ para.name }}</h1>
            <h2>Village: {{ village.name }}</h2>
            
            <div class="summary-box">
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {% for stat in para.stats %}
                    <tr>
                        <td>{{ stat.label }}</td>
                        <td>{{ stat.value }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            {% for water_type in ['Surface', 'Ground'] %}
            <div class="chart-block">
                <h2>{{ water_type }} Water Usage</h2>
                <img src="data:image/png;base64,{{ para.charts[water_type] }}" alt="{{ water_type }} Water Usage Chart for {{ para.name }}">
            </div>
            {% endfor %}
        </div>
        {% endfor %}
    </div>
    {% endfor %}
</body>
</html>
"""

def create_water_usage_chart(df, water_type, village_name, para_name=None):
    """Create a bar chart for water usage by para"""
    if para_name:
        title = f"{water_type} Water Usage in {para_name}"
        data = df[df['পাড়ার নাম'] == para_name]
    else:
        title = f"{water_type} Water Usage by Para"
        data = df
    
    # Convert water_type to lowercase for column name
    water_col = f"{water_type.lower()}_water_usage"
    
    # Remove outliers
    outlier_mask = detect_outliers_zscore(data[water_col])
    data_clean = data[~outlier_mask]
    
    # Group by para and calculate statistics
    para_stats = data_clean.groupby('পাড়ার নাম')[water_col].agg(['mean', 'median', 'count'])
    para_stats = para_stats[para_stats['count'] >= 10]  # Filter paras with <10 datapoints
    
    if len(para_stats) == 0:
        return None
    
    # Create bar chart with improved styling
    fig = go.Figure()
    
    # Define a professional color palette
    colors = {
        'mean': '#2E86C1',  # Professional blue
        'median': '#27AE60'  # Professional green
    }
    
    # Add mean bars
    fig.add_trace(go.Bar(
        name='Mean',
        x=para_stats.index,
        y=para_stats['mean'],
        text=para_stats['mean'].round(2),
        textposition='auto',
        marker_color=colors['mean'],
        marker=dict(
            line=dict(width=1, color='white')
        )
    ))
    
    # Add median bars
    fig.add_trace(go.Bar(
        name='Median',
        x=para_stats.index,
        y=para_stats['median'],
        text=para_stats['median'].round(2),
        textposition='auto',
        marker_color=colors['median'],
        marker=dict(
            line=dict(width=1, color='white')
        )
    ))
    
    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='#2C3E50'),
            x=0.5,
            y=0.95
        ),
        xaxis_title='Para',
        yaxis_title=f'{water_type} Water Usage',
        barmode='group',
        showlegend=True,
        height=400,
        width=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12, color="#2C3E50"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(t=80, b=40, l=60, r=40)
    )
    
    # Update axes styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E5E5'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E5E5'
    )
    
    # Convert to base64 for embedding in HTML
    img_bytes = fig.to_image(format="png")
    return base64.b64encode(img_bytes).decode('utf-8')

def generate_village_data(df, village_name):
    """Generate data for a village's HTML report section"""
    village_data = df[df['গ্রামের নাম'] == village_name].copy()
    
    # Remove outliers
    surface_outliers = detect_outliers_zscore(village_data['surface_water_usage'])
    ground_outliers = detect_outliers_zscore(village_data['ground_water_usage'])
    village_data_clean = village_data[~(surface_outliers | ground_outliers)]
    
    # Calculate village statistics
    para_counts = village_data_clean.groupby('পাড়ার নাম').size()
    valid_paras = para_counts[para_counts >= 10]
    
    # Sort paras by household count in descending order
    valid_paras = valid_paras.sort_values(ascending=False)
    
    # Calculate mean and median for village
    surface_mean = village_data_clean['surface_water_usage'].mean()
    surface_median = village_data_clean['surface_water_usage'].median()
    ground_mean = village_data_clean['ground_water_usage'].mean()
    ground_median = village_data_clean['ground_water_usage'].median()
    
    # Create village statistics
    stats = [
        {'label': 'Total Households Surveyed', 'value': str(len(village_data_clean))},
        {'label': 'Number of Paras Surveyed', 'value': str(len(valid_paras))},
        {'label': 'Surface Water Usage', 'value': f"mean/median: {surface_mean:.2f}/{surface_median:.2f}"},
        {'label': 'Ground Water Usage', 'value': f"mean/median: {ground_mean:.2f}/{ground_median:.2f}"}
    ]
    
    # Generate charts
    charts = {
        'Surface': create_water_usage_chart(village_data, 'surface', village_name),
        'Ground': create_water_usage_chart(village_data, 'ground', village_name)
    }
    
    # Generate para data
    paras = []
    for para_name in valid_paras.index:
        para_data = village_data[village_data['পাড়ার নাম'] == para_name].copy()
        para_outliers = detect_outliers_zscore(para_data['surface_water_usage']) | detect_outliers_zscore(para_data['ground_water_usage'])
        para_data_clean = para_data[~para_outliers]
        
        # Filter out zero usage data for para statistics
        para_data_with_usage = para_data_clean[~((para_data_clean['surface_water_usage'] == 0) & (para_data_clean['ground_water_usage'] == 0))]
        
        # Calculate mean and median for para
        surface_mean = para_data_with_usage['surface_water_usage'].mean()
        surface_median = para_data_with_usage['surface_water_usage'].median()
        ground_mean = para_data_with_usage['ground_water_usage'].mean()
        ground_median = para_data_with_usage['ground_water_usage'].median()
        
        para_stats = [
            {'label': 'Households Surveyed', 'value': str(len(para_data_clean))},
            {'label': 'Outliers Removed', 'value': f"{len(para_data) - len(para_data_clean)} ({((len(para_data) - len(para_data_clean)) / len(para_data) * 100):.1f}%)"},
            {'label': 'Surface Water Usage', 'value': f"mean/median: {surface_mean:.2f}/{surface_median:.2f}"},
            {'label': 'Surface Water Usage (Std Dev)', 'value': f"{para_data_with_usage['surface_water_usage'].std():.2f}"},
            {'label': 'Ground Water Usage', 'value': f"mean/median: {ground_mean:.2f}/{ground_median:.2f}"},
            {'label': 'Ground Water Usage (Std Dev)', 'value': f"{para_data_with_usage['ground_water_usage'].std():.2f}"}
        ]
        
        para_charts = {
            'Surface': create_water_usage_chart(df, 'surface', village_name, para_name),
            'Ground': create_water_usage_chart(df, 'ground', village_name, para_name)
        }
        
        paras.append({
            'name': para_name,
            'count': str(valid_paras[para_name]),
            'stats': para_stats,
            'charts': para_charts
        })
    
    return {
        'name': village_name,
        'stats': stats,
        'charts': charts,
        'paras': paras
    }

def generate_html_report(df):
    """Generate the complete HTML report"""
    # Filter out rows where both surface and ground water usage are 0
    df = df[~((df['surface_water_usage'] == 0) & (df['ground_water_usage'] == 0))].copy()
    
    # Get unique villages
    villages = df['গ্রামের নাম'].unique()
    
    # Generate data for each village
    village_data = [generate_village_data(df, village) for village in villages]
    
    # Create HTML using template
    template = jinja2.Template(REPORT_TEMPLATE)
    html_content = template.render(villages=village_data)
    
    return html_content

def save_and_open_html(html_content):
    """Save HTML content to a temporary file and open it in the default browser"""
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
        f.write(html_content)
        temp_path = f.name
    
    # Open the HTML file in the default browser
    webbrowser.open('file://' + temp_path)
    return temp_path

# Add report generation button to the sidebar
with st.sidebar:
    st.header("Report Generation")
    if st.button("Generate Village Profiles Report"):
        # Use the filtered dataframe if available
        report_df = st.session_state.filtered_df if st.session_state.filtered_df is not None else df
        
        with st.spinner("Generating HTML report..."):
            try:
                html_content = generate_html_report(report_df)
                temp_file = save_and_open_html(html_content)
                
                st.success("Report generated successfully! Opening in your browser...")
                st.info("To save as PDF: Use your browser's Print function (Ctrl+P or Cmd+P) and select 'Save as PDF'")
                
                # Add a download button for the HTML file
                with open(temp_file, 'rb') as f:
                    st.download_button(
                        label="Download HTML Report",
                        data=f,
                        file_name=f"village_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
            except Exception as e:
                st.error(f"Error generating report: {str(e)}") 