import folium
import pandas as pd
import numpy as np
from folium.plugins import HeatMap
import geopandas as gpd
from shapely.geometry import Point, Polygon
import branca.colormap as cm
import h3
from typing import Dict, List, Tuple, Optional
import json
import requests
from io import BytesIO
import streamlit as st



class HexbinMapGenerator:
    def __init__(self, geojson_path: str):
        """
        Initialize the HexbinMapGenerator with GeoJSON data.
        
        Args:
            geojson_path: URL or path to the GeoJSON file containing village boundaries
        """
        self.geojson_path = geojson_path
        
        
        self.village_boundaries = self._process_geojson()
        self.h3_resolution = self._calculate_h3_resolution()
        
        # Define indicators and their display names
        self.indicators = {
            'total_water_usage': 'Total Water Usage',
            'surface_water_usage': 'Surface Water Usage',
            'ground_water_usage': 'Ground Water Usage'
        }
        
        # Define color schemes for each indicator
        self.color_schemes = {
            'total_water_usage': ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
            'surface_water_usage': ['#e5f5e0', '#a1d99b', '#31a354'],
            'ground_water_usage': ['#eff3ff', '#bdd7e7', '#6baed6', '#2171b5']
        }
        
        # Column names will be set when create_hexbin_map is called
        self.lat_col = None
        self.lon_col = None

    def _load_geojson(self, geojson_path: str) -> gpd.GeoDataFrame:
        """Load and validate GeoJSON data"""
        try:
            if geojson_path.startswith(('http://', 'https://')):
                response = requests.get(geojson_path)
                response.raise_for_status()
                return gpd.GeoDataFrame.from_features(json.loads(response.text))
            else:
                return gpd.read_file(geojson_path)
        except Exception as e:
            raise Exception(f"Error loading GeoJSON: {str(e)}")

    def _process_geojson(self) -> Dict[str, Polygon]:
        """Process GeoJSON into a dictionary of village boundaries"""
        return {row['tv_name']: row.geometry 
                for _, row in self._load_geojson(self.geojson_path).iterrows()}

    def _calculate_h3_resolution(self) -> int:
        """Calculate appropriate H3 resolution for given hex size"""
        # H3 resolution 9 is approximately 100 square meters
        return 10

    def _validate_village(self, village_name: str) -> Polygon:
        """Check if village exists in GeoJSON"""
        if village_name not in self.village_boundaries:
            raise ValueError(f"Village '{village_name}' not found in GeoJSON data")
        return self.village_boundaries[village_name]

    def _filter_points_to_village(self, df: pd.DataFrame, village_name: str) -> pd.DataFrame:
        """
        Filter points to only include those within the specified village boundary.
        """
       
        
        # Check if required columns exist
        if self.lat_col not in df.columns:
            st.error(f"Latitude column '{self.lat_col}' not found. Available columns: {df.columns.tolist()}")
            # Try to find similar column names
            lat_cols = [col for col in df.columns if 'latitude' in col.lower()]
            
            return pd.DataFrame()
            
        if self.lon_col not in df.columns:
            st.error(f"Longitude column '{self.lon_col}' not found. Available columns: {df.columns.tolist()}")
            # Try to find similar column names
            lon_cols = [col for col in df.columns if 'longitude' in col.lower()]
            
            return pd.DataFrame()
        
        # Get village boundary
        try:
            village_boundary = self._validate_village(village_name)
        except Exception as e:
            st.error(f"Error getting village boundary: {str(e)}")
            return pd.DataFrame()
        
      
        
        # Filter points to village boundary
        try:
            mask = df.apply(lambda row: village_boundary.contains(Point(row[self.lon_col], row[self.lat_col])), axis=1)
            filtered_df = df[mask].copy()
            
            if len(filtered_df) == 0:
                st.error(f"No points found within {village_name} boundary")
                return pd.DataFrame()
            
            return filtered_df
        except Exception as e:
            return pd.DataFrame()

    def _create_hexbins(self, df: pd.DataFrame, indicator: str) -> pd.DataFrame:
        """Create hexbins from the filtered points"""
       
        try:
            # Convert points to h3 hexagons
            df['h3'] = df.apply(
                lambda row: h3.latlng_to_cell(
                    row[self.lat_col],
                    row[self.lon_col],
                    self.h3_resolution
                ),
                axis=1
            )
            
            # Group by hexagon and calculate statistics
            hexbin_stats = df.groupby('h3').agg({
                indicator: ['mean', 'median', 'count']
            }).reset_index()
            
            # Flatten column names
            hexbin_stats.columns = ['h3', f'{indicator}_mean', f'{indicator}_median', 'point_count']
            
            # Get hexagon boundaries and store as coordinates for display
            hexbin_stats['boundary_coords'] = hexbin_stats['h3'].apply(
                lambda h: h3.cell_to_boundary(h)
            )
            
            # Create geometry column for internal use only
            hexbin_stats['_geometry'] = hexbin_stats['boundary_coords'].apply(Polygon)
            
            return hexbin_stats
            
        except Exception as e:
            st.error(f"DEBUG: Error in _create_hexbins: {str(e)}")
            raise

    def _create_color_scale(self, indicator: str, min_val: float, max_val: float) -> cm.LinearColormap:
        """Create a color scale for the selected indicator"""
        return cm.LinearColormap(
            colors=self.color_schemes[indicator],
            vmin=min_val,
            vmax=max_val,
            caption=f"{self.indicators[indicator]} Value"
        )

    def _create_popup_html(self, row: pd.Series, indicator: str) -> str:
        """Create HTML content for popup"""
        return f"""
        <div style='font-family: Arial; font-size: 12px;'>
            <b>Mean {self.indicators[indicator]}:</b> {row[f'{indicator}_mean']:.2f}<br>
            <b>Median {self.indicators[indicator]}:</b> {row[f'{indicator}_median']:.2f}<br>
            <b>Number of Points:</b> {row['point_count']}<br>
        </div>
        """

    def _set_column_names(self, df: pd.DataFrame):
        """Set the column names based on available columns in the dataframe"""
        # Find latitude and longitude columns with more detailed debugging
        lat_cols = [col for col in df.columns if 'latitude' in col.lower() and '_বাড়ির' in col]
        lon_cols = [col for col in df.columns if 'longitude' in col.lower() and '_বাড়ির' in col]
        # Try to find columns using the GPS_COLUMN_MAPPING from app.py
        expected_lat_col = '_বাড়ির অবস্থান _latitude'
        expected_lon_col = '_বাড়ির অবস্থান _longitude'
        if lat_cols and lon_cols:
            self.lat_col = lat_cols[0]
            self.lon_col = lon_cols[0]
        elif expected_lat_col in df.columns and expected_lon_col in df.columns:
            # Fallback to expected column names from GPS_COLUMN_MAPPING
            self.lat_col = expected_lat_col
            self.lon_col = expected_lon_col
        else:
            st.error("Could not find required coordinate columns in the data")
    def create_hexbin_map(self, df: pd.DataFrame, village_name: str, indicator: str = 'total_water_usage') -> folium.Map:
        """Create a hexbin map for the specified village and indicator."""
        # Set column names based on available columns
        self._set_column_names(df)
        if not self.lat_col or not self.lon_col:
            st.error("Required coordinate columns not found")
            return None
        # Filter points to village
        try:
            village_df = self._filter_points_to_village(df, village_name)
        except Exception as e:
            st.error(f"Error filtering points to village: {str(e)}")
            return None
        # Create hexbins
        try:
            hexbin_df = self._create_hexbins(village_df, indicator)
        except Exception as e:
            st.error(f"Error creating hexbins: {str(e)}")
            return None
        # Get village boundary
        try:
            village_boundary = self._validate_village(village_name)
        except Exception as e:
            st.error(f"Error getting village boundary: {str(e)}")
            return None
        # Create base map
        try:
            # Calculate map center from the village data
            center_lat = village_df[self.lat_col].mean()
            center_lon = village_df[self.lon_col].mean()
            
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=14,
                tiles='CartoDB positron'
            )
           
        except Exception as e:
            st.error(f"Error creating base map: {str(e)}")
            return None
            

        
        try:
            # Create color scale using mean values
            color_scale = self._create_color_scale(
                indicator,
                hexbin_df[f'{indicator}_mean'].min(),
                hexbin_df[f'{indicator}_mean'].max()
            )
            
            # Add hexbins to map
            for _, row in hexbin_df.iterrows():
                hexbin = row['boundary_coords']
                # Use mean value for coloring
                color = color_scale(row[f'{indicator}_mean'])
                popup_html = self._create_popup_html(row, indicator)
                
                folium.Polygon(
                    locations=hexbin,
                    color='black',
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(m)
            
            # Add village boundary
            folium.GeoJson(
                village_boundary,
                style_function=lambda x: {
                    'fillColor': 'none',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0
                }
            ).add_to(m)
            
            # Add color scale to map
            color_scale.add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            return None

    def get_available_villages(self, df: pd.DataFrame) -> List[str]:
        """Get list of villages that have data in both the dataset and GeoJSON"""
        # Get villages from dataset
        dataset_villages = set(df['গ্রামের নাম'].unique())
        
        # Get villages from GeoJSON
        geojson_villages = set(self.village_boundaries.keys())
        
        # Find intersection
        available_villages = sorted(list(dataset_villages.intersection(geojson_villages)))
        
        return available_villages 