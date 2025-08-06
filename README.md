# OAK India WASH Data Analysis

A comprehensive Streamlit web application for analyzing Water, Sanitation, and Hygiene (WASH) data from OAK India. This application provides interactive data visualization, mapping, and statistical analysis capabilities for WASH-related datasets.

## 🌊 Features

### 📊 Data Analysis & Visualization
- **Interactive Dashboards**: Multi-page Streamlit application with comprehensive data analysis tools
- **Statistical Analysis**: Outlier detection using Z-score and IQR methods
- **Data Clustering**: DBSCAN clustering for spatial data analysis
- **Chart Generation**: Multiple chart types including bar charts, scatter plots, and custom water usage visualizations

### 🗺️ Geographic Visualization
- **Interactive Maps**: Folium-based maps with marker clustering
- **Hexbin Maps**: Custom hexbin visualization for spatial data density
- **GPS Data Processing**: Support for Bengali GPS column mapping
- **Village-level Analysis**: Geographic analysis at village and para (neighborhood) levels

### 📈 Advanced Analytics
- **Outlier Detection**: Automated detection and highlighting of statistical outliers
- **Data Quality Assessment**: String similarity matching for column mapping
- **Report Generation**: HTML report generation with embedded charts
- **Export Capabilities**: PDF and image export functionality

### 🌐 Multi-language Support
- **Bengali Language Support**: Native support for Bengali column names and data
- **Font Integration**: NotoSansBengali font for proper Bengali text rendering

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Conda (recommended) or pip

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd oak-india-streamlit

# Create and activate conda environment
conda env create -f environment.yml
conda activate oak-wash

# Run the application
streamlit run app.py
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone <repository-url>
cd oak-india-streamlit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage
1. Open your web browser and navigate to `http://localhost:8501`
2. Upload your WASH data CSV file or use the provided sample data
3. Explore the various analysis tools and visualizations
4. Generate reports and export results as needed

## 📁 Project Structure

```
oak-india-streamlit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies for pip
├── environment.yml        # Conda environment configuration
├── README.md             # This file
└── src/                  # Source code modules (if applicable)
    └── visualization.py   # Custom visualization modules
```

