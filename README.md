Mortality Data Analysis Tool
An interactive dashboard analyzing US mortality data. Features monthly death trends, state-level comparisons, and forecasting with confidence intervals. Built with Streamlit, Plotly, and statsmodels.

📋 Overview
Death Relevant is a comprehensive tool for analyzing and visualizing US mortality trends. It provides an interactive interface to explore monthly death patterns, compare mortality rates across years and states, and forecast future trends with statistical confidence intervals.

✨ Features
Monthly Trends: Visualize mortality data with line, bar, or area charts
Year-to-Year Comparison: Compare monthly deaths between any two years
Monthly Details: Examine monthly patterns for specific years
State Analysis: Analyze mortality trends across different states (1985-2004)
Mortality Forecasting: Project future death rates with configurable confidence intervals
State-Level Forecasts: Generate state-specific projections (focus on NY and NJ)
🚀 Installation
📦 Dependencies
streamlit
pandas
numpy
plotly
matplotlib
seaborn
statsmodels
scikit-learn
📊 Usage
Ensure your mortality data CSV files are placed in the analysis_results directory
Run the Streamlit application:
For the New York specific dashboard:

📁 Project Structure
mortality_streamlit.py: Main Streamlit dashboard application
ny_streamlit.py: New York specific dashboard
new_york_analysis.py: Analysis functions for New York data
analysis_results: Directory containing processed mortality data files
monthly_deaths.csv: National monthly mortality data
state_deaths_1985_2004.csv: State-level mortality data
📄 Data Requirements
The application expects the following data files:

monthly_deaths.csv: National monthly mortality data
state_deaths_1985_2004.csv: State-level mortality data (optional)
Data Format
National data should include columns for year, month, and death count. State data should include columns for year, month, state (FIPS code or abbreviation), and death count.

🧪 Features in Detail
Monthly Trends
- Interactive time series visualizations
- Customizable date ranges
- Multiple chart types (line, bar, area)
  
Summary statistics
- Year-to-Year Comparison
- Side-by-side monthly comparisons
- Percentage difference analysis
- Detailed comparison tables
  
State Analysis
- Top 10 states by mortality rate
- Individual state detailed analysis
- Monthly patterns for selected states
  
Mortality Forecasting
- Time series forecasting using SARIMAX models
- Configurable forecast horizon (1-10 years)
- Adjustable confidence intervals
- State-level projections based on historical proportions
- Downloadable forecast data
