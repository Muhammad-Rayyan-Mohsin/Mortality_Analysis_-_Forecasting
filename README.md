# Death Relevant

## Overview
Death Relevant is an interactive dashboard built with Streamlit for analyzing and visualizing U.S. mortality data. It includes monthly, state-level, and forecasted death trends, leveraging CDC data and time series models.

## Features
- Monthly mortality trends with year-over-year comparisons  
- State-level analysis for historical data (1985–2004)  
- SARIMAX-based forecasting model with confidence intervals  
- Interactive charts using Plotly and extensive filtering options  

## Installation
1. Clone this repository or download the source code.  
2. Navigate to the project directory.  
3. Install the required Python packages:  
   ```
   pip install -r requirements.txt
   ```
4. Ensure the necessary CSV files are placed in the `analysis_results` folder.

## Usage
1. Open a terminal in the project directory.  
2. Run the Streamlit app:  
   ```
   streamlit run mortality_streamlit.py
   ```
3. Access the dashboard locally at the URL displayed in the terminal (usually http://localhost:8501).

## Project Structure
```
Death Relevant/
  ├─ analysis_results/            # Analysis outputs and CSV data
  ├─ mortality_streamlit.py       # Main Streamlit dashboard
  ├─ new_york_analysis.py         # (Example) Additional analysis module
  ├─ requirements.txt             # Python dependencies
  └─ README.md                    # Project documentation (this file)
```

## Data Sources
- CDC’s National Center for Health Statistics for mortality counts  
- State-level historical data files for detailed local comparison  

## Methodology
- Data processing includes converting 2-digit years, sorting, and cleaning  
- SARIMAX modeling applied to monthly aggregated counts  
- State forecasts derived by scaling the national forecast based on historical share  

## Contributing
Contributions or suggestions are welcome. Fork this repository and submit pull requests with any enhancements or fixes.

## License
Please review the repository for license details or contact the author for clarification.
