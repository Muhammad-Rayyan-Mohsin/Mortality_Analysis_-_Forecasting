import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

# State FIPS code mapping - for decoding state numbers to abbreviations
STATE_FIPS_MAPPING = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT',
    10: 'DE', 11: 'DC', 12: 'FL', 13: 'GA', 14: 'GU', 15: 'HI', 16: 'ID',
    17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA',
    23: 'ME', 24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS',
    29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH', 34: 'NJ',
    35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK',
    41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN',
    48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV',
    55: 'WI', 56: 'WY', 72: 'PR'
}

# Full state names for better display
STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia', 'PR': 'Puerto Rico'
}

def convert_to_four_digit_year(year):
    """
    Convert 2-digit years to 4-digit years.
    
    Args:
        year: A year value that could be 2-digit or 4-digit
        
    Returns:
        int: 4-digit year
    """
    year = int(year)
    # If it's already a 4-digit year, return as is
    if year >= 1000:
        return year
    # Otherwise, add the appropriate century
    # Assume years 00-69 are 2000s, years 70-99 are 1900s
    elif year < 70:
        return 2000 + year
    else:
        return 1900 + year

def load_data():
    """Load and prepare the monthly mortality data"""
    try:
        monthly_data_path = "./analysis_results/monthly_deaths.csv"
        df = pd.read_csv(monthly_data_path)
        
        # Convert 2-digit years to 4-digit years if needed
        df['year'] = df['year'].apply(convert_to_four_digit_year)
        
        df.sort_values(["year", "month"], inplace=True)
        
        # Create date column for time series
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + 
                                    df["month"].astype(str) + "-15")
        
        # Add month names for better display
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        df["month_name"] = df["month"].apply(lambda x: month_names[int(x)-1])
        
        # Calculate percentage changes
        df["pct_change"] = df["count"].pct_change() * 100
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_state_data():
    """Load state-level mortality data if available"""
    try:
        state_data_path = "./analysis_results/state_deaths_1985_2004.csv"
        if os.path.exists(state_data_path):
            df = pd.read_csv(state_data_path)
            
            # Convert 2-digit years to 4-digit years if needed
            df['year'] = df['year'].apply(convert_to_four_digit_year)
            
            # Convert numeric state codes to abbreviations if needed
            df['state'] = df['state'].apply(
                lambda x: STATE_FIPS_MAPPING.get(int(x), x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()) else x
            )
            
            df.sort_values(["year", "month", "state"], inplace=True)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading state data: {e}")
        return None

def build_forecasting_model(data, freq='MS'):
    """Build and train a forecasting model for mortality data"""
    try:
        # Prepare the data for time series modeling
        ts_data = data.copy()
        ts_data.set_index('date', inplace=True)
        ts_data = ts_data['count'].resample(freq).sum()
        
        # Train SARIMAX model (handles both trend and seasonality)
        # Order (p,d,q) for ARIMA and (P,D,Q,s) for seasonal component
        model = SARIMAX(ts_data, 
                        order=(1, 1, 1), 
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        results = model.fit(disp=False)
        
        return results, ts_data
    except Exception as e:
        st.error(f"Error building forecast model: {e}")
        return None, None

def forecast_deaths(model, ts_data, steps=72, alpha=0.05):
    """Generate a forecast for future deaths with confidence intervals"""
    try:
        if model is None:
            return None
            
        # Forecast future values (steps months ahead)
        forecast_result = model.get_forecast(steps=steps)
        
        # Get predicted values and confidence intervals
        predicted_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        # Create a dataframe with the forecast results
        forecast_df = pd.DataFrame({
            'date': pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), 
                                 periods=steps, freq='MS'),
            'forecast': predicted_mean.values,
            'lower_ci': conf_int.iloc[:, 0].values,
            'upper_ci': conf_int.iloc[:, 1].values
        })
        
        # Add year and month columns for easier grouping
        forecast_df['year'] = forecast_df['date'].dt.year
        forecast_df['month'] = forecast_df['date'].dt.month
        
        return forecast_df
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        return None

def forecast_state_deaths(national_forecast, state_data, state_name):
    """Forecast deaths for a specific state based on historical proportion"""
    try:
        # Get data for the specific state
        state_df = state_data[state_data['state'] == state_name].copy()
        
        if len(state_df) == 0:
            return None, None
            
        # Calculate the proportion of national deaths for this state
        state_total = state_df['count'].sum()
        
        # Get overlapping years between state data and national data
        state_years = state_df['year'].unique()
        
        # Find national deaths for same period
        national_df = load_data()
        national_df = national_df[national_df['year'].isin(state_years)]
        national_total = national_df['count'].sum()
        
        # Calculate the proportion
        proportion = state_total / national_total
        
        # Apply this proportion to the forecast
        state_forecast = national_forecast.copy()
        state_forecast['forecast'] = state_forecast['forecast'] * proportion
        state_forecast['lower_ci'] = state_forecast['lower_ci'] * proportion
        state_forecast['upper_ci'] = state_forecast['upper_ci'] * proportion
        
        return state_forecast, proportion
    except Exception as e:
        st.error(f"Error forecasting state deaths: {e}")
        return None, None

def main():
    st.set_page_config(
        page_title="Mortality Data Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Interactive Mortality Data Analysis")
    st.write("Explore month-over-month mortality patterns in the United States")
    
    # Load the data
    df = load_data()
    state_df = load_state_data()
    
    if df is None:
        st.error("Could not load mortality data. Please run the mortality_analysis.py script first.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Monthly Trends", 
        "🔄 Year-to-Year Comparison", 
        "🔍 Monthly Details",
        "🗺️ State Analysis (1985-2004)",
        "🔮 Mortality Forecasting"
    ])
    
    with tab1:
        st.header("Monthly Death Trends Over Time")
        
        # Year range selector
        years = sorted(df["year"].unique())
        min_year, max_year = min(years), max(years)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_years = st.slider(
                "Select year range for trend analysis:", 
                min_value=min_year, 
                max_value=max_year, 
                value=(min_year, max_year)
            )
            
            # Visualization options
            viz_type = st.radio(
                "Chart type:",
                ["Line Chart", "Bar Chart", "Area Chart"]
            )
            
            show_pct_change = st.checkbox("Show % change line", value=False)
        
        # Filter data by selected years
        filtered_df = df[(df["year"] >= selected_years[0]) & (df["year"] <= selected_years[1])]
        
        with col2:
            # Create the appropriate chart based on selection
            if viz_type == "Line Chart":
                fig = px.line(
                    filtered_df, 
                    x="date", 
                    y="count",
                    title=f"Monthly Deaths ({selected_years[0]}-{selected_years[1]})",
                    labels={"count": "Number of Deaths", "date": "Date"},
                    markers=True
                )
                
                if show_pct_change:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df["date"],
                            y=filtered_df["pct_change"],
                            mode="lines",
                            name="% Change",
                            yaxis="y2"
                        )
                    )
                    fig.update_layout(
                        yaxis2=dict(
                            title="% Change",
                            overlaying="y",
                            side="right"
                        )
                    )
                    
            elif viz_type == "Bar Chart":
                fig = px.bar(
                    filtered_df, 
                    x="date", 
                    y="count",
                    title=f"Monthly Deaths ({selected_years[0]}-{selected_years[1]})",
                    labels={"count": "Number of Deaths", "date": "Date"}
                )
                
            else:  # Area Chart
                fig = px.area(
                    filtered_df, 
                    x="date", 
                    y="count",
                    title=f"Monthly Deaths ({selected_years[0]}-{selected_years[1]})",
                    labels={"count": "Number of Deaths", "date": "Date"}
                )
            
            fig.update_layout(
                hovermode="x unified",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics summary
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Deaths", f"{filtered_df['count'].sum():,.0f}")
        with col2:
            st.metric("Average Monthly Deaths", f"{filtered_df['count'].mean():,.0f}")
        with col3:
            st.metric("Max Monthly Deaths", f"{filtered_df['count'].max():,.0f}")
        with col4:
            std_dev = np.std(filtered_df["pct_change"].dropna())
            st.metric("Std Dev of % Change", f"{std_dev:.2f}%")
            
    with tab2:
        st.header("Year-to-Year Monthly Comparison")
        
        # Year selector for comparison
        col1, col2 = st.columns([1, 3])
        
        with col1:
            available_years = sorted(df["year"].unique())
            
            selected_year1 = st.selectbox(
                "Select first year:", 
                available_years,
                index=len(available_years)-2
            )
            
            selected_year2 = st.selectbox(
                "Select second year:", 
                available_years,
                index=len(available_years)-1
            )
            
            show_percent_diff = st.checkbox("Show % difference", value=True)
            
        # Filter data for selected years
        year1_data = df[df["year"] == selected_year1]
        year2_data = df[df["year"] == selected_year2]
        
        # Prepare comparison data
        comparison_df = pd.merge(
            year1_data[["month", "month_name", "count"]], 
            year2_data[["month", "count"]], 
            on="month", 
            suffixes=(f"_{selected_year1}", f"_{selected_year2}")
        )
        
        if len(comparison_df) > 0:
            # Calculate percent difference
            comparison_df["pct_diff"] = ((comparison_df[f"count_{selected_year2}"] - 
                                          comparison_df[f"count_{selected_year1}"]) / 
                                         comparison_df[f"count_{selected_year1}"] * 100)
            
            with col2:
                # Create comparison chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=comparison_df["month_name"],
                    y=comparison_df[f"count_{selected_year1}"],
                    name=f"{selected_year1}",
                    marker_color='royalblue'
                ))
                
                fig.add_trace(go.Bar(
                    x=comparison_df["month_name"],
                    y=comparison_df[f"count_{selected_year2}"],
                    name=f"{selected_year2}",
                    marker_color='crimson'
                ))
                
                if show_percent_diff:
                    fig.add_trace(go.Scatter(
                        x=comparison_df["month_name"],
                        y=comparison_df["pct_diff"],
                        mode='lines+markers',
                        name='% Difference',
                        yaxis='y2',
                        line=dict(color='green', width=2)
                    ))
                
                fig.update_layout(
                    title=f"Monthly Deaths Comparison: {selected_year1} vs {selected_year2}",
                    xaxis_title="Month",
                    yaxis_title="Number of Deaths",
                    barmode='group',
                    height=500,
                    yaxis2=dict(
                        title="% Difference",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        zeroline=True,
                        zerolinecolor='gray',
                        zerolinewidth=1
                    ) if show_percent_diff else {},
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                st.subheader("Comparison Summary")
                
                total_diff = year2_data["count"].sum() - year1_data["count"].sum()
                total_pct_diff = (total_diff / year1_data["count"].sum()) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        f"Total Deaths {selected_year1}", 
                        f"{year1_data['count'].sum():,.0f}"
                    )
                with col2:
                    st.metric(
                        f"Total Deaths {selected_year2}", 
                        f"{year2_data['count'].sum():,.0f}"
                    )
                with col3:
                    st.metric(
                        "Change", 
                        f"{total_diff:+,.0f} ({total_pct_diff:+.2f}%)"
                    )
                
                # Show the data table
                st.subheader("Monthly Comparison Data")
                display_df = comparison_df.copy()
                # Format the columns for better display
                display_df[f"count_{selected_year1}"] = display_df[f"count_{selected_year1}"].map('{:,.0f}'.format)
                display_df[f"count_{selected_year2}"] = display_df[f"count_{selected_year2}"].map('{:,.0f}'.format)
                display_df["pct_diff"] = display_df["pct_diff"].map('{:+.2f}%'.format)
                
                # Rename columns for clarity
                display_df = display_df.rename(columns={
                    "month_name": "Month",
                    f"count_{selected_year1}": f"Deaths in {selected_year1}",
                    f"count_{selected_year2}": f"Deaths in {selected_year2}",
                    "pct_diff": "% Difference"
                }).drop("month", axis=1)
                
                st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("Insufficient data for comparison between selected years")
    
    with tab3:
        st.header("Monthly Death Details")
        
        # Year selector for detailed view
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_year = st.selectbox(
                "Select year to examine:", 
                sorted(df["year"].unique())
            )
            
            # Visualization options
            detail_viz = st.radio(
                "Visualization type:",
                ["Bar Chart", "Line Chart", "Table View"]
            )
        
        # Filter data for selected year
        year_data = df[df["year"] == selected_year].copy()
        
        with col2:
            if len(year_data) > 0:
                if detail_viz == "Bar Chart":
                    monthly_fig = px.bar(
                        year_data,
                        x="month_name",
                        y="count",
                        title=f"Monthly Deaths in {selected_year}",
                        labels={"count": "Number of Deaths", "month_name": "Month"},
                        text_auto=True
                    )
                    
                    monthly_fig.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside'
                    )
                    
                    monthly_fig.update_layout(height=500)
                    st.plotly_chart(monthly_fig, use_container_width=True)
                    
                elif detail_viz == "Line Chart":
                    monthly_fig = px.line(
                        year_data,
                        x="month_name",
                        y="count",
                        title=f"Monthly Deaths in {selected_year}",
                        labels={"count": "Number of Deaths", "month_name": "Month"},
                        markers=True,
                    )
                    
                    monthly_fig.update_layout(height=500)
                    st.plotly_chart(monthly_fig, use_container_width=True)
                    
                else:  # Table View
                    st.subheader(f"Monthly Deaths in {selected_year}")
                    
                    # Calculate month-over-month change
                    year_data["mom_change"] = year_data["count"].diff()
                    year_data["mom_pct_change"] = year_data["count"].pct_change() * 100
                    
                    # Format for display
                    display_df = year_data[["month_name", "count", "mom_change", "mom_pct_change"]].copy()
                    display_df["count"] = display_df["count"].map('{:,.0f}'.format)
                    display_df["mom_change"] = display_df["mom_change"].map(lambda x: '{:+,.0f}'.format(x) if pd.notnull(x) else '-')
                    display_df["mom_pct_change"] = display_df["mom_pct_change"].map(lambda x: '{:+.2f}%'.format(x) if pd.notnull(x) else '-')
                    
                    # Rename columns
                    display_df.columns = ["Month", "Deaths", "Change from Previous Month", "% Change from Previous Month"]
                    
                    st.dataframe(display_df, use_container_width=True)
                
                # Summary statistics
                st.subheader(f"Summary for {selected_year}")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Deaths", f"{year_data['count'].sum():,.0f}")
                with col2:
                    st.metric("Average Monthly", f"{year_data['count'].mean():,.0f}")
                with col3:
                    max_month = year_data.loc[year_data["count"].idxmax(), "month_name"]
                    st.metric("Highest Month", f"{max_month} ({year_data['count'].max():,.0f})")
                with col4:
                    min_month = year_data.loc[year_data["count"].idxmin(), "month_name"]
                    st.metric("Lowest Month", f"{min_month} ({year_data['count'].min():,.0f})")
            else:
                st.warning(f"No data available for {selected_year}")
    
    with tab4:
        st.header("State-Level Analysis (1985-2004)")
        
        if state_df is None:
            st.warning("State-level data is not available. Please run the mortality_analysis.py script with state analysis enabled.")
        else:
            # Identify available years in state data
            available_state_years = sorted(state_df["year"].unique())
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Year selector for state data
                selected_state_year = st.selectbox(
                    "Select year:",
                    available_state_years
                )
                
                # Get states for that year and convert codes to names
                states_in_year = sorted(state_df[state_df["year"] == selected_state_year]["state"].unique())
                
                # Create a list of states with full names for display
                display_states = [f"{STATE_NAMES.get(state, state)} ({state})" for state in states_in_year]
                state_to_display_map = dict(zip(display_states, states_in_year))
                
                # Top states or select individual state
                view_type = st.radio(
                    "View type:",
                    ["Top 10 States", "Select State"]
                )
                
                if view_type == "Select State":
                    selected_state_display = st.selectbox("Select state:", display_states)
                    selected_state = state_to_display_map[selected_state_display]
                else:
                    # Get top 10 states by death count
                    top_states = state_df[state_df["year"] == selected_state_year].groupby("state")["count"].sum().nlargest(10).index.tolist()
            
            with col2:
                if view_type == "Select State":
                    # Filter for selected state and year
                    state_year_data = state_df[(state_df["year"] == selected_state_year) & 
                                            (state_df["state"] == selected_state)]
                    
                    # Monthly deaths for selected state and year
                    if len(state_year_data) > 0:
                        state_name = STATE_NAMES.get(selected_state, selected_state)
                        state_fig = px.bar(
                            state_year_data,
                            x="month",
                            y="count",
                            title=f"Monthly Deaths in {state_name} ({selected_state_year})",
                            labels={"count": "Number of Deaths", "month": "Month"},
                            text_auto=True
                        )
                        
                        state_fig.update_traces(
                            texttemplate='%{y:,.0f}',
                            textposition='outside'
                        )
                        
                        state_fig.update_xaxes(
                            tickvals=list(range(1, 13)),
                            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        )
                        
                        state_fig.update_layout(height=500)
                        st.plotly_chart(state_fig, use_container_width=True)
                    else:
                        st.warning(f"No data available for {selected_state} in {selected_state_year}")
                else:
                    # Show top 10 states comparison
                    top_states_data = state_df[(state_df["year"] == selected_state_year) & 
                                            (state_df["state"].isin(top_states))]
                    
                    # Sum by state
                    top_states_sum = top_states_data.groupby("state")["count"].sum().reset_index()
                    top_states_sum = top_states_sum.sort_values("count", ascending=False)
                    
                    # Add full state names for display
                    top_states_sum['display_name'] = top_states_sum['state'].apply(
                        lambda x: f"{STATE_NAMES.get(x, x)}"
                    )
                    
                    top_fig = px.bar(
                        top_states_sum,
                        x="display_name",
                        y="count",
                        title=f"Top 10 States by Death Count ({selected_state_year})",
                        labels={"count": "Number of Deaths", "display_name": "State"},
                        text_auto=True
                    )
                    
                    top_fig.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside'
                    )
                    
                    top_fig.update_layout(height=500)
                    st.plotly_chart(top_fig, use_container_width=True)

    with tab5:
        st.header("🔮 Mortality Forecasting")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Configure Forecast")
            
            # Forecast horizon options
            forecast_years = st.slider(
                "Forecast horizon (years):", 
                min_value=1, 
                max_value=10, 
                value=6
            )
            
            forecast_steps = forecast_years * 12  # Monthly steps
            
            # Confidence interval level
            ci_level = st.slider(
                "Confidence level:", 
                min_value=0.80, 
                max_value=0.99, 
                value=0.95,
                step=0.01
            )
            
            # Show options for annual/monthly view
            view_type = st.radio(
                "View type:",
                ["Monthly Forecast", "Annual Forecast Summary"]
            )
            
            # State forecast options
            st.subheader("State Forecasts")
            show_state_forecast = st.checkbox("Show state-level forecasts", value=True)
            
            if show_state_forecast:
                # Default to New York and New Jersey as per requirements
                selected_states = st.multiselect(
                    "Select states to forecast:",
                    ["NY", "NJ"],
                    default=["NY", "NJ"]
                )
            
            with st.spinner("Building forecast model..."):
                # Build the forecasting model
                model, ts_data = build_forecasting_model(df)
            
            if model is not None:
                st.success("Forecast model ready")
        
        if model is not None:
            with col2:
                # Generate the forecast
                with st.spinner("Generating forecast..."):
                    forecast_df = forecast_deaths(model, ts_data, steps=forecast_steps, alpha=1-ci_level)
                
                if forecast_df is not None:
                    # Limit display range to 2020-2026
                    display_start_year = 2020
                    max_forecast_year = max(forecast_df['year'])
                    
                    # Combine historical data with forecast
                    historical = df[df['year'] >= display_start_year].copy()
                    historical['forecast'] = np.nan
                    historical['lower_ci'] = np.nan
                    historical['upper_ci'] = np.nan
                    
                    # Plot the data based on view type
                    if view_type == "Monthly Forecast":
                        # Monthly view
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=historical['date'],
                            y=historical['count'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'],
                            y=forecast_df['forecast'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Add confidence intervals
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_df['date'], forecast_df['date'].iloc[::-1]]),
                            y=pd.concat([forecast_df['upper_ci'], forecast_df['lower_ci'].iloc[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            showlegend=False
                        ))
                        
                        title = f"Monthly Death Forecast ({display_start_year}-{max_forecast_year})"
                        
                    else:  # Annual Forecast Summary
                        # Aggregate historical data by year
                        historical_annual = historical.groupby('year')['count'].sum().reset_index()
                        
                        # Aggregate forecast data by year
                        forecast_annual = forecast_df.groupby('year').agg({
                            'forecast': 'sum',
                            'lower_ci': 'sum',
                            'upper_ci': 'sum'
                        }).reset_index()
                        
                        # Create annual forecast chart
                        fig = go.Figure()
                        
                        # Add historical annual data
                        fig.add_trace(go.Bar(
                            x=historical_annual['year'],
                            y=historical_annual['count'],
                            name='Historical',
                            marker_color='blue'
                        ))
                        
                        # Add forecast annual data
                        fig.add_trace(go.Bar(
                            x=forecast_annual['year'],
                            y=forecast_annual['forecast'],
                            name='Forecast',
                            marker_color='red'
                        ))
                        
                        # Add error bars for confidence intervals
                        fig.add_trace(go.Scatter(
                            x=forecast_annual['year'],
                            y=forecast_annual['forecast'],
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=forecast_annual['upper_ci'] - forecast_annual['forecast'],
                                arrayminus=forecast_annual['forecast'] - forecast_annual['lower_ci']
                            ),
                            mode='markers',
                            marker=dict(color='red', size=8),
                            name='Confidence Interval'
                        ))
                        
                        title = f"Annual Death Forecast ({display_start_year}-{max_forecast_year})"
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        xaxis_title="Time",
                        yaxis_title="Number of Deaths",
                        hovermode="x unified",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast summary table
                    if view_type == "Annual Forecast Summary":
                        st.subheader("Annual Death Forecast Summary")
                        
                        display_forecast = forecast_annual.copy()
                        
                        # Calculate percent error margin
                        display_forecast['error_margin'] = (
                            (display_forecast['upper_ci'] - display_forecast['lower_ci']) / 
                            (2 * display_forecast['forecast']) * 100
                        )
                        
                        # Format for display
                        display_forecast['forecast'] = display_forecast['forecast'].map('{:,.0f}'.format)
                        display_forecast['lower_ci'] = display_forecast['lower_ci'].map('{:,.0f}'.format)
                        display_forecast['upper_ci'] = display_forecast['upper_ci'].map('{:,.0f}'.format)
                        display_forecast['error_margin'] = display_forecast['error_margin'].map('{:.2f}%'.format)
                        
                        # Rename columns
                        display_forecast.columns = [
                            'Year', 'Predicted Deaths', 'Lower Bound', 
                            'Upper Bound', 'Error Margin'
                        ]
                        
                        st.dataframe(display_forecast, use_container_width=True)
                    
                    # State-level forecasts
                    if show_state_forecast and state_df is not None and len(selected_states) > 0:
                        st.subheader("State-Level Death Forecasts")
                        
                        state_forecasts = {}
                        state_proportions = {}
                        
                        for state in selected_states:
                            state_forecast, proportion = forecast_state_deaths(
                                forecast_df, state_df, state
                            )
                            
                            if state_forecast is not None:
                                state_forecasts[state] = state_forecast
                                state_proportions[state] = proportion
                        
                        if len(state_forecasts) > 0:
                            # State selection for the chart
                            state_tabs = st.tabs(selected_states)
                            
                            for i, state in enumerate(selected_states):
                                if state in state_forecasts:
                                    with state_tabs[i]:
                                        state_forecast = state_forecasts[state]
                                        proportion = state_proportions[state]
                                        
                                        st.write(f"Based on historical data, {state} accounts for approximately " +
                                                f"{proportion*100:.2f}% of national deaths.")
                                        
                                        # Create state forecast visualization
                                        if view_type == "Monthly Forecast":
                                            # Monthly view for state
                                            state_fig = go.Figure()
                                            
                                            # Add forecast
                                            state_fig.add_trace(go.Scatter(
                                                x=state_forecast['date'],
                                                y=state_forecast['forecast'],
                                                mode='lines',
                                                name='Forecast',
                                                line=dict(color='red')
                                            ))
                                            
                                            # Add confidence intervals
                                            state_fig.add_trace(go.Scatter(
                                                x=pd.concat([state_forecast['date'], state_forecast['date'].iloc[::-1]]),
                                                y=pd.concat([state_forecast['upper_ci'], state_forecast['lower_ci'].iloc[::-1]]),
                                                fill='toself',
                                                fillcolor='rgba(255,0,0,0.2)',
                                                line=dict(color='rgba(255,255,255,0)'),
                                                hoverinfo='skip',
                                                showlegend=False
                                            ))
                                            
                                            state_title = f"{state} Monthly Death Forecast ({display_start_year}-{max_forecast_year})"
                                            
                                        else:  # Annual Forecast Summary for state
                                            # Aggregate forecast data by year
                                            state_annual = state_forecast.groupby('year').agg({
                                                'forecast': 'sum',
                                                'lower_ci': 'sum',
                                                'upper_ci': 'sum'
                                            }).reset_index()
                                            
                                            # Create annual forecast chart
                                            state_fig = go.Figure()
                                            
                                            # Add forecast annual data
                                            state_fig.add_trace(go.Bar(
                                                x=state_annual['year'],
                                                y=state_annual['forecast'],
                                                name='Forecast',
                                                marker_color='red'
                                            ))
                                            
                                            # Add error bars for confidence intervals
                                            state_fig.add_trace(go.Scatter(
                                                x=state_annual['year'],
                                                y=state_annual['forecast'],
                                                error_y=dict(
                                                    type='data',
                                                    symmetric=False,
                                                    array=state_annual['upper_ci'] - state_annual['forecast'],
                                                    arrayminus=state_annual['forecast'] - state_annual['lower_ci']
                                                ),
                                                mode='markers',
                                                marker=dict(color='red', size=8),
                                                name='Confidence Interval'
                                            ))
                                            
                                            state_title = f"{state} Annual Death Forecast ({display_start_year}-{max_forecast_year})"
                                            
                                            # Show annual summary table
                                            state_display = state_annual.copy()
                                            
                                            # Calculate percent error margin
                                            state_display['error_margin'] = (
                                                (state_display['upper_ci'] - state_display['lower_ci']) / 
                                                (2 * state_display['forecast']) * 100
                                            )
                                            
                                            # Format for display
                                            state_display['forecast'] = state_display['forecast'].map('{:,.0f}'.format)
                                            state_display['lower_ci'] = state_display['lower_ci'].map('{:,.0f}'.format)
                                            state_display['upper_ci'] = state_display['upper_ci'].map('{:,.0f}'.format)
                                            state_display['error_margin'] = state_display['error_margin'].map('{:.2f}%'.format)
                                            
                                            # Rename columns
                                            state_display.columns = [
                                                'Year', 'Predicted Deaths', 'Lower Bound', 
                                                'Upper Bound', 'Error Margin'
                                            ]
                                            
                                        
                                        # Update layout
                                        state_fig.update_layout(
                                            title=state_title,
                                            xaxis_title="Time",
                                            yaxis_title="Number of Deaths",
                                            hovermode="x unified",
                                            height=500
                                        )
                                        
                                        st.plotly_chart(state_fig, use_container_width=True)
                                        
                                        if view_type == "Annual Forecast Summary":
                                            st.dataframe(state_display, use_container_width=True)
                                
                                else:
                                    with state_tabs[i]:
                                        st.warning(f"Could not generate forecast for {state}. State data might not be available.")
                        else:
                            st.warning("Could not generate state-level forecasts. Please ensure state data is available.")
                            
                    # Download options
                    st.subheader("Download Forecast Data")
                    
                    if view_type == "Monthly Forecast":
                        csv_data = forecast_df.copy()
                    else:
                        csv_data = forecast_annual.copy()
                    
                    # Create a download button
                    csv = csv_data.to_csv(index=False)
                    current_date = datetime.now().strftime("%Y%m%d")
                    st.download_button(
                        "Download Forecast CSV",
                        csv,
                        f"mortality_forecast_{current_date}.csv",
                        "text/csv",
                        key='download-csv'
                    )

    # Footer
    st.markdown("---")
    st.caption("Data source: CDC Mortality Files | Analysis and visualization: Mortality Analysis Tool")

if __name__ == "__main__":
    main()
