"""
Streamlit Web Application for Household Energy and Water Consumption Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_preprocessing import load_and_prepare_data
from src.prophet_models import EnergyPredictor, WaterPredictor, train_models, generate_predictions
from src.user_input import (
    parse_user_input, 
    create_comparison_dataframe, 
    calculate_comparison_metrics,
    plot_comparison, 
    plot_error_distribution, 
    generate_comparison_report
)


# Page configuration
st.set_page_config(
    page_title="Household Consumption Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reference date for the dashboard
from datetime import date
APP_TODAY = date.today()

# Custom CSS - Enhanced with modern design
st.markdown("""
    <style>
    :root {
        --primary-bg: #0b1220;
        --surface: rgba(18, 25, 38, 0.88);
        --surface-alt: rgba(21, 31, 49, 0.92);
        --accent: #22d3ee;
        --accent-soft: rgba(34, 211, 238, 0.18);
        --text-light: #f8fafc;
        --text-muted: #cbd5f5;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(34,211,238,0.12), transparent 45%),
                    radial-gradient(circle at 90% 10%, rgba(129,140,248,0.16), transparent 35%),
                    var(--primary-bg);
        color: var(--text-light);
    }
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(34,211,238,0.22), rgba(76,29,149,0.18)),
                    var(--surface);
        box-shadow: 0 35px 80px rgba(15,23,42,0.35);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        line-height: 1.2;
        margin: 0.75rem 0 0.5rem;
        color: var(--text-light);
    }
    .hero p {
        margin: 0;
        color: var(--text-muted);
        font-size: 0.95rem;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.74rem;
    }
    .callout {
        border-radius: 16px;
        padding: 1rem 1.2rem;
        background: var(--surface-alt);
        border: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 1.2rem;
        color: var(--text-muted);
    }
    .metric-card {
        background: var(--surface);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow: 0 18px 45px rgba(8, 15, 30, 0.35);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.35rem;
        color: var(--text-light);
    }
    .metric-card .metric-sub {
        margin-top: 0.35rem;
        font-size: 0.85rem;
        color: rgba(203,213,225,0.85);
    }
    .metric-delta-up {
        color: #22d3ee;
        font-weight: 600;
    }
    .metric-delta-down {
        color: #f97316;
        font-weight: 600;
    }
    .metric-delta-flat {
        color: rgba(203,213,225,0.85);
        font-weight: 600;
    }
    .alert-chip {
        margin-top: 0.75rem;
        padding: 0.6rem 0.9rem;
        border-radius: 12px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(248, 113, 113, 0.5);
        color: #fecaca;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .success-chip {
        margin-top: 0.75rem;
        padding: 0.6rem 0.9rem;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(74, 222, 128, 0.5);
        color: #bbf7d0;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .stTabs [role="tablist"] {
        border-bottom: 1px solid rgba(226,232,240,0.1);
    }
    .stTabs [role="tab"] {
        padding: 0.75rem 1.25rem;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: var(--surface-alt);
    }
    .chart-card {
        background: var(--surface);
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(energy_path, water_folder, sample_size=None):
    """Load and cache the data"""
    return load_and_prepare_data(energy_path, water_folder, energy_sample_size=sample_size)


@st.cache_resource
def get_trained_models(energy_df, water_df):
    """Train and cache the models"""
    return train_models(energy_df, water_df, save_models=False)


def metric_card_html(title: str, value: str, subtitle: str = None) -> str:
    """Return pre-styled HTML snippet for a dashboard metric card."""
    subtitle_html = f"<div class='metric-sub'>{subtitle}</div>" if subtitle else ""
    return (
        "<div class='metric-card'>"
        f"<h3>{title}</h3>"
        f"<div class='metric-value'>{value}</div>"
        f"{subtitle_html}"
        "</div>"
    )


def create_plotly_forecast(predictor, title, y_label):
    """Create an interactive Plotly forecast chart"""
    forecast = predictor.forecast
    train_data = predictor.train_data
    
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=train_data['ds'],
        y=train_data['y'],
        mode='markers',
        name='Actual',
        marker=dict(size=3, color='black', opacity=0.5)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5f5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.18)",
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
    )
    
    return fig


def create_components_plot(predictor, title):
    """Create components plot using Plotly"""
    forecast = predictor.forecast
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Trend', 'Weekly Seasonality', 'Yearly Seasonality'),
        vertical_spacing=0.1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['trend'], 
                  mode='lines', name='Trend', line=dict(color='#d62728')),
        row=1, col=1
    )
    
    # Weekly seasonality
    if 'weekly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['weekly'], 
                      mode='lines', name='Weekly', line=dict(color='#2ca02c')),
            row=2, col=1
        )
    
    # Yearly seasonality
    if 'yearly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yearly'], 
                      mode='lines', name='Yearly', line=dict(color='#ff7f0e')),
            row=3, col=1
        )
    
    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=800,
        template='plotly_white',
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5f5"),
    )
    
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
    
    return fig


def main():
    """Main application function"""
    
    # Hero Header
    st.markdown(
        f"""
        <div class="hero">
            <span class="hero-pill">⚡💧 Prophet AI · {APP_TODAY:%d %b %Y}</span>
            <h1>Forecast tomorrow's energy and water footprint.</h1>
            <p>
                Predict household consumption with Prophet ML, explore multi-year trends,
                validate forecast quality with performance metrics, and optimize your resource usage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Forecast Controls")
        
        # Data paths
        st.subheader("📁 Data Paths")
        energy_path = st.text_input(
            "Energy Data Path",
            value="Dataset/household_power_consumption.txt"
        )
        water_folder = st.text_input(
            "Water Data Folder",
            value="Dataset/Tank 1-Data Files"
        )
        
        # Sampling
        st.subheader("📊 Data Sampling")
        use_sample = st.checkbox("Use sample data (faster)", value=True)
        sample_size = None
        if use_sample:
            sample_size = st.slider("Sample size", 10000, 500000, 100000, 10000)
        
        # Forecast settings
        st.subheader("🔮 Forecast Settings")
        forecast_days = st.slider("Forecast days", 7, 90, 30, 7)
        
        # Load data button
        load_button = st.button("🚀 Load Data & Train Models", type="primary")
    
    # Main content
    if load_button:
        with st.spinner("Loading data..."):
            try:
                energy_df, water_df = load_data(energy_path, water_folder, sample_size)
                st.session_state['energy_df'] = energy_df
                st.session_state['water_df'] = water_df
                st.success("✅ Data loaded successfully!")
                
                # Show data summary
                st.info(f"📊 Loaded {len(energy_df)} energy records and {len(water_df)} water records")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                st.markdown("""
                    <div class="callout">
                        <strong>Troubleshooting tips:</strong><br>
                        • Check that file paths are correct<br>
                        • Ensure data files exist and are accessible<br>
                        • Try using sample data first (check the box in sidebar)<br>
                        • Look for detailed error messages above
                    </div>
                """, unsafe_allow_html=True)
                return
        
        with st.spinner("Training models... This may take a few minutes."):
            try:
                energy_predictor, water_predictor = get_trained_models(energy_df, water_df)
                st.session_state['energy_predictor'] = energy_predictor
                st.session_state['water_predictor'] = water_predictor
                st.success("✅ Models trained successfully!")
                
                # Show quick stats
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"⚡ Energy model ready with {len(energy_df)} training days")
                with col2:
                    st.info(f"💧 Water model ready with {len(water_df)} training days")
                    if len(water_df) < 60:
                        st.warning(f"⚠️ Limited water data ({len(water_df)} days). Predictions may be less accurate. Sensor data spans Feb-Apr 2018 only.")
                    
            except Exception as e:
                st.error(f"❌ Error training models: {e}")
                
                # Show detailed error information
                import traceback
                with st.expander("🔍 Show detailed error"):
                    st.code(traceback.format_exc())
                
                st.markdown("""
                    <div class="callout">
                        <strong>Troubleshooting tips:</strong><br>
                        • Check that data was loaded successfully<br>
                        • Ensure sufficient data points (at least 60 days recommended)<br>
                        • Verify Prophet is installed: <code>pip install prophet</code><br>
                        • Run diagnostic: <code>python test_prophet.py</code><br>
                        • Check terminal output for detailed logs
                    </div>
                """, unsafe_allow_html=True)
                return
        
        with st.spinner("Generating predictions..."):
            try:
                energy_predictor.predict(periods=forecast_days, freq='D')
                water_predictor.predict(periods=forecast_days, freq='D')
                st.session_state['forecast_days'] = forecast_days
                st.success("✅ Predictions generated!")
            except Exception as e:
                st.error(f"❌ Error generating predictions: {e}")
                return
    
    # Display results if models are trained
    if 'energy_predictor' in st.session_state and 'water_predictor' in st.session_state:
        energy_predictor = st.session_state['energy_predictor']
        water_predictor = st.session_state['water_predictor']
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "⚡ Energy Analysis", 
            "💧 Water Analysis", 
            "📈 Predictions",
            "📝 Recent Data",
            "🔍 Compare Your Data"
        ])
        
        with tab1:
            st.header("📊 Dataset Pulse")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚡ Energy Data")
                energy_df = st.session_state['energy_df']
                
                # Summary metrics
                col_a, col_b = st.columns(2)
                col_a.markdown(
                    metric_card_html(
                        "Total Records",
                        f"{len(energy_df):,} days",
                        subtitle=f"{energy_df['ds'].min():%d %b %Y} → {energy_df['ds'].max():%d %b %Y}"
                    ),
                    unsafe_allow_html=True
                )
                col_b.markdown(
                    metric_card_html(
                        "Avg Daily Usage",
                        f"{energy_df['y'].mean():.2f} kWh",
                        subtitle=f"Total: {energy_df['y'].sum():.2f} kWh"
                    ),
                    unsafe_allow_html=True
                )
                
                # Recent trend
                latest_value = energy_df['y'].iloc[-1]
                avg_30_day = energy_df['y'].tail(30).mean()
                delta = latest_value - avg_30_day
                if abs(delta) < 1e-6:
                    delta_class = "metric-delta-flat"
                    delta_symbol = "–"
                elif delta >= 0:
                    delta_class = "metric-delta-up"
                    delta_symbol = "▲"
                else:
                    delta_class = "metric-delta-down"
                    delta_symbol = "▼"
                
                st.markdown(
                    metric_card_html(
                        "Latest Energy Usage",
                        f"{latest_value:.2f} kWh",
                        subtitle=f"30-day avg: {avg_30_day:.2f} kWh · <span class='{delta_class}'>{delta_symbol} {delta:+.2f}</span>"
                    ),
                    unsafe_allow_html=True
                )
                
                # Energy statistics
                with st.expander("📈 Detailed Statistics"):
                    st.dataframe(energy_df.describe(), use_container_width=True)
            
            with col2:
                st.subheader("💧 Water Data")
                water_df = st.session_state['water_df']
                
                # Summary metrics
                col_a, col_b = st.columns(2)
                col_a.markdown(
                    metric_card_html(
                        "Total Records",
                        f"{len(water_df):,} days",
                        subtitle=f"{water_df['ds'].min():%d %b %Y} → {water_df['ds'].max():%d %b %Y}"
                    ),
                    unsafe_allow_html=True
                )
                col_b.markdown(
                    metric_card_html(
                        "Avg Daily Usage",
                        f"{water_df['y'].mean():.2f} L",
                        subtitle=f"Total: {water_df['y'].sum():.2f} L"
                    ),
                    unsafe_allow_html=True
                )
                
                # Recent trend
                latest_value = water_df['y'].iloc[-1]
                avg_30_day = water_df['y'].tail(30).mean()
                delta = latest_value - avg_30_day
                if abs(delta) < 1e-6:
                    delta_class = "metric-delta-flat"
                    delta_symbol = "–"
                elif delta >= 0:
                    delta_class = "metric-delta-up"
                    delta_symbol = "▲"
                else:
                    delta_class = "metric-delta-down"
                    delta_symbol = "▼"
                
                st.markdown(
                    metric_card_html(
                        "Latest Water Usage",
                        f"{latest_value:.2f} L",
                        subtitle=f"30-day avg: {avg_30_day:.2f} L · <span class='{delta_class}'>{delta_symbol} {delta:+.2f}</span>"
                    ),
                    unsafe_allow_html=True
                )
                
                # Water statistics
                with st.expander("📈 Detailed Statistics"):
                    st.dataframe(water_df.describe(), use_container_width=True)
            
            # Combined trend chart
            st.markdown("---")
            st.subheader("📈 Historical Trends")
            
            trend_chart = go.Figure()
            trend_chart.add_trace(
                go.Scatter(
                    x=energy_df['ds'],
                    y=energy_df['y'],
                    name='Energy (kWh)',
                    line=dict(color='#22d3ee', width=2),
                )
            )
            trend_chart.add_trace(
                go.Scatter(
                    x=water_df['ds'],
                    y=water_df['y'],
                    name='Water (Liters)',
                    line=dict(color='#a855f7', width=2),
                    opacity=0.7,
                    yaxis='y2',
                )
            )
            
            trend_chart.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5f5"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hoverlabel=dict(bgcolor="#1e293b"),
                yaxis=dict(title="Energy (kWh)"),
                yaxis2=dict(
                    title="Water (Liters)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
            )
            trend_chart.update_xaxes(
                gridcolor="rgba(148, 163, 184, 0.18)",
                showgrid=True,
                zeroline=False,
            )
            trend_chart.update_yaxes(
                gridcolor="rgba(148, 163, 184, 0.18)",
                zeroline=False,
            )
            
            st.plotly_chart(trend_chart, use_container_width=True, config={"displayModeBar": False})
        
        with tab2:
            st.header("⚡ Energy Consumption Analysis")
            
            # Model metrics with styled cards
            energy_metrics = energy_predictor.get_metrics()
            if energy_metrics:
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.markdown(
                    metric_card_html("MAE", f"{energy_metrics['MAE']:.4f}", "Mean Absolute Error"),
                    unsafe_allow_html=True
                )
                col2.markdown(
                    metric_card_html("MAPE", f"{energy_metrics['MAPE']:.2f}%", "Mean Absolute % Error"),
                    unsafe_allow_html=True
                )
                col3.markdown(
                    metric_card_html("RMSE", f"{energy_metrics['RMSE']:.4f}", "Root Mean Square Error"),
                    unsafe_allow_html=True
                )
            
            # Forecast plot
            st.subheader("Energy Consumption Forecast")
            fig_energy = create_plotly_forecast(
                energy_predictor,
                "Energy Consumption Forecast",
                "Energy (kWh)"
            )
            st.plotly_chart(fig_energy, use_container_width=True)
            
            # Components plot
            st.subheader("Forecast Components")
            fig_energy_comp = create_components_plot(
                energy_predictor,
                "Energy Forecast Components"
            )
            st.plotly_chart(fig_energy_comp, use_container_width=True)
        
        with tab3:
            st.header("💧 Water Consumption Analysis")
            
            # Model metrics with styled cards
            water_metrics = water_predictor.get_metrics()
            if water_metrics:
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.markdown(
                    metric_card_html("MAE", f"{water_metrics['MAE']:.4f}", "Mean Absolute Error"),
                    unsafe_allow_html=True
                )
                col2.markdown(
                    metric_card_html("MAPE", f"{water_metrics['MAPE']:.2f}%", "Mean Absolute % Error"),
                    unsafe_allow_html=True
                )
                col3.markdown(
                    metric_card_html("RMSE", f"{water_metrics['RMSE']:.4f}", "Root Mean Square Error"),
                    unsafe_allow_html=True
                )
            
            # Forecast plot
            st.subheader("Water Consumption Forecast")
            fig_water = create_plotly_forecast(
                water_predictor,
                "Water Consumption Forecast",
                "Water (Liters)"
            )
            st.plotly_chart(fig_water, use_container_width=True)
            
            # Components plot
            st.subheader("Forecast Components")
            fig_water_comp = create_components_plot(
                water_predictor,
                "Water Forecast Components"
            )
            st.plotly_chart(fig_water_comp, use_container_width=True)
        
        with tab4:
            st.header("📈 Future Predictions")
            
            forecast_days = st.session_state.get('forecast_days', 30)
            
            # Prediction summary cards
            st.subheader(f"Forecast Summary ({forecast_days} days)")
            
            energy_summary = energy_predictor.get_forecast_summary(periods=forecast_days)
            water_summary = water_predictor.get_forecast_summary(periods=forecast_days)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Energy prediction card
                avg_energy_pred = energy_summary['Predicted'].mean()
                total_energy_pred = energy_summary['Predicted'].sum()
                
                st.markdown(
                    metric_card_html(
                        "⚡ Energy Forecast",
                        f"{avg_energy_pred:.2f} kWh/day",
                        subtitle=f"Total predicted: {total_energy_pred:.2f} kWh"
                    ),
                    unsafe_allow_html=True
                )
                
                st.markdown(f"#### Next {forecast_days} Days - Energy")
                st.dataframe(energy_summary, use_container_width=True)
                
                # Download button
                csv_energy = energy_summary.to_csv(index=False)
                st.download_button(
                    "📥 Download Energy Forecast",
                    csv_energy,
                    "energy_forecast.csv",
                    "text/csv",
                    key='download-energy'
                )
            
            with col2:
                # Water prediction card
                avg_water_pred = water_summary['Predicted'].mean()
                total_water_pred = water_summary['Predicted'].sum()
                
                st.markdown(
                    metric_card_html(
                        "💧 Water Forecast",
                        f"{avg_water_pred:.2f} L/day",
                        subtitle=f"Total predicted: {total_water_pred:.2f} L"
                    ),
                    unsafe_allow_html=True
                )
                
                st.markdown(f"#### Next {forecast_days} Days - Water")
                st.dataframe(water_summary, use_container_width=True)
                
                # Download button
                csv_water = water_summary.to_csv(index=False)
                st.download_button(
                    "📥 Download Water Forecast",
                    csv_water,
                    "water_forecast.csv",
                    "text/csv",
                    key='download-water'
                )
            
            # Combined view
            st.subheader("Combined Consumption Trends")
            
            # Normalize for comparison
            energy_norm = energy_predictor.forecast.tail(forecast_days).copy()
            water_norm = water_predictor.forecast.tail(forecast_days).copy()
            
            fig_combined = go.Figure()
            
            fig_combined.add_trace(go.Scatter(
                x=energy_norm['ds'],
                y=(energy_norm['yhat'] - energy_norm['yhat'].min()) / (energy_norm['yhat'].max() - energy_norm['yhat'].min()),
                mode='lines',
                name='Energy (Normalized)',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig_combined.add_trace(go.Scatter(
                x=water_norm['ds'],
                y=(water_norm['yhat'] - water_norm['yhat'].min()) / (water_norm['yhat'].max() - water_norm['yhat'].min()),
                mode='lines',
                name='Water (Normalized)',
                line=dict(color='#2ca02c', width=2)
            ))
            
            fig_combined.update_layout(
                title="Normalized Consumption Comparison",
                xaxis_title="Date",
                yaxis_title="Normalized Consumption (0-1)",
                hovermode='x unified',
                template='plotly_white',
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5f5"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            fig_combined.update_xaxes(
                gridcolor="rgba(148, 163, 184, 0.18)",
                showgrid=True,
                zeroline=False,
            )
            fig_combined.update_yaxes(
                gridcolor="rgba(148, 163, 184, 0.18)",
                zeroline=False,
            )
            
            st.plotly_chart(fig_combined, use_container_width=True)
        
        with tab5:
            st.header("📝 Recent Data Records")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚡ Energy - Last 30 Days")
                energy_df = st.session_state['energy_df']
                recent_energy = energy_df.set_index('ds').tail(30)
                st.dataframe(recent_energy, use_container_width=True)
            
            with col2:
                st.subheader("💧 Water - Last 30 Days")
                water_df = st.session_state['water_df']
                recent_water = water_df.set_index('ds').tail(30)
                st.dataframe(recent_water, use_container_width=True)
        
        with tab6:
            st.header("🔍 Compare Your Actual Data with Predictions")
            
            st.markdown("""
                <div class="callout">
                    <strong>How to use:</strong> Enter your actual consumption values below to compare 
                    them against the model's predictions. This helps validate the model's accuracy with 
                    your real-world data.
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚡ Energy Comparison")
                
                st.markdown("#### Input Format")
                st.code("2024-11-01: 45.5\n2024-11-02: 48.2\n2024-11-03: 42.1", language="text")
                
                energy_input = st.text_area(
                    "Enter your actual energy consumption (kWh)",
                    height=200,
                    placeholder="Date: Value\n2024-11-01: 45.5\n2024-11-02: 48.2",
                    help="Format: YYYY-MM-DD: value (one per line)"
                )
                
                if st.button("📊 Compare Energy Data", type="primary"):
                    if energy_input.strip():
                        try:
                            # Parse user input
                            energy_actual = parse_user_input(energy_input)
                            
                            if energy_actual:
                                # Get forecast data
                                energy_predictor = st.session_state['energy_predictor']
                                
                                # Create comparison
                                energy_comparison = create_comparison_dataframe(
                                    energy_actual,
                                    energy_predictor.forecast,
                                    "energy"
                                )
                                
                                if not energy_comparison.empty:
                                    # Store in session state
                                    st.session_state['energy_comparison'] = energy_comparison
                                    
                                    # Calculate metrics
                                    energy_metrics = calculate_comparison_metrics(energy_comparison)
                                    st.session_state['energy_comparison_metrics'] = energy_metrics
                                    
                                    st.success(f"✅ Compared {len(energy_comparison)} data points!")
                                else:
                                    st.warning("⚠️ No matching dates found in forecast data.")
                            else:
                                st.error("❌ Could not parse input. Please check the format.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.warning("⚠️ Please enter some data to compare.")
                
                # Display comparison if available
                if 'energy_comparison' in st.session_state:
                    comparison_df = st.session_state['energy_comparison']
                    metrics = st.session_state['energy_comparison_metrics']
                    
                    st.markdown("---")
                    st.markdown("#### Results")
                    
                    # Metrics cards
                    col_a, col_b, col_c = st.columns(3)
                    col_a.markdown(
                        metric_card_html(
                            "MAE",
                            f"{metrics['mae']:.2f} kWh",
                            "Mean Absolute Error"
                        ),
                        unsafe_allow_html=True
                    )
                    col_b.markdown(
                        metric_card_html(
                            "MAPE",
                            f"{metrics['mape']:.2f}%",
                            "Mean % Error"
                        ),
                        unsafe_allow_html=True
                    )
                    col_c.markdown(
                        metric_card_html(
                            "Accuracy",
                            f"{metrics['accuracy_within_bounds']:.1f}%",
                            "Within Bounds"
                        ),
                        unsafe_allow_html=True
                    )
                    
                    # Comparison table
                    st.markdown("#### Detailed Comparison")
                    display_df = comparison_df[['date', 'actual', 'predicted', 'difference', 'percent_error', 'within_bounds']].copy()
                    display_df.columns = ['Date', 'Actual', 'Predicted', 'Difference', 'Error %', 'Within Bounds']
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Chart
                    st.markdown("#### Visual Comparison")
                    fig = plot_comparison(comparison_df, "Energy: Actual vs Predicted")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Error distribution
                    st.markdown("#### Error Distribution")
                    error_fig = plot_error_distribution(comparison_df, "Energy Prediction Errors")
                    st.plotly_chart(error_fig, use_container_width=True)
                    
                    # Report
                    report = generate_comparison_report(comparison_df, metrics)
                    st.markdown(report)
            
            with col2:
                st.subheader("💧 Water Comparison")
                
                st.markdown("#### Input Format")
                st.code("2024-11-01: 450\n2024-11-02: 480\n2024-11-03: 420", language="text")
                
                water_input = st.text_area(
                    "Enter your actual water consumption (Liters)",
                    height=200,
                    placeholder="Date: Value\n2024-11-01: 450\n2024-11-02: 480",
                    help="Format: YYYY-MM-DD: value (one per line)"
                )
                
                if st.button("📊 Compare Water Data", type="primary"):
                    if water_input.strip():
                        try:
                            # Parse user input
                            water_actual = parse_user_input(water_input)
                            
                            if water_actual:
                                # Get forecast data
                                water_predictor = st.session_state['water_predictor']
                                
                                # Create comparison
                                water_comparison = create_comparison_dataframe(
                                    water_actual,
                                    water_predictor.forecast,
                                    "water"
                                )
                                
                                if not water_comparison.empty:
                                    # Store in session state
                                    st.session_state['water_comparison'] = water_comparison
                                    
                                    # Calculate metrics
                                    water_metrics = calculate_comparison_metrics(water_comparison)
                                    st.session_state['water_comparison_metrics'] = water_metrics
                                    
                                    st.success(f"✅ Compared {len(water_comparison)} data points!")
                                else:
                                    st.warning("⚠️ No matching dates found in forecast data.")
                            else:
                                st.error("❌ Could not parse input. Please check the format.")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.warning("⚠️ Please enter some data to compare.")
                
                # Display comparison if available
                if 'water_comparison' in st.session_state:
                    comparison_df = st.session_state['water_comparison']
                    metrics = st.session_state['water_comparison_metrics']
                    
                    st.markdown("---")
                    st.markdown("#### Results")
                    
                    # Metrics cards
                    col_a, col_b, col_c = st.columns(3)
                    col_a.markdown(
                        metric_card_html(
                            "MAE",
                            f"{metrics['mae']:.2f} L",
                            "Mean Absolute Error"
                        ),
                        unsafe_allow_html=True
                    )
                    col_b.markdown(
                        metric_card_html(
                            "MAPE",
                            f"{metrics['mape']:.2f}%",
                            "Mean % Error"
                        ),
                        unsafe_allow_html=True
                    )
                    col_c.markdown(
                        metric_card_html(
                            "Accuracy",
                            f"{metrics['accuracy_within_bounds']:.1f}%",
                            "Within Bounds"
                        ),
                        unsafe_allow_html=True
                    )
                    
                    # Comparison table
                    st.markdown("#### Detailed Comparison")
                    display_df = comparison_df[['date', 'actual', 'predicted', 'difference', 'percent_error', 'within_bounds']].copy()
                    display_df.columns = ['Date', 'Actual', 'Predicted', 'Difference', 'Error %', 'Within Bounds']
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Chart
                    st.markdown("#### Visual Comparison")
                    fig = plot_comparison(comparison_df, "Water: Actual vs Predicted")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Error distribution
                    st.markdown("#### Error Distribution")
                    error_fig = plot_error_distribution(comparison_df, "Water Prediction Errors")
                    st.plotly_chart(error_fig, use_container_width=True)
                    
                    # Report
                    report = generate_comparison_report(comparison_df, metrics)
                    st.markdown(report)
    
    else:
        # Welcome message with styled callout
        st.markdown(
            """
            <div class="callout">
                <strong>👋 Welcome to the Household Energy & Water Consumption Predictor!</strong><br><br>
                This advanced forecasting tool uses Facebook Prophet ML to predict your household's 
                energy and water consumption with high accuracy.
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Quick Start")
            st.markdown("""
                1. **Configure** data paths in the sidebar
                2. **Choose** sample data option for faster testing
                3. **Set** forecast period (7-90 days)
                4. **Click** "🚀 Load Data & Train Models"
            """)
            
            st.markdown("#### ⏱️ Processing Time")
            st.info("Training may take 2-5 minutes depending on data size. Using sample data is recommended for initial testing.")
        
        with col2:
            st.markdown("#### 🎯 Key Features")
            st.markdown("""
                ✅ **Energy prediction** in kWh with historical analysis  
                ✅ **Water prediction** in Liters from sensor data  
                ✅ **Interactive visualizations** with Plotly charts  
                ✅ **Performance metrics** (MAE, MAPE, RMSE)  
                ✅ **Downloadable forecasts** in CSV format  
                ✅ **Trend analysis** with seasonality detection  
            """)
            
            st.markdown("#### 🔬 Model Technology")
            st.markdown("""
                Powered by **Prophet** - Meta's time series forecasting system that handles:
                - Yearly, weekly, and monthly seasonality
                - Holiday effects and trend changes
                - Missing data and outliers
                - Confidence intervals for predictions
            """)


if __name__ == "__main__":
    main()
