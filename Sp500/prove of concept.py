import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def load_predictions():
    """Load the saved predictions file"""
    try:
        predictions = pd.read_csv('final_model_predictions.csv')
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        return predictions
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return None


def create_historical_plot(data):
    """Create an interactive plot comparing actual vs predicted values"""
    fig = go.Figure()

    # Add predicted values
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Predicted'],
        name='Predicted Returns',
        line=dict(color='#17a2b8', width=2)
    ))

    # Add actual values
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Actual'],
        name='Actual Returns',
        line=dict(color='#28a745', width=2, dash='dot')
    ))

    # Add error band
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Predicted'] + data['Error'],
        name='Error Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Predicted'] - data['Error'],
        name='Error Lower Bound',
        fill='tonexty',
        fillcolor='rgba(23, 162, 184, 0.2)',
        line=dict(width=0),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title='S&P 500 Returns: Historical Performance',
        xaxis_title='Date',
        yaxis_title='Returns (%)',
        template='plotly_dark',
        hovermode='x unified'
    )

    return fig


def calculate_future_value(initial_investment, returns, months):
    """Calculate future value based on predicted returns"""
    value = initial_investment
    values = [value]

    for i in range(months):
        value = value * (1 + returns[i] / 100)
        values.append(value)

    return values


def create_investment_plot(dates, values, initial_investment, confidence_level=0.95):
    """Create an interactive plot showing investment growth"""
    fig = go.Figure()

    # Calculate confidence intervals
    std_dev = np.std(np.diff(values))
    z_score = 1.96  # 95% confidence interval

    upper_bound = [values[0]]
    lower_bound = [values[0]]
    for i in range(1, len(values)):
        upper_bound.append(values[i] + z_score * std_dev * np.sqrt(i))
        lower_bound.append(max(values[i] - z_score * std_dev * np.sqrt(i), 0))

    # Add predicted value line
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        name='Predicted Value',
        line=dict(color='#17a2b8', width=2)
    ))

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=dates,
        y=upper_bound,
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=lower_bound,
        name='Lower Bound',
        fill='tonexty',
        fillcolor='rgba(23, 162, 184, 0.2)',
        line=dict(width=0),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title='Future Investment Projection',
        xaxis_title='Date',
        yaxis_title='Investment Value ($)',
        template='plotly_dark',
        hovermode='x unified'
    )

    return fig


def main():
    # Page config
    st.set_page_config(page_title="S&P 500 Analysis & Prediction", layout="wide")

    # Header
    st.title("📈 S&P 500 Analysis & Investment Predictor")

    # Load predictions
    predictions = load_predictions()

    if predictions is not None:
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Historical Analysis", "💰 Investment Prediction"])

        with tab1:
            st.subheader("Historical Performance Analysis")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                mape = (abs(predictions['Error']) / abs(predictions['Actual'])).mean() * 100
                st.metric("Mean Absolute % Error", f"{mape:.2f}%")

            with col2:
                rmse = (predictions['Error'] ** 2).mean() ** 0.5
                st.metric("Root Mean Square Error", f"{rmse:.2f}%")

            with col3:
                accuracy = 100 - mape
                st.metric("Model Accuracy", f"{accuracy:.2f}%")

            with col4:
                correlation = predictions['Actual'].corr(predictions['Predicted'])
                st.metric("Correlation", f"{correlation:.2f}")

            # Time period selection
            st.subheader("📅 Select Time Period")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=predictions['Date'].min().date(),
                    min_value=predictions['Date'].min().date(),
                    max_value=predictions['Date'].max().date()
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=predictions['Date'].max().date(),
                    min_value=predictions['Date'].min().date(),
                    max_value=predictions['Date'].max().date()
                )

            # Filter data
            mask = (predictions['Date'].dt.date >= start_date) & (predictions['Date'].dt.date <= end_date)
            filtered_data = predictions.loc[mask]

            # Historical plot
            st.plotly_chart(create_historical_plot(filtered_data), use_container_width=True)

            # Recent predictions
            st.subheader("🔍 Recent Performance")
            recent_data = filtered_data.tail(10).sort_values('Date', ascending=False)
            st.dataframe(
                recent_data[['Date', 'Actual', 'Predicted', 'Error']].style.format({
                    'Actual': '{:.2f}%',
                    'Predicted': '{:.2f}%',
                    'Error': '{:.2f}%'
                })
            )

        with tab2:
            st.subheader("Investment Projection")

            # Investment inputs
            col1, col2, col3 = st.columns(3)

            with col1:
                initial_investment = st.number_input(
                    "Initial Investment ($)",
                    min_value=100,
                    max_value=10000000,
                    value=10000,
                    step=100
                )

            with col2:
                investment_months = st.slider(
                    "Investment Horizon (months)",
                    min_value=1,
                    max_value=12,
                    value=6
                )

            with col3:
                risk_tolerance = st.select_slider(
                    "Risk Tolerance",
                    options=["Conservative", "Moderate", "Aggressive"],
                    value="Moderate"
                )

            # Calculate projections
            start_date = datetime.now()
            dates = [start_date + timedelta(days=30 * i) for i in range(investment_months + 1)]
            historical_returns = predictions['Predicted'].tail(investment_months).values
            values = calculate_future_value(initial_investment, historical_returns, investment_months)

            # Display projection metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                final_value = values[-1]
                st.metric(
                    "Projected Final Value",
                    f"${final_value:,.2f}",
                    f"{((final_value / initial_investment - 1) * 100):.1f}%"
                )

            with col2:
                total_return = final_value - initial_investment
                st.metric("Total Return", f"${total_return:,.2f}")

            with col3:
                monthly_return = ((final_value / initial_investment) ** (1 / investment_months) - 1) * 100
                st.metric("Average Monthly Return", f"{monthly_return:.2f}%")

            # Projection plot
            st.plotly_chart(create_investment_plot(dates, values, initial_investment), use_container_width=True)

            # Projection table
            st.subheader("📅 Monthly Projection")
            projection_df = pd.DataFrame({
                'Date': dates,
                'Projected Value': values,
                'Monthly Return (%)': [0] + list(historical_returns),
                'Cumulative Return (%)': [(v / initial_investment - 1) * 100 for v in values]
            })

            st.dataframe(
                projection_df.style.format({
                    'Projected Value': '${:,.2f}',
                    'Monthly Return (%)': '{:.2f}%',
                    'Cumulative Return (%)': '{:.2f}%'
                })
            )




        # Download buttons (bottom of page)
        st.subheader("⬇️ Download Data")
        col1, col2 = st.columns(2)

        with col1:
            hist_csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download Historical Analysis",
                data=hist_csv,
                file_name=f"sp500_historical_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            proj_csv = projection_df.to_csv(index=False)
            st.download_button(
                label="Download Projection Data",
                data=proj_csv,
                file_name=f"sp500_projection_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()