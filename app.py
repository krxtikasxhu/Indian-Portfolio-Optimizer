
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

st.set_page_config(
    page_title="Indian Portfolio Analytics",
    layout="wide"
)

TRADING_DAYS = 252

INDIAN_ASSETS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS"
}


# ============================================================
# DOWNLOAD REAL DATA
# ============================================================

@st.cache_data(ttl=3600)
def download_real_prices(tickers, start, end=None):

    yahoo_tickers = [
        INDIAN_ASSETS[t] for t in tickers
    ]

    st.write("Downloading real market data from Yahoo Finance...")

    try:

        data = yf.download(
            yahoo_tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column"
        )

        if data is None or data.empty:
            raise ValueError(
                "Yahoo Finance returned no data."
            )

        # Handle different yfinance DataFrame formats

        if isinstance(data.columns, pd.MultiIndex):

            if "Close" not in data.columns.get_level_values(0):
                raise ValueError(
                    "Close prices were not found."
                )

            prices = data["Close"].copy()

        else:

            if "Close" not in data.columns:
                raise ValueError(
                    "Close prices were not found."
                )

            prices = data[["Close"]].copy()

            if len(yahoo_tickers) == 1:
                prices.columns = [yahoo_tickers[0]]

        # Rename Yahoo symbols to simple Indian tickers

        reverse_map = {
            value: key
            for key, value in INDIAN_ASSETS.items()
        }

        prices = prices.rename(columns=reverse_map)

        prices.index = pd.to_datetime(prices.index)

        if getattr(prices.index, "tz", None) is not None:
            prices.index = prices.index.tz_localize(None)

        prices = prices.apply(
            pd.to_numeric,
            errors="coerce"
        )

        prices = prices.replace(
            [np.inf, -np.inf],
            np.nan
        )

        prices = prices.dropna(
            how="all"
        ).ffill().dropna()

        available = [
            t for t in tickers
            if t in prices.columns
            and prices[t].notna().sum() > 0
        ]

        prices = prices[available].dropna()

        if len(available) < 2:
            raise ValueError(
                "Less than two stocks were downloaded."
            )

        if len(prices) < 30:
            raise ValueError(
                "Not enough historical observations."
            )

        return prices

    except Exception as error:

        raise RuntimeError(
            f"Real data download failed: {error}"
        )


# ============================================================
# CLEAN UPLOADED CSV
# ============================================================

def clean_price_data(df):

    if "Date" not in df.columns:
        raise ValueError(
            "CSV must contain a Date column."
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = (
        df.dropna(subset=["Date"])
        .sort_values("Date")
        .set_index("Date")
    )

    valid = []

    for col in df.columns:

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if values.notna().sum() > 0:

            df[col] = values
            valid.append(col)

    df = df[valid]

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
        .ffill()
        .dropna()
    )

    if df.shape[1] < 2:
        raise ValueError(
            "At least two valid assets are required."
        )

    if len(df) < 30:
        raise ValueError(
            "At least 30 observations are required."
        )

    return df


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(prices):

    daily_returns = prices.pct_change().dropna()

    annualized_returns = (
        daily_returns.mean() * TRADING_DAYS
    )

    annualized_volatility = (
        daily_returns.std() * np.sqrt(TRADING_DAYS)
    )

    correlation = daily_returns.corr()

    covariance = (
        daily_returns.cov() * TRADING_DAYS
    )

    summary = pd.DataFrame({
        "Annualized Return": annualized_returns,
        "Annualized Volatility": annualized_volatility
    })

    return (
        daily_returns,
        summary,
        correlation,
        covariance
    )

from scipy.optimize import minimize


def portfolio_metrics(weights, expected_returns, covariance):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(
        weights,
        np.dot(covariance, weights)
    )

    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_return, portfolio_volatility


def optimize_minimum_variance(expected_returns, covariance):
    n_assets = len(expected_returns)

    initial_weights = np.ones(n_assets) / n_assets

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    bounds = [(0, 1) for _ in range(n_assets)]

    result = minimize(
        lambda w: portfolio_metrics(
            w, expected_returns, covariance
        )[1] ** 2,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(result.message)

    return result.x

def optimize_maximum_sharpe(
    expected_returns,
    covariance,
    risk_free_rate=0.06
):
    n_assets = len(expected_returns)

    initial_weights = np.ones(n_assets) / n_assets

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    bounds = [(0, 1) for _ in range(n_assets)]

    def negative_sharpe(weights):
        portfolio_return, portfolio_volatility = (
            portfolio_metrics(
                weights,
                expected_returns,
                covariance
            )
        )

        if portfolio_volatility == 0:
            return 0

        sharpe = (
            portfolio_return - risk_free_rate
        ) / portfolio_volatility

        return -sharpe

    result = minimize(
        negative_sharpe,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(result.message)

    return result.x

def calculate_efficient_frontier(
    expected_returns,
    covariance,
    risk_free_rate=0.06,
    n_points=50
):
    n_assets = len(expected_returns)

    # Find the minimum-variance portfolio
    min_variance_weights = optimize_minimum_variance(
        expected_returns,
        covariance
    )

    min_return, _ = portfolio_metrics(
        min_variance_weights,
        expected_returns,
        covariance
    )

    # Maximum achievable return without short selling
    max_return = float(np.max(expected_returns))

    target_returns = np.linspace(
        min_return,
        max_return,
        n_points
    )

    frontier_rows = []

    for target in target_returns:

        initial_weights = np.ones(n_assets) / n_assets

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1
            },
            {
                "type": "eq",
                "fun": lambda w, target=target:
                    np.dot(w, expected_returns) - target
            }
        ]

        bounds = [(0, 1) for _ in range(n_assets)]

        result = minimize(
            lambda w: np.dot(
                w,
                np.dot(covariance, w)
            ),
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if result.success:

            portfolio_return, portfolio_volatility = (
                portfolio_metrics(
                    result.x,
                    expected_returns,
                    covariance
                )
            )

            sharpe_ratio = (
                portfolio_return - risk_free_rate
            ) / portfolio_volatility

            frontier_rows.append({
                "Return": portfolio_return,
                "Volatility": portfolio_volatility,
                "Sharpe Ratio": sharpe_ratio
            })

    return pd.DataFrame(frontier_rows)

# ============================================================
# DASHBOARD
# ============================================================

st.title("Indian Stock Portfolio Analytics")

st.caption(
    "Phase 1: Real market data loading, cleaning, "
    "and statistical exploration"
)

st.sidebar.header("Choose Input Data")

source = st.sidebar.radio(
    "Data source",
    [
        "Real market data (Yahoo Finance)",
        "Upload CSV"
    ]
)

try:

    if source == "Real market data (Yahoo Finance)":

        selected = st.sidebar.multiselect(
            "Select Indian stocks",
            list(INDIAN_ASSETS.keys()),
            default=[
                "RELIANCE",
                "TCS",
                "INFY",
                "HDFCBANK",
                "ICICIBANK"
            ]
        )

        if len(selected) < 2:

            st.warning(
                "Select at least two stocks."
            )

            st.stop()

        prices = download_real_prices(
            tuple(selected),
            "2019-01-01"
        )

        st.success(
            "Real historical market data loaded "
            "from Yahoo Finance."
        )

    else:

        uploaded = st.sidebar.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        if uploaded is None:

            st.info(
                "Upload a CSV containing Date "
                "and stock price columns."
            )

            st.stop()

        prices = clean_price_data(
            pd.read_csv(uploaded)
        )

        st.success(
            "CSV data loaded successfully."
        )

    daily_returns, summary, correlation, covariance = (
        calculate_statistics(prices)
    )

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Assets",
        prices.shape[1]
    )

    c2.metric(
        "Price observations",
        prices.shape[0]
    )

    c3.metric(
        "Start date",
        str(prices.index.min().date())
    )

    c4.metric(
        "End date",
        str(prices.index.max().date())
    )


    # ========================================================
    # PRICE DATA
    # ========================================================

    st.subheader("Historical Price Data")

    st.dataframe(
        prices.tail(10),
        use_container_width=True
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    st.subheader("Annualized Statistics")

    display_summary = summary.copy()

    display_summary[
        "Annualized Return"
    ] = (
        display_summary["Annualized Return"] * 100
    ).round(2).astype(str) + "%"

    display_summary[
        "Annualized Volatility"
    ] = (
        display_summary["Annualized Volatility"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        display_summary,
        use_container_width=True
    )

    # ========================================================
    # NORMALIZED PRICE PERFORMANCE
    # ========================================================

    st.subheader("Normalized Price Performance")

    normalized = (
        prices / prices.iloc[0] * 100
    )

    long_data = (
        normalized.reset_index()
        .melt(
            id_vars="Date",
            var_name="Asset",
            value_name="Indexed Price"
        )
    )

    fig = px.line(
        long_data,
        x="Date",
        y="Indexed Price",
        color="Asset",
        title="Normalized Prices (Starting Value = 100)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # CORRELATION MATRIX
    # ========================================================

    st.subheader("Correlation Matrix")

    fig_corr = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Daily Return Correlation"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )
    
    st.subheader("Markowitz Portfolio Optimization")
    
    risk_free_rate = st.slider(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5
    ) / 100
    
    expected_returns = summary["Annualized Return"].values
    covariance_matrix = covariance.values
    
    min_variance_weights = optimize_minimum_variance(
        expected_returns,
        covariance_matrix
    )
    
    max_sharpe_weights = optimize_maximum_sharpe(
        expected_returns,
        covariance_matrix,
        risk_free_rate
    )
    
    
    def display_portfolio(name, weights):
        portfolio_return, portfolio_volatility = (
            portfolio_metrics(
                weights,
                expected_returns,
                covariance_matrix
            )
        )
    
        sharpe_ratio = (
            portfolio_return - risk_free_rate
        ) / portfolio_volatility
    
        st.markdown(f"### {name}")
    
        allocation = pd.DataFrame({
            "Asset": prices.columns,
            "Weight": weights * 100
        })
    
        allocation["Weight"] = allocation["Weight"].round(2)
    
        st.dataframe(
            allocation,
            use_container_width=True
        )
    
        c1, c2, c3 = st.columns(3)
    
        c1.metric(
            "Expected Annual Return",
            f"{portfolio_return * 100:.2f}%"
        )
    
        c2.metric(
            "Annual Volatility",
            f"{portfolio_volatility * 100:.2f}%"
        )
    
        c3.metric(
            "Sharpe Ratio",
            f"{sharpe_ratio:.3f}"
        )
    
    
    display_portfolio(
        "Minimum Variance Portfolio",
        min_variance_weights
    )
    
    display_portfolio(
        "Maximum Sharpe Portfolio",
        max_sharpe_weights
    )
    
    # ============================================================
    # INVESTMENT CALCULATOR
    # ============================================================

    st.subheader("Investment Calculator")

    investment_amount = st.number_input(
        "Enter investment amount (₹)",
        min_value=1000.0,
        value=100000.0,
        step=5000.0
    )

    portfolio_choice = st.radio(
        "Choose portfolio for investment allocation",
        [
            "Minimum Variance Portfolio",
            "Maximum Sharpe Portfolio"
        ]
    )

    if portfolio_choice == "Minimum Variance Portfolio":
        selected_weights = min_variance_weights
    else:
        selected_weights = max_sharpe_weights

    # Calculate allocation amount
    allocation_amount = selected_weights * investment_amount

    # Get latest stock prices
    latest_prices = prices.iloc[-1]

    # Calculate approximate number of shares
    number_of_shares = allocation_amount / latest_prices.values

    investment_allocation = pd.DataFrame({
        "Asset": prices.columns,
        "Weight (%)": selected_weights * 100,
        "Latest Price (₹)": latest_prices.values,
        "Investment Amount (₹)": allocation_amount,
        "Approx. Shares": number_of_shares
    })

    investment_allocation["Weight (%)"] = (
        investment_allocation["Weight (%)"].round(2)
    )

    investment_allocation["Latest Price (₹)"] = (
        investment_allocation["Latest Price (₹)"].round(2)
    )

    investment_allocation["Investment Amount (₹)"] = (
        investment_allocation["Investment Amount (₹)"].round(2)
    )

    investment_allocation["Approx. Shares"] = (
        investment_allocation["Approx. Shares"].round(2)
    )

    st.dataframe(
        investment_allocation,
        use_container_width=True
    )

    st.metric(
        "Total Investment",
        f"₹{investment_amount:,.2f}"
    )
    
    # ============================================================
    # PORTFOLIO ALLOCATION VISUALIZATION
    # ============================================================

    st.subheader("Portfolio Allocation Visualization")

    col1, col2 = st.columns(2)

    # Minimum Variance Portfolio Pie Chart
    with col1:

        min_allocation = pd.DataFrame({
            "Asset": prices.columns,
            "Weight": min_variance_weights * 100
        })

        fig_min_pie = px.pie(
            min_allocation,
            names="Asset",
            values="Weight",
            title="Minimum Variance Portfolio",
            hole=0.3
        )

        fig_min_pie.update_traces(
            textposition="inside",
            textinfo="label+percent"
        )

        st.plotly_chart(
            fig_min_pie,
            use_container_width=True
        )

    # Maximum Sharpe Portfolio Pie Chart
    with col2:

        max_allocation = pd.DataFrame({
            "Asset": prices.columns,
            "Weight": max_sharpe_weights * 100
        })

        fig_max_pie = px.pie(
            max_allocation,
            names="Asset",
            values="Weight",
            title="Maximum Sharpe Portfolio",
            hole=0.3
        )

        fig_max_pie.update_traces(
            textposition="inside",
            textinfo="label+percent"
        )

        st.plotly_chart(
            fig_max_pie,
            use_container_width=True
        )
        
    # ============================================================
    # PORTFOLIO PERFORMANCE COMPARISON
    # ============================================================

    st.subheader("Portfolio Performance Comparison")

    # Calculate metrics for Minimum Variance Portfolio
    min_return, min_volatility = portfolio_metrics(
        min_variance_weights,
        expected_returns,
        covariance_matrix
    )

    min_sharpe = (
        (min_return - risk_free_rate) / min_volatility
        if min_volatility > 0 else 0
    )

    # Calculate metrics for Maximum Sharpe Portfolio
    max_return, max_volatility = portfolio_metrics(
        max_sharpe_weights,
        expected_returns,
        covariance_matrix
    )

    max_sharpe = (
        (max_return - risk_free_rate) / max_volatility
        if max_volatility > 0 else 0
    )

    comparison_data = pd.DataFrame({
        "Portfolio": [
            "Minimum Variance",
            "Maximum Sharpe"
        ],
        "Expected Return (%)": [
            min_return * 100,
            max_return * 100
        ],
        "Volatility (%)": [
            min_volatility * 100,
            max_volatility * 100
        ],
        "Sharpe Ratio": [
            min_sharpe,
            max_sharpe
        ]
    })

    st.dataframe(
        comparison_data.round(3),
        use_container_width=True
    )

    # Comparison chart
    chart_data = comparison_data.melt(
        id_vars="Portfolio",
        var_name="Metric",
        value_name="Value"
    )

    fig_comparison = px.bar(
        chart_data,
        x="Portfolio",
        y="Value",
        color="Metric",
        barmode="group",
        title="Portfolio Metrics Comparison"
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )
    
    # ============================================================
    # HISTORICAL PORTFOLIO BACKTESTING
    # ============================================================

    st.subheader("Historical Portfolio Backtesting")

    st.write(
        "This chart shows how ₹1,00,000 would have grown "
        "if invested in each portfolio historically."
    )

    # Calculate daily stock returns
    historical_returns = prices.pct_change().dropna()

    # Align weights with the stock columns
    min_weights_series = pd.Series(
        min_variance_weights,
        index=prices.columns
    )

    max_weights_series = pd.Series(
        max_sharpe_weights,
        index=prices.columns
    )

    # Calculate daily portfolio returns
    min_portfolio_returns = historical_returns.dot(
        min_weights_series
    )

    max_portfolio_returns = historical_returns.dot(
        max_weights_series
    )

    # Initial investment for backtesting
    initial_backtest_investment = 100000

    # Calculate portfolio value over time
    min_portfolio_value = (
        1 + min_portfolio_returns
    ).cumprod() * initial_backtest_investment

    max_portfolio_value = (
        1 + max_portfolio_returns
    ).cumprod() * initial_backtest_investment

    # Create backtesting DataFrame
    backtest_data = pd.DataFrame({
        "Minimum Variance Portfolio": min_portfolio_value,
        "Maximum Sharpe Portfolio": max_portfolio_value
    })

    # Plot historical portfolio growth
    fig_backtest = px.line(
        backtest_data,
        x=backtest_data.index,
        y=backtest_data.columns,
        title="Historical Portfolio Growth",
        labels={
            "value": "Portfolio Value (₹)",
            "index": "Date",
            "variable": "Portfolio"
        }
    )

    st.plotly_chart(
        fig_backtest,
        use_container_width=True
    )

    # Display final portfolio values
    final_min_value = min_portfolio_value.iloc[-1]
    final_max_value = max_portfolio_value.iloc[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Final Minimum Variance Value",
            f"₹{final_min_value:,.2f}"
        )

    with col2:
        st.metric(
            "Final Maximum Sharpe Value",
            f"₹{final_max_value:,.2f}"
        )

    # Calculate total historical returns
    min_total_return = (
        (final_min_value / initial_backtest_investment) - 1
    ) * 100

    max_total_return = (
        (final_max_value / initial_backtest_investment) - 1
    ) * 100

    st.write(
        f"Minimum Variance Total Return: "
        f"**{min_total_return:.2f}%**"
    )

    st.write(
        f"Maximum Sharpe Total Return: "
        f"**{max_total_return:.2f}%**"
    )
    
    # ============================================================
    # RISK ANALYSIS
    # ============================================================

    st.subheader("Portfolio Risk Analysis")

    # ------------------------------------------------------------
    # 1. VALUE AT RISK (VaR)
    # ------------------------------------------------------------

    st.write("### 1. Value at Risk (VaR)")

    st.write(
        "Historical VaR estimates the potential daily loss "
        "based on past portfolio returns."
    )

    confidence_level = 0.95

    min_var = (
        min_portfolio_returns.quantile(1 - confidence_level)
        * initial_backtest_investment
    )

    max_var = (
        max_portfolio_returns.quantile(1 - confidence_level)
        * initial_backtest_investment
    )

    var_col1, var_col2 = st.columns(2)

    with var_col1:
        st.metric(
            "Minimum Variance Daily VaR (95%)",
            f"₹{abs(min_var):,.2f}"
        )

    with var_col2:
        st.metric(
            "Maximum Sharpe Daily VaR (95%)",
            f"₹{abs(max_var):,.2f}"
        )

    # ------------------------------------------------------------
    # 2. MAXIMUM DRAWDOWN
    # ------------------------------------------------------------

    st.write("### 2. Maximum Drawdown")

    st.write(
        "Maximum drawdown measures the largest decline "
        "from a historical peak in portfolio value."
    )

    min_running_max = min_portfolio_value.cummax()
    max_running_max = max_portfolio_value.cummax()

    min_drawdown = (
        min_portfolio_value - min_running_max
    ) / min_running_max

    max_drawdown = (
        max_portfolio_value - max_running_max
    ) / max_running_max

    min_max_drawdown = min_drawdown.min() * 100
    max_max_drawdown = max_drawdown.min() * 100

    dd_col1, dd_col2 = st.columns(2)

    with dd_col1:
        st.metric(
            "Minimum Variance Maximum Drawdown",
            f"{min_max_drawdown:.2f}%"
        )

    with dd_col2:
        st.metric(
            "Maximum Sharpe Maximum Drawdown",
            f"{max_max_drawdown:.2f}%"
        )

    # ------------------------------------------------------------
    # 3. DRAWDOWN CHART
    # ------------------------------------------------------------

    drawdown_data = pd.DataFrame({
        "Minimum Variance Portfolio": min_drawdown * 100,
        "Maximum Sharpe Portfolio": max_drawdown * 100
    })

    fig_drawdown = px.line(
        drawdown_data,
        x=drawdown_data.index,
        y=drawdown_data.columns,
        title="Historical Portfolio Drawdown",
        labels={
            "value": "Drawdown (%)",
            "index": "Date",
            "variable": "Portfolio"
        }
    )

    st.plotly_chart(
        fig_drawdown,
        use_container_width=True
    )
    
    # ============================================================
    # AUTOMATED PORTFOLIO INSIGHTS
    # ============================================================

    st.subheader("Portfolio Insights")

    st.write("### Statistical Interpretation")

    # Compare expected returns
    if max_return > min_return:
        st.info(
            "The Maximum Sharpe Portfolio has a higher expected "
            "annual return than the Minimum Variance Portfolio."
        )
    elif min_return > max_return:
        st.info(
            "The Minimum Variance Portfolio has a higher expected "
            "annual return than the Maximum Sharpe Portfolio."
        )
    else:
        st.info(
            "Both portfolios have the same expected annual return."
        )

    # Compare volatility
    if min_volatility < max_volatility:
        st.success(
            "The Minimum Variance Portfolio has lower annual "
            "volatility, indicating lower estimated risk."
        )
    elif max_volatility < min_volatility:
        st.success(
            "The Maximum Sharpe Portfolio has lower annual "
            "volatility than the Minimum Variance Portfolio."
        )
    else:
        st.info(
            "Both portfolios have the same annual volatility."
        )

    # Compare Sharpe Ratios
    if max_sharpe > min_sharpe:
        st.info(
            "The Maximum Sharpe Portfolio has a higher Sharpe "
            "Ratio based on the selected risk-free rate."
        )
    elif min_sharpe > max_sharpe:
        st.info(
            "The Minimum Variance Portfolio has a higher Sharpe "
            "Ratio based on the selected risk-free rate."
        )
    else:
        st.info(
            "Both portfolios have the same Sharpe Ratio."
        )

    # Display summary table
    insights_summary = pd.DataFrame({
        "Metric": [
            "Expected Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown"
        ],
        "Minimum Variance": [
            f"{min_return * 100:.2f}%",
            f"{min_volatility * 100:.2f}%",
            f"{min_sharpe:.3f}",
            f"{min_max_drawdown:.2f}%"
        ],
        "Maximum Sharpe": [
            f"{max_return * 100:.2f}%",
            f"{max_volatility * 100:.2f}%",
            f"{max_sharpe:.3f}",
            f"{max_max_drawdown:.2f}%"
        ]
    })

    st.dataframe(
        insights_summary,
        use_container_width=True,
        hide_index=True
    )
    
    # ============================================================
    # PORTFOLIO VALIDATION
    # ============================================================

    st.subheader("Portfolio Validation")

    # Check whether weights sum to 1
    min_weight_sum = min_variance_weights.sum()
    max_weight_sum = max_sharpe_weights.sum()

    # Check whether weights are within the allowed range
    min_weights_valid = (
        (min_variance_weights >= 0).all()
        and (min_variance_weights <= 1).all()
    )

    max_weights_valid = (
        (max_sharpe_weights >= 0).all()
        and (max_sharpe_weights <= 1).all()
    )

    # Display validation results
    validation_data = pd.DataFrame({
        "Validation Check": [
            "Minimum Variance Weights Sum",
            "Maximum Sharpe Weights Sum",
            "Minimum Variance Weights Valid",
            "Maximum Sharpe Weights Valid"
        ],
        "Result": [
            f"{min_weight_sum:.6f}",
            f"{max_weight_sum:.6f}",
            "Passed" if min_weights_valid else "Failed",
            "Passed" if max_weights_valid else "Failed"
        ]
    })

    st.dataframe(
        validation_data,
        use_container_width=True,
        hide_index=True
    )

    # Display validation status
    if (
        abs(min_weight_sum - 1) < 0.0001
        and abs(max_weight_sum - 1) < 0.0001
        and min_weights_valid
        and max_weights_valid
    ):
        st.success(
            "All portfolio weight validation checks passed."
        )
    else:
        st.warning(
            "One or more portfolio validation checks failed. "
            "Review the optimization results."
        )

    st.subheader("Efficient Frontier")

    frontier_data = calculate_efficient_frontier(
        expected_returns,
        covariance_matrix,
        risk_free_rate=risk_free_rate
    )

    if not frontier_data.empty:

        fig_frontier = px.scatter(
            frontier_data,
            x="Volatility",
            y="Return",
            color="Sharpe Ratio",
            title="Efficient Frontier",
            labels={
                "Volatility": "Annualized Risk",
                "Return": "Annualized Return"
            }
        )

        st.plotly_chart(
            fig_frontier,
            use_container_width=True
        )

        st.dataframe(
            frontier_data.style.format(
                {
                    "Return": "{:.2%}",
                    "Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.3f}"
                }
            ),
            use_container_width=True
        )

    else:
        st.warning(
            "Could not calculate the efficient frontier."
        )

    # ========================================================
    # DAILY RETURNS
    # ========================================================

    st.subheader("Daily Returns")

    st.dataframe(
    daily_returns.tail(10),
    use_container_width=True
    )

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    st.download_button(
    "Download Historical Price Data",
    prices.to_csv().encode("utf-8"),
    "indian_stock_prices.csv",
    "text/csv"
    )

    st.success(
        "Phase 1 completed successfully."
    )


    # ============================================================
    # NIFTY INDEX ANALYSIS
    # ============================================================

    st.subheader("NIFTY Index Analysis")
    st.write(
        "Analyze uploaded NIFTY Midcap and Smallcap index datasets "
        "separately from the individual-stock portfolio optimizer."
    )

    INDEX_FILES = {
        "NIFTY MIDCAP 50": "midcap 50.csv",
        "NIFTY MIDCAP 100": "midcap100.csv",
        "NIFTY MIDCAP 150": "midcap150.csv",
        "NIFTY SMALLCAP 50": "small 50.csv",
        "NIFTY SMALLCAP 100": "small100.csv",
        "NIFTY SMALLCAP 250": "small150.csv"
    }

    @st.cache_data
    def load_index_data(file_name):
        import os

        file_path = os.path.join(
            os.path.dirname(__file__),
            file_name
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Could not find {file_name}. "
                "Make sure the CSV file is in the same folder as app.py."
            )

        df = pd.read_csv(file_path)

        required_columns = [
            "Date", "Open", "High", "Low", "Close"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {file_name}: {missing}"
            )

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = (
            df.dropna(subset=["Date", "Close"])
            .sort_values("Date")
            .drop_duplicates(subset="Date")
            .set_index("Date")
        )

        df["Daily Return"] = df["Close"].pct_change()

        return df

    selected_indices = st.multiselect(
        "Select NIFTY indices",
        list(INDEX_FILES.keys()),
        default=[
            "NIFTY MIDCAP 50",
            "NIFTY MIDCAP 100",
            "NIFTY MIDCAP 150",
            "NIFTY SMALLCAP 50",
            "NIFTY SMALLCAP 100",
            "NIFTY SMALLCAP 250"
        ]
    )

    if selected_indices:
        index_data = {}
        index_summary_rows = []

        for index_name in selected_indices:
            try:
                df_index = load_index_data(
                    INDEX_FILES[index_name]
                )
                index_data[index_name] = df_index

                returns = df_index["Daily Return"].dropna()
                annualized_return = returns.mean() * TRADING_DAYS
                annualized_volatility = (
                    returns.std() * np.sqrt(TRADING_DAYS)
                )

                running_max = df_index["Close"].cummax()
                drawdown = (
                    df_index["Close"] - running_max
                ) / running_max

                max_drawdown = drawdown.min()

                var_95 = (
                    returns.quantile(0.05)
                )

                index_summary_rows.append({
                    "Index": index_name,
                    "Observations": len(df_index),
                    "Start Date": df_index.index.min().date(),
                    "End Date": df_index.index.max().date(),
                    "Annualized Return": annualized_return,
                    "Annualized Volatility": annualized_volatility,
                    "Maximum Drawdown": max_drawdown,
                    "Daily VaR (95%)": var_95
                })

            except Exception as index_error:
                st.warning(
                    f"Could not load {index_name}: {index_error}"
                )

        if index_data:
            index_summary = pd.DataFrame(index_summary_rows)

            st.markdown("### Index Statistics")

            display_index_summary = index_summary.copy()
            display_index_summary["Annualized Return"] = (
                display_index_summary["Annualized Return"] * 100
            ).round(2).astype(str) + "%"
            display_index_summary["Annualized Volatility"] = (
                display_index_summary["Annualized Volatility"] * 100
            ).round(2).astype(str) + "%"
            display_index_summary["Maximum Drawdown"] = (
                display_index_summary["Maximum Drawdown"] * 100
            ).round(2).astype(str) + "%"
            display_index_summary["Daily VaR (95%)"] = (
                display_index_summary["Daily VaR (95%)"] * 100
            ).round(2).astype(str) + "%"

            st.dataframe(
                display_index_summary,
                use_container_width=True,
                hide_index=True
            )

            # Normalized index performance
            normalized_indices = pd.DataFrame()

            for index_name, df_index in index_data.items():
                normalized_indices[index_name] = (
                    df_index["Close"] /
                    df_index["Close"].iloc[0] * 100
                )

            normalized_long = (
                normalized_indices
                .reset_index()
                .melt(
                    id_vars="Date",
                    var_name="Index",
                    value_name="Indexed Value"
                )
            )

            fig_index_performance = px.line(
                normalized_long,
                x="Date",
                y="Indexed Value",
                color="Index",
                title="NIFTY Index Normalized Performance (Starting Value = 100)"
            )

            st.plotly_chart(
                fig_index_performance,
                use_container_width=True
            )

            # Closing price comparison
            st.markdown("### Historical Closing Prices")

            close_prices = pd.DataFrame({
                index_name: df_index["Close"]
                for index_name, df_index in index_data.items()
            })

            st.line_chart(close_prices)

            # Index return correlation
            st.markdown("### Index Return Correlation")

            index_returns = pd.DataFrame({
                index_name: df_index["Daily Return"]
                for index_name, df_index in index_data.items()
            }).dropna()

            if index_returns.shape[1] >= 2:
                index_correlation = index_returns.corr()

                fig_index_corr = px.imshow(
                    index_correlation,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="NIFTY Index Daily Return Correlation"
                )

                st.plotly_chart(
                    fig_index_corr,
                    use_container_width=True
                )

            # Selected index details
            detail_index = st.selectbox(
                "View detailed data for an index",
                list(index_data.keys())
            )

            detail_df = index_data[detail_index].copy()

            st.markdown(
                f"### {detail_index} — Historical Data"
            )

            st.dataframe(
                detail_df.tail(20),
                use_container_width=True
            )

            st.download_button(
                "Download Selected Index Data",
                detail_df.to_csv().encode("utf-8"),
                f"{detail_index.lower().replace(' ', '_')}_data.csv",
                "text/csv"
            )

    else:
        st.info(
            "Select at least one NIFTY index to display the analysis."
        )

except Exception as error:

    st.error(
        f"Could not process the data: {error}"
    )

    st.info(
        "Check your internet connection, "
        "Yahoo Finance availability, and selected stocks."
    )