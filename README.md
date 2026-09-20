# Indian-Portfolio-Optimizer
#  Indian Stock Portfolio Analytics & Optimization

An interactive Streamlit dashboard for analyzing Indian stocks and constructing optimized investment portfolios using Modern Portfolio Theory (Markowitz Portfolio Optimization).

##  Features

* Real-time historical market data through Yahoo Finance
* Historical price analysis
* Annualized returns and volatility
* Covariance and correlation analysis
* Minimum Variance Portfolio optimization
* Maximum Sharpe Ratio Portfolio optimization
* Efficient Frontier visualization
* Investment allocation calculator
* Portfolio allocation charts
* Historical portfolio backtesting
* Value at Risk (VaR)
* Maximum Drawdown analysis
* Automated portfolio insights
* Portfolio weight validation

##  Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* SciPy
* yfinance

##  Portfolio Optimization

The dashboard implements two optimization approaches:

### Minimum Variance Portfolio

Constructs a portfolio that minimizes estimated portfolio variance subject to the implemented investment constraints.

### Maximum Sharpe Ratio Portfolio

Optimizes the portfolio's estimated risk-adjusted return using the Sharpe Ratio and selected risk-free rate.

##  Supported Assets

The dashboard includes selected Indian stocks listed on the NSE, including:

* Reliance Industries
* TCS
* Infosys
* HDFC Bank
* ICICI Bank
* ITC
* Larsen & Toubro

##  How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Indian-Portfolio-Optimizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

##  Disclaimer

This dashboard is intended for educational and analytical purposes. Historical performance and optimization results do not guarantee future investment returns. The results depend on the selected data period, assumptions, and methodology.

## 👨‍💻 Author
Krutika Sahu
