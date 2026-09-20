
#  Indian Portfolio Optimizer

An interactive Python and Streamlit dashboard for analyzing Indian stock market data and constructing optimized investment portfolios using **Modern Portfolio Theory (Markowitz Portfolio Optimization)**.

The dashboard retrieves historical stock prices from Yahoo Finance, calculates financial statistics, generates optimized portfolios, and provides portfolio performance and risk analysis.

---

##  Project Overview

The Indian Portfolio Optimizer is designed to explore the relationship between risk and return in a portfolio of Indian stocks.

It uses historical market data and numerical optimization techniques to construct two portfolio strategies:

- Minimum Variance Portfolio
- Maximum Sharpe Ratio Portfolio

Users can select stocks, analyze historical performance, calculate investment allocations, and evaluate portfolio risk through an interactive dashboard.

> **Disclaimer:** This project is intended for educational and analytical purposes. It does not provide personalized financial advice.

---

##  Features

### 1. Historical Market Data

- Retrieves historical stock prices using Yahoo Finance.
- Supports Indian stocks listed on the National Stock Exchange (NSE).
- Historical data begins from January 2019.
- Uses adjusted prices through Yahoo Finance.
- Automatically processes and cleans downloaded data.
- Handles missing values and invalid observations.

### 2. Supported Indian Stocks

The dashboard currently supports:

| Company | Yahoo Finance Ticker |
|---|---|
| Reliance Industries | RELIANCE.NS |
| Tata Consultancy Services | TCS.NS |
| Infosys | INFY.NS |
| HDFC Bank | HDFCBANK.NS |
| ICICI Bank | ICICIBANK.NS |
| ITC | ITC.NS |
| Larsen & Toubro | LT.NS |

Users can select multiple stocks for portfolio analysis.

---

### 3. CSV Upload

Users can upload their own historical price dataset in CSV format.

The uploaded dataset must contain:

- A `Date` column
- At least two valid asset price columns
- At least 30 observations

The dashboard performs:

- Date conversion
- Date sorting
- Numeric conversion
- Missing-value handling
- Infinite-value removal
- Forward filling of missing values
- Dataset validation

---

### 4. Dataset Overview

The dashboard displays:

- Number of assets
- Number of price observations
- Dataset start date
- Dataset end date

It also displays the latest historical price observations.

---

### 5. Financial Statistics

The dashboard calculates annualized financial statistics using 252 trading days per year.

The calculated measures include:

- Daily returns
- Annualized returns
- Annualized volatility
- Correlation matrix
- Annualized covariance matrix

#### Annualized Return

Annualized return is estimated using the average daily return multiplied by 252.

#### Annualized Volatility

Annualized volatility is calculated using:

```text
Annualized Volatility =
Daily Return Standard Deviation × √252
```

#### Annualized Covariance

The daily return covariance matrix is annualized by multiplying it by 252.

---

### 6. Normalized Price Performance

The dashboard provides an interactive line chart comparing normalized stock prices.

Each stock begins at an indexed value of 100, allowing users to compare historical price performance.

```text
Normalized Price =
(Current Price / Initial Price) × 100
```

---

### 7. Correlation Matrix

An interactive correlation heatmap displays the relationship between the daily returns of selected assets.

Correlation values range from:

- `+1`: Perfect positive correlation
- `0`: No linear correlation
- `-1`: Perfect negative correlation

The correlation matrix helps users explore diversification relationships between assets.

---

##  Markowitz Portfolio Optimization

The dashboard uses Modern Portfolio Theory to construct optimized portfolios.

### 1. Minimum Variance Portfolio

The Minimum Variance Portfolio attempts to minimize portfolio variance while satisfying the following constraints:

- Portfolio weights must be between 0 and 1.
- Short selling is not permitted.
- The sum of all portfolio weights must equal 1.

### 2. Maximum Sharpe Ratio Portfolio

The Maximum Sharpe Ratio Portfolio attempts to maximize the Sharpe ratio.

The optimization considers:

- Expected annualized returns
- Portfolio volatility
- Risk-free rate
- Portfolio weight constraints

The default risk-free rate is 6%, and users can adjust it using the dashboard slider.

### Sharpe Ratio

```text
Sharpe Ratio =
(Portfolio Return − Risk-Free Rate) / Portfolio Volatility
```

---

##  Investment Calculator

The investment calculator allows users to enter an investment amount and select an optimized portfolio.

The dashboard calculates:

- Portfolio allocation percentage
- Latest stock price
- Investment amount per stock
- Approximate number of shares
- Total investment amount

The default investment amount is ₹1,00,000.

The share quantities are approximate and do not account for:

- Brokerage charges
- Taxes
- Slippage
- Minimum order restrictions
- Other transaction costs

---

##  Portfolio Allocation Visualization

The dashboard displays interactive pie charts for:

- Minimum Variance Portfolio
- Maximum Sharpe Ratio Portfolio

These charts show the percentage allocation of each selected stock within the optimized portfolios.

---

##  Portfolio Performance Comparison

The dashboard compares the two optimized portfolios using:

- Expected annual return
- Annual volatility
- Sharpe ratio

The comparison is displayed through a data table and an interactive bar chart.

---

##  Historical Portfolio Backtesting

The dashboard estimates how an initial investment of ₹1,00,000 would have performed historically under the two portfolio strategies.

The backtesting section includes:

- Historical portfolio value
- Minimum Variance Portfolio growth
- Maximum Sharpe Portfolio growth
- Final portfolio values
- Total historical returns

The portfolio value is calculated using cumulative daily portfolio returns.

> **Backtesting limitation:** The implementation uses the optimized weights calculated from the selected historical dataset. It does not simulate periodic portfolio rebalancing or transaction costs.

---

##  Portfolio Risk Analysis

### 1. Value at Risk (VaR)

The dashboard calculates historical daily Value at Risk at a 95% confidence level.

VaR estimates a potential daily loss based on historical portfolio returns.

The dashboard displays the estimated VaR for:

- Minimum Variance Portfolio
- Maximum Sharpe Portfolio

Historical VaR is based on past observations and does not guarantee future loss limits.

### 2. Maximum Drawdown

Maximum drawdown measures the largest historical decline from a portfolio's previous peak.

The dashboard calculates maximum drawdown for both optimized portfolios.

### 3. Drawdown Visualization

An interactive line chart displays historical portfolio drawdowns over time.

---

##  Automated Portfolio Insights

The dashboard generates statistical interpretations by comparing:

- Expected annual returns
- Annual volatility
- Sharpe ratios

The insights section also provides a summary table comparing both portfolio strategies.

The interpretations are based on the selected assets, historical data, and chosen risk-free rate.

---

##  Portfolio Validation

The dashboard validates the optimized portfolio weights.

Validation checks include:

- Minimum Variance Portfolio weights sum to 1.
- Maximum Sharpe Portfolio weights sum to 1.
- Portfolio weights remain between 0 and 1.
- Overall validation status.

The dashboard displays whether the portfolio weight constraints have passed.

---

##  Efficient Frontier

The dashboard generates an efficient frontier using different target return levels.

The efficient frontier displays:

- Annualized portfolio return
- Annualized portfolio volatility
- Sharpe ratio

The results are visualized using an interactive scatter plot and a data table.

The efficient frontier is calculated under long-only portfolio constraints.

---

##  Data Download

Users can download the historical stock price data as a CSV file.

The downloaded file contains the cleaned historical price data used in the analysis.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical calculations |
| Plotly | Interactive visualizations |
| SciPy | Portfolio optimization |
| yfinance | Historical market data |

---

##  Project Structure

```text
Indian-Portfolio-Optimizer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .venv/
```

> The virtual environment is excluded from Git tracking using `.gitignore`.

---

##  Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Indian-Portfolio-Optimizer.git
```

### 2. Navigate to the Project Directory

```bash
cd Indian-Portfolio-Optimizer
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Dashboard

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

##  Methodology

The project uses the following workflow:

```text
Historical Market Data
        
Data Cleaning and Validation
        
Daily Return Calculation
        
Annualized Statistics
        
Correlation and Covariance Analysis
        
Portfolio Optimization
        
Investment Allocation
        
Performance and Risk Analysis
```

The optimization process uses the Sequential Least Squares Programming (SLSQP) method from SciPy.

Portfolio constraints include:

- Long-only investments
- Weight range from 0 to 1
- Total portfolio weight equal to 1

---

##  Limitations

- Historical performance does not guarantee future returns.
- Expected returns are estimated from historical data.
- The optimizer is sensitive to the selected time period.
- Transaction costs and taxes are not included.
- The investment calculator provides approximate share quantities.
- Backtesting does not include periodic rebalancing.
- Historical VaR is not a guarantee of future losses.
- Yahoo Finance data availability may vary.
- The model does not account for all real-world investment constraints.
- This project is not financial advice.

---

##  Future Improvements

Potential future improvements include:

- Periodic portfolio rebalancing
- Transaction cost modelling
- NIFTY 50 benchmark comparison
- Additional stocks and ETFs
- Portfolio export to Excel
- Cloud deployment
- More advanced risk metrics
- Custom investment constraints
- Sector-wise diversification analysis
- Interactive portfolio history tracking

---

##  Learning Outcomes

This project provided practical experience in:

- Financial data retrieval
- Data cleaning and preprocessing
- Statistical analysis
- Return and volatility calculation
- Correlation and covariance analysis
- Numerical optimization
- Modern Portfolio Theory
- Risk analysis
- Financial data visualization
- Streamlit dashboard development
- Python project organization

---

##  Author

**Krutika Sahu**

B.Sc. Applied Statistics and Analytics

---

##  Disclaimer

This dashboard is developed for educational and analytical purposes.

The portfolio allocations, returns, risk measures, and optimization results are based on historical data and mathematical assumptions. They should not be considered personalized investment recommendations.
