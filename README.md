# PolicyPulse
**Global Monetary Policy Impact (GMP) Research Terminal**

> *Disclaimer: this project is for academic and research purposes only and does not constitute investment advice.*

PolicyPulse is an interactive research terminal that analyzes and simulates how **central bank policy decisions** move financial markets, combining event studies, NLP sentiment on policy statements, and machine learning.

**Live demo:** https://policypulse-devangi.streamlit.app/

---

## Screenshots

| US macro view (FEDFUNDS vs 10Y) | UK macro view (BoE rate pre/post) |
|---|---|
| ![US macro](screenshots/policypulse-1.png) | ![UK macro](screenshots/policypulse-5.png) |

![Monte Carlo and stress test](screenshots/policypulse-3.png)

---

## Overview
The project studies how decisions by the **Federal Reserve (FOMC)** and the **Bank of England (BoE MPC)** affect equity markets and bond yields. It brings macroeconomic indicators, policy-text sentiment and market reactions into one analytical framework, organized as six tabs:

1. **Macro**: policy rate, yields and macro indicators over time
2. **Events**: event study with cumulative abnormal returns (CAR) around announcements
3. **NLP**: sentiment features extracted from FOMC statements (2010 onward)
4. **Diagnostics**: model fit and feature diagnostics
5. **Scenario Sim**: Monte Carlo and stress-test lab driven by sliders (financial stress, inflation, labor market, policy rate, BoE rate shock)
6. **3D Lab**: 3D visualizations of market reactions

## Key Features
- Event study framework (CAR and volatility around announcement windows)
- NLP sentiment scoring of central bank statements
- Random Forest model predicting market reaction
- Stress-testing and scenario simulation
- Interactive Streamlit dashboard with 2D and 3D Plotly charts

## Data Sources
- Federal Reserve Economic Data (FRED)
- FOMC policy statements (2010 to present)
- Bank of England policy rate data
- FTSE 100 market data (Yahoo Finance)

## Tech Stack
Python, Pandas, NumPy, Statsmodels, Scikit-learn, Plotly, Streamlit

---

## Run it locally
```bash
git clone https://github.com/devangi-agrahari/PolicyPulse
cd PolicyPulse
pip install -r requirements.txt
streamlit run gmp_global_app.py
```
The app reads the four CSV files in this repo, so no API keys are needed.

## Future Enhancements
- Expand coverage to additional central banks (ECB, BOJ)
- Intraday data for higher-frequency event analysis
- Finance-specific language models for sentiment
- Portfolio-level backtesting on policy signals

---

## Team
**Team project** by:

- **Devangi Agrahari**: MS in Finance, Quantitative Finance, University at Buffalo
- **Bhanu Teja Dunna**: MS in Finance, Quantitative Finance & FinTech, University at Buffalo

Original repository: https://github.com/Bhanu-Teja-Dunna/PolicyPulse
