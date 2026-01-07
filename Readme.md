# 📈 Quantitative Trading & Feature Engineering Toolkit

A comprehensive suite for **automated alpha discovery**, **order book analysis**, and **live execution**. This project integrates advanced genetic programming and neural networks to bridge the gap between feature engineering and live market trading.

---

## 🚀 Core Features

### 1. Genetic Programming for Feature Engineering
* **Automated Alpha Discovery:** Uses symbolic regression and genetic programming to evolve raw data into predictive mathematical expressions.
* **Fitness Optimization:** Supports custom fitness functions like $Sharpe Ratio$ and $Information Coefficient (IC)$ to rank evolved features.



### 2. Order Book Visualization & Imbalance Analysis
* **Microstructure Observation:** Real-time visualization of the Limit Order Book (LOB) to identify liquidity clusters.
* **Imbalance Metrics:** Tools to detect and visualize **Order Book Imbalance (OBI)** and trade flow toxicity (VPIN).



### 3. Execution Engine (Live Trading)
* **Multi-Exchange Support:** Integrated with **CCXT** for connectivity to 100+ cryptocurrency exchanges.
* **Cybotrade Integration:** High-performance execution via the Cybotrade ecosystem for low-latency order management and reliable API connectivity.

### 4. Neural-Network Based Feature Construction
* **Deep Feature Synthesis:** An automated pipeline that leverages Neural Networks to identify non-linear relationships and construct high-dimensional financial features.

### 5. CTA Backtesting Engine
* **Strategy Validation:** A specialized backtester designed for Trend Following and Commodity Trading Advisor (CTA) strategies.
* **Performance Metrics:** Detailed reporting on Drawdown, Win Rate, Profit Factor, and Equity Curves.



---

## 🛠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Languages** | Python (Pandas, NumPy, Scikit-Learn) |
| **ML/AI** | PyTorch, TensorFlow, gplearn |
| **Trading** | CCXT, Cybotrade API |
| **Visualization** | Matplotlib, Plotly, Dash |

🚧 Ongoing Development
Paper Trading System: Currently building a robust virtual execution environment to test live strategies in real-time market conditions without capital risk.