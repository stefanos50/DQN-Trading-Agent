Trading Agent using DQN Reinforcement Learning Algorithm with a custom environment (without using GYM library) and PyTorch for the Neural Network.

Datasets:
* [NIFTY-50 Stock Market Data (2000 - 2021)](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data?select=LT.csv)

Actions (Descrete):
* Buy
* Sell
* Hold

Rewards:
* Init Reward (zero)
* Buy without enough budget (negative)
* Sell without any asset (negative)
* Budget Based Reward when buying an asset (negative or positive based on the current budget)
* Sell Profit Reward (positive or zero)

Observation:
* Budget (normalized)
* Number of Assets (normalized)
* Price History with a Window (normalized)

Some of the results:
| Dataset  | Profit |
| ------------- | ------------- |
| IOC  | $13963.10  |
| NTPC  | $2878.95  |
| UPL  | $22665.95  |
| SBIN  | $19313.30  |
| TCS  | $52458.40  |
| ITC  | $25706.35 |
| INFY  | $12698.70  |
| ADANIPORTS  | $4838.00  |
| ICICIBANK  | $32205.40  |
| TATASTEEL  | $18527.10  |
| HDFC  | $77032.55  |
