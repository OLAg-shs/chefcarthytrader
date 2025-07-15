import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

import matplotlib.pyplot as plt
import requests
import ta
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def fetch_price():
    url = "https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&apikey=" + os.getenv("TWELVE_API_KEY")
    data = requests.get(url).json()
    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.astype(float).sort_index()
    return df

def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['ema12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df

def generate_chart(df):
    plt.figure(figsize=(10, 5))
    df[['close', 'ema12', 'ema26']].plot(ax=plt.gca())
    plt.title("XAU/USD Price & EMAs")
    plt.savefig("static/market_chart.png")
    plt.close()

def generate_prompt(df):
    latest = df.iloc[-1]
    return f"""
Based on the latest data:
- Price: {latest['close']:.2f}
- RSI: {latest['rsi']:.2f}
- MACD: {latest['macd']:.2f}
- EMA12: {latest['ema12']:.2f}
- EMA26: {latest['ema26']:.2f}
- BB High: {latest['bb_high']:.2f}, BB Low: {latest['bb_low']:.2f}

Give a complete professional analysis, clear entry, stop loss, take profit, trend prediction, and explain it in beginner-friendly terms.
"""

def run_analysis():
    df = fetch_price()
    df = add_indicators(df)
    generate_chart(df)
    prompt = generate_prompt(df)
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content
