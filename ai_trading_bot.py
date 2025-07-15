# === Flask Project Structure ===
#.
#├── app.py
#├── .env
#├── templates/
#│   ├── login.html
#│   ├── dashboard.html
#├── static/
#│   └── market_chart.png
#├── utils/
#│   └── trading_bot.py
#└── requirements.txt

# STEP 1: requirements.txt
# Put this in requirements.txt:
'''
flask
flask-login
matplotlib
pandas
ta
python-dotenv
requests
groq
schedule
email-validator
'''

# STEP 2: .env (sensitive keys)
# Add your secrets here (already provided by you):
'''
GROQ_API_KEY=your-groq-key
EMAIL_ADDR=you@gmail.com
EMAIL_PASSWORD=your-app-password
ADMIN_USER=admin
ADMIN_PWD=pass123
FLASK_SECRET=supersecret
'''

# STEP 3: utils/trading_bot.py (the actual bot logic)
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
import ta
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_ADDR = os.getenv("EMAIL_ADDR")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

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

# STEP 4: app.py (Flask web app)
from flask import Flask, render_template, request, redirect, session
from utils.trading_bot import run_analysis
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET")

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form['username'] == os.getenv("ADMIN_USER") and request.form['password'] == os.getenv("ADMIN_PWD"):
            session['user'] = "admin"
            return redirect("/dashboard")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if session.get("user") != "admin":
        return redirect("/")
    analysis = run_analysis()
    return render_template("dashboard.html", analysis=analysis)

if __name__ == "__main__":
    app.run(debug=True)

# STEP 5: templates/login.html
'''
<!doctype html>
<html><body>
<h2>Login</h2>
<form method="POST">
  <input name="username" placeholder="Username"><br>
  <input type="password" name="password" placeholder="Password"><br>
  <button type="submit">Login</button>
</form>
</body></html>
'''

# STEP 6: templates/dashboard.html
'''
<!doctype html>
<html><body>
<h2>Trading Dashboard</h2>
<img src="/static/market_chart.png" width="600">
<pre>{{ analysis }}</pre>
</body></html>
'''

# === HOW TO RUN LOCALLY ===
# 1. Create folder and place files accordingly
# 2. Create and activate virtual environment:
#    python -m venv venv && venv\Scripts\activate
# 3. Install dependencies:
#    pip install -r requirements.txt
# 4. Add your .env keys in root
# 5. Run Flask app:
#    python app.py
# 6. Visit: http://127.0.0.1:5000
