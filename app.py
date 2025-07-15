import os
from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for servers

import matplotlib.pyplot as plt
from utils.trading_bot import run_analysis

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "defaultsecret")

ADMIN_USER = os.getenv("ADMIN_USER")
ADMIN_PWD = os.getenv("ADMIN_PWD")

# Global variables
profit_log = []
latest_insight = ""

def generate_dummy_profit():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    profit = round(50 + (len(profit_log) * 12.5), 2)
    profit_log.append((now, profit))
    return profit

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username")
        pwd = request.form.get("password")
        if user == ADMIN_USER and pwd == ADMIN_PWD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="❌ Invalid credentials.")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    global latest_insight
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    generate_dummy_profit()

    # Create profit chart
    if profit_log:
        times, profits = zip(*profit_log)
        plt.figure(figsize=(10, 4))
        plt.plot(times, profits, marker="o", color="lime")
        plt.title("📈 Profit Over Time")
        plt.xlabel("Time")
        plt.ylabel("Profit ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs("static", exist_ok=True)
        chart_path = os.path.join("static", "profit_chart.png")
        plt.savefig(chart_path)
        plt.close()
    else:
        chart_path = None

    return render_template("dashboard.html", chart_path=chart_path, ai_insight=latest_insight)

@app.route("/start", methods=["POST"])
def start_bot():
    global latest_insight
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        insight = run_analysis()
        latest_insight = "🧠 AI Trading Insight:\n\n" + insight
    except Exception as e:
        latest_insight = f"❌ Bot failed: {str(e)}"

    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
