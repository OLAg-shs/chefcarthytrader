import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from datetime import datetime

# Import run_analysis from your utils/trading_bot.py
from utils.trading_bot import run_analysis
from models import db, User, TradingHistory

# Import Firebase Admin SDK components (optional for local SQLAlchemy setup)
try:
    from firebase_admin import credentials, initialize_app, auth as firebase_auth
except ImportError:
    credentials = None # Handle case where firebase_admin might not be installed or configured

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_strong_random_secret_key_for_development")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trading_bot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Firebase Admin SDK Initialization (for authentication, not Firestore data storage in this setup)
firebase_config = json.loads(os.getenv('__firebase_config', '{}'))
app_id = os.getenv('__app_id', 'default-app-id')

try:
    if credentials and firebase_config and "type" in firebase_config and firebase_config["type"] == "service_account":
        cred = credentials.Certificate(firebase_config)
        if not hasattr(app, 'firebase_initialized'):
            from firebase_admin import initialize_app as firebase_initialize_app
            firebase_initialize_app(cred)
            app.firebase_initialized = True
            print("Firebase Admin SDK initialized with service account credentials.")
    else:
        print("Firebase Admin SDK not initialized with service account. Ensure __firebase_config is correctly set for deployment or firebase_admin is installed.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}. Check __firebase_config or GOOGLE_APPLICATION_CREDENTIALS.")


# Create admin user if not exists
def create_admin_user():
    with app.app_context():
        db.create_all() # Create tables if they don't exist
        admin = User.query.filter_by(username=os.getenv("ADMIN_USER")).first()
        if not admin:
            admin = User(
                username=os.getenv("ADMIN_USER"),
                password=generate_password_hash(os.getenv("ADMIN_PWD")),
                is_admin=True,
                symbol="XAUUSD", # <--- CHANGED THIS LINE TO XAUUSD
                analysis_preference="full" # Admin always sees full analysis
            )
            db.session.add(admin)
            db.session.commit()
            print(f"Admin user '{os.getenv('ADMIN_USER')}' created.")
        else:
            print(f"Admin user '{os.getenv('ADMIN_USER')}' already exists.")

create_admin_user()

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return render_template("register.html")
        
        user = User(
            username=username,
            password=generate_password_hash(password),
            symbol="XAUUSD", # <--- CHANGED THIS LINE TO XAUUSD
            analysis_preference="full" # Default analysis preference for new users
        )
        db.session.add(user)
        db.session.commit()
        
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    
    return render_template("register.html")

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["is_admin"] = user.is_admin
            user.last_login = datetime.utcnow()
            db.session.commit()
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials.", "error")
            return render_template("login.html")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])
    
    analysis_history = TradingHistory.query.filter_by(user_id=user.id).order_by(TradingHistory.timestamp.desc()).limit(10).all()

    context = {
        'user': user,
        'analysis_history': analysis_history,
        'is_admin': session.get('is_admin', False),
        'current_timestamp': datetime.utcnow().timestamp() # Pass current timestamp for cache busting
    }

    if session.get('is_admin'):
        context['all_users'] = User.query.filter_by(is_admin=False).all()
        context['total_users'] = User.query.filter_by(is_admin=False).count()
        context['total_analyses_run'] = TradingHistory.query.count()

    return render_template("dashboard.html", **context)

@app.route("/start_analysis", methods=["POST"])
def start_analysis():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = User.query.get(session["user_id"])
    
    try:
        analysis_text, simplified_summary_data = run_analysis(user.symbol)

        # Handle cases where simplified_summary_data might be None due to API errors
        if simplified_summary_data is None:
            flash(f"❌ Market analysis failed for {user.symbol}. Please check your API key and try again.", "error")
            # Return a 200 OK with a message, as the flash message is the primary feedback
            return jsonify({"success": False, "error": "Market data could not be retrieved or analyzed. Check API key and symbol."}), 200 

        # Store the analysis in TradingHistory
        new_analysis_record = TradingHistory(
            user_id=user.id,
            analysis_full=analysis_text,
            analysis_current_price=simplified_summary_data.get('current_price'),
            analysis_overall_sentiment=simplified_summary_data.get('overall_sentiment'),
            analysis_primary_trend=simplified_summary_data.get('primary_trend'),
            entry_price=simplified_summary_data.get('entry_price'),
            stop_loss=simplified_summary_data.get('stop_loss'),
            take_profit=simplified_summary_data.get('take_profit'),
            expected_loss=simplified_summary_data.get('expected_loss'),
            expected_profit=simplified_summary_data.get('expected_profit'),
            simple_trend=simplified_summary_data.get('simple_trend')
        )
        db.session.add(new_analysis_record)
        db.session.commit()
        flash("Market analysis completed!", "success")

        # If user wants only the summary, return HTML snippet
        if user.analysis_preference == "simple":
            return render_template(
                "summary_snippet.html",
                symbol=user.symbol,
                current_price=simplified_summary_data.get('current_price'),
                overall_sentiment=simplified_summary_data.get('overall_sentiment'),
                primary_trend=simplified_summary_data.get('primary_trend')
            )
        # Otherwise, return full JSON (for admin/advanced users)
        return jsonify({
            "success": True,
            "analysis": analysis_text,
            "simplified_summary": simplified_summary_data,
            "user_preference": user.analysis_preference # Send user's preference back
        })
    except Exception as e:
        flash(f"❌ Analysis failed: {str(e)}", "error")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])
    
    if request.method == "POST":
        # Update trading symbol
        symbol = request.form.get("symbol")
        if symbol:
            user.symbol = symbol.strip()
        
        # Update analysis preference
        analysis_preference = request.form.get("analysis_preference")
        if analysis_preference in ['full', 'simple']:
            user.analysis_preference = analysis_preference
        
        db.session.commit()
        flash("✅ Settings updated successfully!", "success")
        return redirect(url_for("dashboard"))
    
    return render_template("settings.html", user=user)

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    # Ensure tables are created before running the app
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
