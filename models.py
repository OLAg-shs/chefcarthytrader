from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    # Trading symbol/market
    symbol = db.Column(db.String(20), default='XAUUSDm')
    # Analysis preferences
    analysis_preference = db.Column(db.String(10), default='simple')  # 'simple' or 'full'
    def __repr__(self):
        return f'<User {self.username}>'

class TradingHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    symbol = db.Column(db.String(20))  # The market analyzed
    analysis_full = db.Column(db.Text)  # Full analysis text
    analysis_current_price = db.Column(db.Float)  # Current price at analysis time
    analysis_overall_sentiment = db.Column(db.String(20))  # bullish/bearish/neutral
    analysis_primary_trend = db.Column(db.String(20))  # primary trend direction
    # New simple actionable fields
    entry_price = db.Column(db.Float)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    expected_loss = db.Column(db.Float)
    expected_profit = db.Column(db.Float)
    simple_trend = db.Column(db.String(32))

    user = db.relationship('User', backref=db.backref('analyses', lazy=True))

    def __repr__(self):
        return f'<TradingHistory {self.id}>'
