# AI Trading Bot 🤖📈

An advanced AI-powered trading bot that analyzes gold (XAU/USD) market data using technical indicators and provides trading insights using LLaMA model.

## Features

- Real-time market data analysis
- Advanced technical indicators (RSI, MACD, EMA, Bollinger Bands, etc.)
- AI-powered trading insights
- User registration and authentication
- Admin dashboard with user management
- Trading history and performance tracking
- Interactive charts and analytics

## Technical Stack

- **Backend**: Flask, SQLAlchemy
- **Database**: SQLite
- **AI Model**: LLaMA
- **Technical Analysis**: Pandas, TA-Lib
- **Charts**: Matplotlib
- **API Integration**: Twelve Data, Groq

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/OLAg-shs/ai-trading-dashboard.git
cd ai-trading-dashboard
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file with the following variables:
```env
TWELVE_API_KEY=your_twelve_data_api_key
GROQ_API_KEY=your_groq_api_key
ADMIN_USER=admin
ADMIN_PWD=your_admin_password
FLASK_SECRET=your_secret_key
```

5. Initialize the database:
```bash
python deploy.py
```

## Running the Application

1. Development mode:
```bash
python deploy.py
```

2. Production mode (using Gunicorn):
```bash
gunicorn app:app
```

## Admin Features

- View all registered users
- Monitor user trading activities
- Enable/disable user trading access
- Track platform-wide performance metrics

## User Features

- Register and manage account
- View personal trading history
- Access AI-powered trading insights
- Track individual performance metrics
- Real-time market analysis

## API Integration

- **Twelve Data**: Used for real-time gold price data
- **Groq**: Powers the AI analysis using LLaMA model

## Security Features

- Password hashing using Werkzeug
- Session management
- Environment variable configuration
- Protected admin routes

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.