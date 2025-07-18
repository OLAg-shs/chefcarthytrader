import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Use non-GUI backend for matplotlib

import matplotlib.pyplot as plt
import requests
import ta
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
import logging
import re
import json # Import json for parsing AI output

# Configure logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def fetch_price(symbol='XAUUSD', interval='1h'): # Default symbol is XAUUSD
    """Fetches historical price data from TwelveData."""
    if not TWELVE_API_KEY:
        logger.error("TWELVE_API_KEY is not set in environment variables.")
        raise ValueError("TwelveData API key is missing.")

    # Convert symbol to TwelveData format (with forward slash)
    if symbol == 'XAUUSD':
        api_symbol = 'XAU/USD'
    elif symbol == 'GBPUSD':
        api_symbol = 'GBP/USD'
    else:
        api_symbol = symbol
        
    # First validate the symbol using the quote endpoint
    quote_url = f"https://api.twelvedata.com/quote?symbol={api_symbol}&apikey={TWELVE_API_KEY}"
    try:
        quote_response = requests.get(quote_url)
        quote_response.raise_for_status()
        quote_data = quote_response.json()
        
        # Check for rate limit headers
        credits_used = quote_response.headers.get('api-credits-used')
        credits_left = quote_response.headers.get('api-credits-left')
        if credits_used and credits_left:
            logger.info(f"API Credits - Used: {credits_used}, Left: {credits_left}")
        
        # Check for error responses
        if 'code' in quote_data:
            error_msg = quote_data.get('message', 'Unknown error')
            if quote_data.get('code') == 429:
                logger.error(f"Rate limit exceeded for {api_symbol}. Credits left: {credits_left}")
                raise ValueError(f"API rate limit exceeded. Please try again in a minute.")
            logger.error(f"TwelveData API error for {api_symbol}: {quote_data}")
            raise ValueError(f"Invalid symbol or API key: {error_msg}")
            
        if 'symbol' not in quote_data:
            logger.error(f"Unexpected API response format for {api_symbol}: {quote_data}")
            raise ValueError(f"Invalid response from TwelveData API for {api_symbol}")
            
        logger.info(f"Successfully validated symbol {api_symbol} with quote endpoint")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking symbol validity for {api_symbol}: {e}")
        raise ConnectionError(f"Could not connect to TwelveData API: {e}")

    url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&outputsize=100&apikey={TWELVE_API_KEY}"
    
    # This log will show the exact URL and symbol being used for the API call
    logger.info(f"Attempting to fetch data for requested symbol: '{symbol}'. Using API symbol: '{api_symbol}'")

    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        # Check rate limit headers
        credits_used = response.headers.get('api-credits-used')
        credits_left = response.headers.get('api-credits-left')
        if credits_used and credits_left:
            logger.info(f"Time Series API Credits - Used: {credits_used}, Left: {credits_left}")
        
        data = response.json()
        
        # Check for API error responses
        if 'code' in data:
            error_msg = data.get('message', 'Unknown error')
            if data.get('code') == 429:
                logger.error(f"Rate limit exceeded for time series request. Credits left: {credits_left}")
                raise ValueError("API rate limit exceeded. Please try again in a minute.")
            logger.error(f"TwelveData API error: {data}")
            raise ValueError(f"API error: {error_msg}")

        if 'values' not in data or not data['values']:
            logger.warning(f"No data received from TwelveData for {api_symbol} with interval {interval}. Response: {data}")
            raise ValueError(f"No market data available for {symbol}. The symbol might be invalid or temporarily unavailable.")

        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.astype(float).sort_index()
        logger.info(f"Successfully fetched {len(df)} data points for {symbol}.")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while fetching data from TwelveData: {e}")
        raise ConnectionError(f"Could not connect to TwelveData API. Please check your internet connection and try again.")
    except ValueError as e:
        logger.error(f"Error processing TwelveData response: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while fetching data: {e}")
        raise ValueError(f"An unexpected error occurred while fetching market data: {str(e)}")

def add_indicators(df):
    """Adds various technical indicators to the DataFrame."""
    if df.empty:
        logger.warning("DataFrame is empty, cannot add indicators.")
        return df

    # Ensure columns are numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['close'], inplace=True) # Drop rows where close price is NaN

    if df.empty: # Check again after dropping NaNs
        logger.warning("DataFrame became empty after dropping NaNs, cannot add indicators.")
        return df

    # Trend Indicators
    df['sma200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
    df['ema12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

    # Momentum Indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Volume and Trend Strength
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()

    # Volatility Indicators
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

    # Support/Resistance & Pivot Points
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    logger.info("Indicators added to DataFrame.")
    return df

def generate_chart(df, symbol='XAUUSD'): # Changed default symbol to XAUUSD
    """Generates and saves a market chart as a PNG image."""
    if df.empty:
        logger.warning("DataFrame is empty, cannot generate chart.")
        return

    if len(df) < 2:
        logger.warning("Not enough data points to generate a meaningful chart.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Price and Moving Averages
    ax1.plot(df.index, df['close'], label='Price', color='#2962FF', linewidth=1.5)
    # Check if indicator columns exist and are not all NaN before plotting
    if 'ema12' in df.columns and not df['ema12'].isnull().all():
        ax1.plot(df.index, df['ema12'], label='EMA12', color='#FF6D00', linewidth=1)
    if 'ema26' in df.columns and not df['ema26'].isnull().all():
        ax1.plot(df.index, df['ema26'], label='EMA26', color='#00C853', linewidth=1)
    if 'sma200' in df.columns and not df['sma200'].isnull().all():
        ax1.plot(df.index, df['sma200'], label='SMA200', color='#AA00FF', linewidth=1)

    # Bollinger Bands
    if 'bb_high' in df.columns and 'bb_low' in df.columns and not df['bb_high'].isnull().all() and not df['bb_low'].isnull().all():
        ax1.fill_between(df.index, df['bb_high'], df['bb_low'], alpha=0.1, color='gray')

    ax1.set_title(f'{symbol} Technical Analysis', color='white')
    ax1.set_ylabel('Price', color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left', frameon=False, labelcolor='white')
    ax1.set_facecolor('#1a202c') # Dark background for chart area
    fig.patch.set_facecolor('#1a202c') # Dark background for figure

    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_hist' in df.columns:
        ax2.plot(df.index, df['macd'], label='MACD', color='#2962FF', linewidth=1)
        ax2.plot(df.index, df['macd_signal'], label='Signal', color='#FF6D00', linewidth=1)
        ax2.bar(df.index, df['macd_hist'], label='Histogram', color=['#00C853' if v >= 0 else '#FF1744' for v in df['macd_hist']], alpha=0.5)

    ax2.set_ylabel('MACD', color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper left', frameon=False, labelcolor='white')
    ax2.set_facecolor('#1a202c') # Dark background for chart area

    plt.tight_layout()
    chart_path = "static/market_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Chart saved to {chart_path}.")

def analyze_market_conditions(df):
    """Analyzes market conditions based on the latest DataFrame row."""
    if df.empty:
        logger.warning("DataFrame is empty, cannot analyze market conditions.")
        return {}

    if len(df) < 2:
        logger.warning("Not enough data points for comprehensive market conditions analysis (need at least 2).")
        latest = df.iloc[-1]
        # Provide default/N/A values if not enough data for full analysis
        return {
            'trend': {
                'short_term': 'N/A',
                'long_term': 'N/A',
                'strength': 'N/A'
            },
            'momentum': {
                'rsi': 'N/A',
                'stochastic': 'N/A'
            },
            'volatility': {
                'bb_position': 'N/A',
                'bb_width': 'N/A'
            },
            'signals': {
                'macd': 'N/A',
                'price_action': 'N/A'
            }
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    conditions = {
        'trend': {
            'short_term': 'bullish' if latest['ema12'] > latest['ema26'] else 'bearish',
            'long_term': 'bullish' if latest['close'] > latest['sma200'] else 'bearish',
            'strength': 'strong' if latest['adx'] > 25 else 'weak'
        },
        'momentum': {
            'rsi': 'overbought' if latest['rsi'] > 70 else 'oversold' if latest['rsi'] < 30 else 'neutral',
            'stochastic': 'overbought' if latest['stoch_k'] > 80 else 'oversold' if latest['stoch_k'] < 20 else 'neutral'
        },
        'volatility': {
            'bb_position': 'upper' if latest['close'] > latest['bb_mid'] else 'lower',
            'bb_width': 'expanding' if latest['bb_width'] > prev['bb_width'] else 'contracting'
        },
        'signals': {
            'macd': 'bullish' if latest['macd'] > latest['macd_signal'] else 'bearish',
            'price_action': 'bullish' if latest['close'] > prev['close'] else 'bearish'
        }
    }
    logger.info("Market conditions analyzed.")
    return conditions

def generate_prompt(df, symbol='XAUUSD'): # Changed default symbol to XAUUSD
    """Generates the prompt for the AI model based on market data."""
    latest = df.iloc[-1]
    conditions = analyze_market_conditions(df)

    # Ensure all values are available before formatting
    def get_value(df_row, key, default='N/A', precision=2):
        if key in df_row and pd.notna(df_row[key]):
            return f"{df_row[key]:.{precision}f}"
        return default

    return f"""
You are an advanced AI trading assistant. Your job is to provide a professional, step-by-step, and beginner-friendly market analysis for **{symbol}**. Your analysis must:
- Clearly explain what each indicator means, in plain English, and what it signals right now.
- Give a direct, actionable recommendation: entry (buy/sell/hold), stop loss, take profit, and trend prediction.
- Use analogies or simple explanations for beginners (e.g., compare indicators to car dashboard, weather, etc.).
- Avoid generic phrases like "mixed" or "complex interplay"—be specific and direct.
- Summarize everything in a way that even a non-trader can understand and act on.

**CRITICAL: You MUST use these exact values in your analysis and recommendations.**
- The current price is: {get_value(latest, 'close')}
- RSI: {get_value(latest, 'rsi')}
- MACD: {get_value(latest, 'macd')}
- EMA12: {get_value(latest, 'ema12')}
- EMA26: {get_value(latest, 'ema26')}
- BB High: {get_value(latest, 'bb_high')}
- BB Low: {get_value(latest, 'bb_low')}
- ATR: {get_value(latest, 'atr')}

**You MUST NOT invent, guess, or use any other numbers. If you do not use the provided values, your output will be rejected.**

**Your summary JSON block MUST include:**
- "expected_profit": [expected profit as float, calculated as abs(take_profit - entry_price)]

**Example Style:**

AI Trading Insight:

Based on the current market data, here's a comprehensive analysis:

_Current Price:_ {get_value(latest, 'close')}

_RSI (Relative Strength Index):_ {get_value(latest, 'rsi')} (explain if neutral, overbought, oversold, and what it means for the next move)

_MACD (Moving Average Convergence Divergence):_ (explain value, bullish/bearish, and what it means for momentum)

_EMA12 (12-period Exponential Moving Average):_ (explain value, direction, and what it means for trend)

_EMA26 (26-period Exponential Moving Average):_ (explain value, direction, and what it means for trend)

_BB High (Bollinger Band High):_ {get_value(latest, 'bb_high')}
_BB Low (Bollinger Band Low):_ {get_value(latest, 'bb_low')}

(Explain what the Bollinger Bands are showing: breakout, breakdown, consolidation, etc.)

_Analysis and Recommendations:_

Based on these indicators, here are my recommendations:

_Entry:_ (buy/sell/hold, at what price, and why)
_Stop Loss:_ (where to set it and why)
_Take Profit:_ (where to set it and why)
_Trend Prediction:_ (what to expect next, and what would change your mind)

_Beginner-Friendly Explanation:_
(Use an analogy or simple story to explain the current market situation and what to do.)

**Output Format (CRITICAL FOR SIMPLIFIED VIEW):**

First, present your detailed market analysis as a natural language response. This should be professional and easy to understand.

Then, at the **very end** of your response, you **MUST** output a summary block in this **exact plain text JSON format**. This block is for automated parsing to provide a simplified view.

```json
{{
  "simplified_summary": {{
    "current_price": [current price as float],
    "overall_sentiment": "[e.g., Bullish, Bearish, Neutral, Volatile, Mixed]",
    "primary_trend": "[e.g., Short-term Bullish, Long-term Bearish, Consolidating]",
    "entry_price": [entry price as float],
    "stop_loss": [stop loss as float],
    "take_profit": [take profit as float],
    "expected_loss": [expected loss as float],
    "simple_trend": "[e.g., Uptrend, Downtrend, Sideways]"
  }}
}}
```

**Example of Expected `simplified_summary` JSON:**

```json
{{
  "simplified_summary": {{
    "current_price": 1935.50,
    "overall_sentiment": "Mixed",
    "primary_trend": "Short-term Bullish",
    "entry_price": 1935.50,
    "stop_loss": 1920.00,
    "take_profit": 1960.00,
    "expected_loss": 15.50,
    "simple_trend": "Uptrend"
  }}
}}
```

Ensure your analysis is thorough, your insights are clear, and the final summary adheres strictly to the specified JSON format for seamless integration with the web application's simplified view.
"""

def extract_simplified_summary(analysis):
    """
    Extracts the simplified summary JSON from the AI analysis.
    Returns a dictionary or None if not found/invalid.
    """
    # Regex to find the JSON block
    json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', analysis, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1)
            summary_data = json.loads(json_str)
            if 'simplified_summary' in summary_data and isinstance(summary_data['simplified_summary'], dict):
                logger.info(f"Successfully extracted simplified summary: {summary_data['simplified_summary']}")
                # Ensure all expected fields are present (set to None if missing)
                fields = [
                    'current_price', 'overall_sentiment', 'primary_trend',
                    'entry_price', 'stop_loss', 'take_profit', 'expected_loss', 'simple_trend'
                ]
                for f in fields:
                    if f not in summary_data['simplified_summary']:
                        summary_data['simplified_summary'][f] = None
                return summary_data['simplified_summary']
            else:
                logger.warning("JSON found but 'simplified_summary' key is missing or not a dict.")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding simplified summary JSON: {e}")
            return None
    else:
        logger.warning("Simplified summary JSON block not found in AI analysis.")
        return None


def run_analysis(symbol):
    """
    Runs market analysis and gets AI recommendation.
    Returns analysis text and simplified_summary data.
    """
    try:
        # Fetch price data for the specified symbol
        df = fetch_price(symbol=symbol)
        
        # Add technical indicators to the DataFrame
        df = add_indicators(df)
        
        # Generate and save the market chart
        generate_chart(df, symbol=symbol)
        
        # Generate the prompt for the AI model
        prompt = generate_prompt(df, symbol=symbol)

        # Get AI analysis from Groq
        completion = client.chat.completions.create(
            model="llama3-8b-8192", # Using llama3-8b-8192 as specified in previous context
            messages=[{"role": "user", "content": prompt}]
        )
        analysis = completion.choices[0].message.content
        logger.info("AI Analysis received.")
        
        # Extract the structured simplified summary from the AI's full analysis
        simplified_summary_data = extract_simplified_summary(analysis)

        # Calculate/overwrite expected_loss and expected_profit
        if simplified_summary_data:
            entry = simplified_summary_data.get('entry_price')
            stop = simplified_summary_data.get('stop_loss')
            take = simplified_summary_data.get('take_profit')
            # Only calculate if all values are present and are numbers
            try:
                entry_f = float(entry) if entry is not None else None
                stop_f = float(stop) if stop is not None else None
                take_f = float(take) if take is not None else None
                if entry_f is not None and stop_f is not None:
                    simplified_summary_data['expected_loss'] = abs(entry_f - stop_f)
                if entry_f is not None and take_f is not None:
                    simplified_summary_data['expected_profit'] = abs(take_f - entry_f)
            except Exception as e:
                logger.warning(f"Could not calculate expected loss/profit: {e}")
            if 'expected_profit' not in simplified_summary_data:
                simplified_summary_data['expected_profit'] = None
        
        # If simplified_summary_data could not be extracted, try to create a basic one
        if not simplified_summary_data and not df.empty:
            latest_price = float(df.iloc[-1]['close'])
            conditions = analyze_market_conditions(df)
            simplified_summary_data = {
                'current_price': latest_price,
                'overall_sentiment': conditions['trend'].get('long_term', 'N/A'), # Fallback to long-term trend
                'primary_trend': conditions['trend'].get('short_term', 'N/A') # Fallback to short-term trend
            }
            logger.warning("Could not extract simplified summary from AI, generated a fallback.")

        return analysis, simplified_summary_data

    except Exception as e:
        logger.error(f"Overall error in run_analysis for {symbol}: {e}")
        # Try to provide a fallback summary if possible
        try:
            # Attempt to fetch price data for fallback
            df = fetch_price(symbol=symbol)
            if not df.empty:
                latest_price = float(df.iloc[-1]['close'])
                fallback_summary = {
                    'current_price': latest_price,
                    'overall_sentiment': 'Unknown',
                    'primary_trend': 'Unknown'
                }
                logger.warning(f"Returning fallback summary for {symbol} due to error: {e}")
                return f"An error occurred during market analysis for {symbol}: {str(e)}", fallback_summary
        except Exception as fallback_e:
            logger.error(f"Fallback also failed: {fallback_e}")
        # If all else fails, return the error and no summary
        return f"An error occurred during market analysis for {symbol}: {str(e)}", None
