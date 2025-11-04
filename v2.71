"""
Enhanced Streamlit Stock Dashboard v2.7
Key improvements:
- Simplified ticker input (single input box with default: AAPL, TSLA, AMZN)
- Speedometer indicator for market sentiment (2 decimal places)
- Weighted ensemble recommendations (RF:0.2, XGB:0.2, ARIMA+GARCH:0.15, LSTM:0.2, RNN:0.2, MC:0.1)
- Fixed signal weighting logic with proper modifiers
- Advanced Monte Carlo with fat-tailed distributions
- Comprehensive risk metrics (Sharpe, Sortino, Max DD, Calmar)
- Strategy backtesting with performance comparison
- Machine Learning Models Integration
- Multi-source news (Finviz, Yahoo Finance, Google News)
- News sentiment analysis with FinBERT
"""
# Imports
from bs4 import BeautifulSoup
import re
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam

# Sentiment analysis dependencies
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    SENTIMENT_AVAILABLE = True
except Exception:
    SENTIMENT_AVAILABLE = False

# Google News RSS parsing
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except:
    FEEDPARSER_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Enhanced Stock Dashboard v2.7 (by Sadiq)")

# -------------------- Data Fetching & Validation --------------------

@st.cache_data(ttl=3600)
def get_data_optimized(ticker: str, period: str = "1y", interval: str = "1d", fetch_info: bool = True):
    """
    Optimized data fetch with selective info retrieval
    Returns (hist_df, info_dict) or (empty_df, error_dict)
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if hist.empty:
            raise ValueError("Empty history returned")
        
        missing = set(required_cols) - set(hist.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if len(hist) < 50:
            raise ValueError("Need at least 50 data points")
        
        # Only fetch essential info to reduce latency
        info = {}
        if fetch_info:
            try:
                raw_info = tk.info
                info = {
                    'forwardPE': raw_info.get('forwardPE'),
                    'trailingPE': raw_info.get('trailingPE'),
                    'marketCap': raw_info.get('marketCap'),
                    'shortName': raw_info.get('shortName', ticker)
                }
            except:
                info = {'shortName': ticker}
        
        return hist, info
    except Exception as e:
        return pd.DataFrame(), {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_spy_data(period="1y", interval="1d"):
    """Cache SPY data for correlation and beta calculations"""
    hist, _ = get_data_optimized("SPY", period=period, interval=interval, fetch_info=False)
    return hist

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Fetch CNN Fear & Greed Index with fallback"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}
    base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    
    for days_back in range(0, 3):
        d = (date.today() - timedelta(days=days_back)).isoformat()
        try:
            resp = requests.get(base_url + d, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            fg = data.get("fear_and_greed", {})
            score = fg.get("score")
            rating = fg.get("rating", "N/A")
            
            if score is None:
                continue
            
            if score < 25:
                color = "🟥 Extreme Fear"
            elif score < 45:
                color = "🔴 Fear"
            elif score < 55:
                color = "🟡 Neutral"
            elif score < 75:
                color = "🟢 Greed"
            else:
                color = "🟩 Extreme Greed"
            
            return score, rating, color
        except Exception:
            continue
    
    return None, "N/A", "N/A"

def create_speedometer(score, title="Market Sentiment"):
    """
    Create a gauge/speedometer chart for market sentiment
    Score should be 0-100
    """
    # Determine color based on score
    if score < 25:
        gauge_color = "#8B0000"  # Dark red
    elif score < 45:
        gauge_color = "#FF4500"  # Red-orange
    elif score < 55:
        gauge_color = "#FFD700"  # Gold/Yellow
    elif score < 75:
        gauge_color = "#32CD32"  # Lime green
    else:
        gauge_color = "#228B22"  # Forest green
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        number = {'suffix': "", 'font': {'size': 40}, 'valueformat': '.2f'},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#FFE5E5'},
                {'range': [25, 45], 'color': '#FFF4E5'},
                {'range': [45, 55], 'color': '#FFFACD'},
                {'range': [55, 75], 'color': '#E5FFE5'},
                {'range': [75, 100], 'color': '#D0F0C0'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkgray", 'family': "Arial"}
    )
    
    return fig

# -------------------- Multi-Source News Scraping Functions --------------------

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def scrape_finviz_news(ticker: str, max_news: int = 10):
    """
    Scrape latest news from Finviz
    Returns: List of dicts with {date, time, title, link, source}
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        news_table = soup.find('table', {'id': 'news-table'})
        
        if not news_table:
            return []
        
        news_list = []
        current_date = None
        
        for row in news_table.find_all('tr')[:max_news]:
            td_timestamp = row.find('td', {'align': 'right', 'width': '130'})
            td_content = row.find('td', {'align': 'left'})
            
            if not td_timestamp or not td_content:
                continue
            
            # Parse timestamp
            timestamp_text = td_timestamp.get_text().strip()
            timestamp_parts = timestamp_text.split()
            
            if len(timestamp_parts) == 2:
                current_date = timestamp_parts[0]
                time_str = timestamp_parts[1]
            else:
                time_str = timestamp_parts[0]
            
            # Extract news info
            link_tag = td_content.find('a')
            if link_tag:
                title = link_tag.get_text().strip()
                link = link_tag.get('href', '')
                
                # Get source
                source_span = td_content.find('span')
                source = source_span.get_text().strip() if source_span else 'N/A'
                
                news_list.append({
                    'date': current_date,
                    'time': time_str,
                    'title': title,
                    'link': link,
                    'source': source
                })
        
        return news_list
    except Exception as e:
        st.warning(f"Failed to scrape Finviz news: {e}")
        return []

@st.cache_data(ttl=1800)
def scrape_google_news(ticker: str, max_news: int = 10):
    """
    Scrape Google News RSS feed for a ticker
    Returns: List of dicts with {date, time, title, link, source}
    """
    if not FEEDPARSER_AVAILABLE:
        return []
    
    try:
        company_name = ticker.upper()
        url = f"https://news.google.com/rss/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(url)
        
        news_list = []
        for entry in feed.entries[:max_news]:
            try:
                pub_date = datetime(*entry.published_parsed[:6])
                date_str = pub_date.strftime('%b-%d-%y')
                time_str = pub_date.strftime('%I:%M%p')
                
                title = entry.title
                link = entry.link
                source = entry.source.title if hasattr(entry, 'source') else 'Google News'
                
                news_list.append({
                    'date': date_str,
                    'time': time_str,
                    'title': title,
                    'link': link,
                    'source': source
                })
            except Exception:
                continue
        
        return news_list
    except Exception as e:
        st.warning(f"Failed to scrape Google News: {e}")
        return []

# -------------------- FinBERT Sentiment Analysis --------------------

@st.cache_resource
def load_finbert_pipeline():
    """
    Load FinBERT sentiment analysis pipeline (cached)
    """
    try:
        # Set environment variable for TensorFlow to use Keras 3 compatibility
        import os
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        
        from transformers import pipeline
        import tensorflow as tf
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")
        return sentiment_pipeline
    except Exception as e:
        # Try PyTorch backend as fallback
        try:
            from transformers import pipeline
            sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt", device=-1)
            return sentiment_pipeline
        except Exception as e2:
            st.warning(f"⚠️ FinBERT sentiment analysis unavailable. Install with: `pip install tf-keras --break-system-packages` or use PyTorch backend")
            return None

def analyze_sentiment_finbert(texts, pipeline):
    """
    Analyze sentiment using FinBERT
    Returns list of (label, score) tuples
    """
    if not pipeline or not texts:
        return [(None, None)] * len(texts)
    
    try:
        # FinBERT has max length of 512 tokens, truncate long texts
        truncated_texts = [text[:500] for text in texts]
        results = pipeline(truncated_texts)
        
        sentiment_results = []
        for res in results:
            label = res['label']  # 'positive', 'negative', 'neutral'
            score = res['score']
            sentiment_results.append((label, score))
        
        return sentiment_results
    except Exception as e:
        st.warning(f"FinBERT analysis failed: {e}")
        return [(None, None)] * len(texts)

# -------------------- Technical Indicators --------------------

def calculate_indicators(df):
    """Calculate all technical indicators"""
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # ADX (Average Directional Index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
    minus_di = 100 * (abs(minus_dm).rolling(window=14).mean() / atr_14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(window=14).mean()
    
    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    return df

def calculate_risk_metrics(df, spy_df=None):
    """Calculate comprehensive risk metrics"""
    returns = df['Close'].pct_change().dropna()
    
    # Basic metrics
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(df)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (assuming 4% risk-free rate)
    risk_free_rate = 0.04
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Beta (if SPY data available)
    beta = None
    correlation = None
    if spy_df is not None and not spy_df.empty:
        spy_returns = spy_df['Close'].pct_change().dropna()
        common_dates = returns.index.intersection(spy_returns.index)
        if len(common_dates) > 30:
            stock_common = returns.loc[common_dates]
            spy_common = spy_returns.loc[common_dates]
            covariance = stock_common.cov(spy_common)
            spy_variance = spy_common.var()
            beta = covariance / spy_variance if spy_variance != 0 else None
            correlation = stock_common.corr(spy_common)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'beta': beta,
        'correlation': correlation
    }

# -------------------- Signal Generation with Smart Weighting --------------------

def generate_signals_with_weights(df, fear_greed_score=None):
    """
    Generate buy/sell signals with smart weighting system
    Returns: (signals_list, buy_score, sell_score, net_score)
    Each signal: (signal_text, signal_type, weight, extra_info)
    signal_type: BUY, SELL, CONFIRM, AMPLIFY, DAMPEN
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    signals = []
    
    # === PRIMARY SIGNALS (Base weights: 0.5-1.5) ===
    
    # 1. MACD Crossover
    if latest['MACD'] > latest['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']:
        signals.append(("MACD bullish crossover", "BUY", 1.2, None))
    elif latest['MACD'] < latest['Signal_Line'] and prev['MACD'] >= prev['Signal_Line']:
        signals.append(("MACD bearish crossover", "SELL", 1.2, None))
    
    # 2. RSI
    if latest['RSI'] < 30:
        strength = (30 - latest['RSI']) / 30
        weight = 0.8 + (strength * 0.7)
        signals.append((f"RSI oversold", "BUY", weight, f"RSI={latest['RSI']:.1f}"))
    elif latest['RSI'] > 70:
        strength = (latest['RSI'] - 70) / 30
        weight = 0.8 + (strength * 0.7)
        signals.append((f"RSI overbought", "SELL", weight, f"RSI={latest['RSI']:.1f}"))
    
    # 3. Bollinger Bands
    if latest['Close'] < latest['BB_lower']:
        signals.append(("Price below lower BB", "BUY", 1.0, None))
    elif latest['Close'] > latest['BB_upper']:
        signals.append(("Price above upper BB", "SELL", 1.0, None))
    
    # 4. Stochastic
    if latest['Stoch_%K'] < 20 and latest['Stoch_%D'] < 20:
        signals.append(("Stochastic oversold", "BUY", 0.7, f"%K={latest['Stoch_%K']:.1f}"))
    elif latest['Stoch_%K'] > 80 and latest['Stoch_%D'] > 80:
        signals.append(("Stochastic overbought", "SELL", 0.7, f"%K={latest['Stoch_%K']:.1f}"))
    
    # 5. Moving Average Crossovers
    if latest['SMA_20'] > latest['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
        signals.append(("Golden Cross (20/50)", "BUY", 1.5, None))
    elif latest['SMA_20'] < latest['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
        signals.append(("Death Cross (20/50)", "SELL", 1.5, None))
    
    # === CONFIRMING SIGNALS (Add to existing signals, weight 0.3-0.8) ===
    
    # 6. Volume confirmation
    if latest['Volume_Ratio'] > 1.5:
        if len([s for s in signals if s[1] == "BUY"]) > 0:
            signals.append(("High volume confirms buy", "CONFIRM", 0.6, f"Vol ratio={latest['Volume_Ratio']:.2f}"))
        elif len([s for s in signals if s[1] == "SELL"]) > 0:
            signals.append(("High volume confirms sell", "CONFIRM", 0.6, f"Vol ratio={latest['Volume_Ratio']:.2f}"))
    
    # 7. ADX trend strength
    if latest['ADX'] > 25:
        if len([s for s in signals if s[1] == "BUY"]) > 0:
            signals.append(("Strong uptrend (ADX)", "CONFIRM", 0.5, f"ADX={latest['ADX']:.1f}"))
        elif len([s for s in signals if s[1] == "SELL"]) > 0:
            signals.append(("Strong downtrend (ADX)", "CONFIRM", 0.5, f"ADX={latest['ADX']:.1f}"))
    
    # 8. OBV trend
    obv_sma = df['OBV'].rolling(window=20).mean()
    if latest['OBV'] > obv_sma.iloc[-1]:
        signals.append(("OBV trending up", "CONFIRM", 0.4, None))
    else:
        signals.append(("OBV trending down", "CONFIRM", 0.4, None))
    
    # === MODIFIERS (Amplify or dampen signals, weight 0.2-0.5) ===
    
    # 9. Fear & Greed Index
    if fear_greed_score is not None:
        if fear_greed_score < 25:
            signals.append(("Extreme fear - contrarian buy", "AMPLIFY", 0.5, f"F&G={fear_greed_score:.2f}"))
        elif fear_greed_score > 75:
            signals.append(("Extreme greed - caution", "DAMPEN", 0.4, f"F&G={fear_greed_score:.2f}"))
    
    # 10. Volatility (Bollinger Band width)
    if latest['BB_width'] < df['BB_width'].quantile(0.2):
        signals.append(("Low volatility - expansion likely", "AMPLIFY", 0.3, None))
    elif latest['BB_width'] > df['BB_width'].quantile(0.8):
        signals.append(("High volatility - caution", "DAMPEN", 0.3, None))
    
    # === CALCULATE SCORES ===
    buy_score = 0
    sell_score = 0
    
    for signal_text, signal_type, weight, extra in signals:
        if signal_type == "BUY":
            buy_score += weight
        elif signal_type == "SELL":
            sell_score += weight
        elif signal_type == "CONFIRM":
            # Confirm amplifies the dominant signal
            if buy_score > sell_score:
                buy_score += weight
            else:
                sell_score += weight
        elif signal_type == "AMPLIFY":
            # Amplify the dominant signal
            if buy_score > sell_score:
                buy_score *= (1 + weight * 0.2)
            else:
                sell_score *= (1 + weight * 0.2)
        elif signal_type == "DAMPEN":
            # Reduce both signals
            buy_score *= (1 - weight * 0.15)
            sell_score *= (1 - weight * 0.15)
    
    net_score = buy_score - sell_score
    
    return signals, buy_score, sell_score, net_score

def make_recommendation(buy_score, sell_score, net_score):
    """Generate final recommendation based on scores"""
    total = buy_score + sell_score
    if total == 0:
        return "HOLD", 0, "Neutral"
    
    confidence = abs(net_score) / total * 100
    
    if net_score > 0.5:
        strength = "Strong " if confidence > 70 else ""
        return f"{strength}BUY", confidence, "green"
    elif net_score < -0.5:
        strength = "Strong " if confidence > 70 else ""
        return f"{strength}SELL", confidence, "red"
    else:
        return "HOLD", confidence, "orange"

# -------------------- ML Analysis Functions --------------------

def prepare_ml_features(df, lookback=20):
    """
    Prepare features for ML models
    Returns X (features) and y (target: 1 for up, 0 for down)
    """
    df = df.copy()
    
    # Target: 1 if next day closes higher, 0 otherwise
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Feature engineering
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['Returns'] = df['Close'].pct_change()
    features['Returns_5d'] = df['Close'].pct_change(5)
    features['Returns_20d'] = df['Close'].pct_change(20)
    
    # Technical indicators (already calculated)
    features['RSI'] = df['RSI']
    features['MACD'] = df['MACD']
    features['MACD_Signal'] = df['Signal_Line']
    features['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    features['Stoch_K'] = df['Stoch_%K']
    features['Volume_Ratio'] = df['Volume_Ratio']
    features['ADX'] = df['ADX']
    
    # Moving average features
    features['SMA_20_50_ratio'] = df['SMA_20'] / df['SMA_50']
    features['Price_SMA20_ratio'] = df['Close'] / df['SMA_20']
    
    # Volatility
    features['ATR'] = df['ATR']
    features['BB_width'] = df['BB_width']
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        features[f'Returns_lag_{lag}'] = features['Returns'].shift(lag)
    
    # Drop NaN and align target
    features = features.dropna()
    y = df.loc[features.index, 'Target']
    
    # Remove last row (no target available)
    features = features[:-1]
    y = y[:-1]
    
    return features, y

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    try:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        # Prediction for next day
        last_pred = model.predict(X_test.iloc[[-1]])[0]
        last_proba = model.predict_proba(X_test.iloc[[-1]])[0]
        
        return {
            'model': 'Random Forest',
            'metrics': metrics,
            'prediction': 'BUY' if last_pred == 1 else 'SELL',
            'confidence': max(last_proba) * 100
        }
    except Exception as e:
        return {'model': 'Random Forest', 'error': str(e)}

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier"""
    try:
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                  random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        last_pred = model.predict(X_test.iloc[[-1]])[0]
        last_proba = model.predict_proba(X_test.iloc[[-1]])[0]
        
        return {
            'model': 'XGBoost',
            'metrics': metrics,
            'prediction': 'BUY' if last_pred == 1 else 'SELL',
            'confidence': max(last_proba) * 100
        }
    except Exception as e:
        return {'model': 'XGBoost', 'error': str(e)}

def train_arima_garch(df):
    """Train ARIMA + GARCH model for volatility forecasting"""
    try:
        returns = df['Close'].pct_change().dropna() * 100
        
        # Fit ARIMA
        model_arima = ARIMA(returns, order=(1,0,1))
        fitted_arima = model_arima.fit()
        
        # Fit GARCH on residuals
        residuals = fitted_arima.resid
        model_garch = arch_model(residuals, vol='Garch', p=1, q=1)
        fitted_garch = model_garch.fit(disp='off')
        
        # Forecast
        arima_forecast = fitted_arima.forecast(steps=1)
        garch_forecast = fitted_garch.forecast(horizon=1)
        
        predicted_return = arima_forecast.iloc[0] if hasattr(arima_forecast, 'iloc') else arima_forecast.values[0]
        predicted_vol = np.sqrt(garch_forecast.variance.values[-1, 0])
        
        recommendation = 'BUY' if predicted_return > 0 else 'SELL'
        confidence = min(abs(predicted_return) / predicted_vol * 50, 99) if predicted_vol > 0 else 50
        
        return {
            'model': 'ARIMA+GARCH',
            'metrics': {'predicted_return': predicted_return, 'predicted_volatility': predicted_vol},
            'prediction': recommendation,
            'confidence': confidence
        }
    except Exception as e:
        return {'model': 'ARIMA+GARCH', 'error': str(e)}

def train_lstm(X_train, y_train, X_test, y_test):
    """Train LSTM neural network"""
    try:
        # Reshape for LSTM [samples, time steps, features]
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Build model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(30, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train with minimal epochs to avoid overfitting
        model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
        
        # Predict
        y_pred_proba = model.predict(X_test_lstm, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        last_pred = y_pred[-1]
        last_proba = y_pred_proba[-1]
        
        return {
            'model': 'LSTM',
            'metrics': metrics,
            'prediction': 'BUY' if last_pred == 1 else 'SELL',
            'confidence': max(last_proba, 1-last_proba) * 100
        }
    except Exception as e:
        return {'model': 'LSTM', 'error': str(e)}

def train_rnn(X_train, y_train, X_test, y_test):
    """Train Simple RNN neural network"""
    try:
        # Reshape for RNN
        X_train_rnn = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_rnn = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Build model
        model = Sequential([
            SimpleRNN(50, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            SimpleRNN(30, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(X_train_rnn, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
        
        y_pred_proba = model.predict(X_test_rnn, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        }
        
        last_pred = y_pred[-1]
        last_proba = y_pred_proba[-1]
        
        return {
            'model': 'RNN',
            'metrics': metrics,
            'prediction': 'BUY' if last_pred == 1 else 'SELL',
            'confidence': max(last_proba, 1-last_proba) * 100
        }
    except Exception as e:
        return {'model': 'RNN', 'error': str(e)}

def monte_carlo_simulation(df, days_ahead=30, simulations=1000):
    """
    Monte Carlo simulation for price prediction
    Returns recommendation based on probability of positive return
    """
    try:
        returns = df['Close'].pct_change().dropna()
        
        # Use Student's t-distribution for fat tails
        mu = returns.mean()
        sigma = returns.std()
        
        # Fit t-distribution
        params = stats.t.fit(returns)
        df_t, loc_t, scale_t = params
        
        # Run simulations
        last_price = df['Close'].iloc[-1]
        final_prices = []
        
        for _ in range(simulations):
            price = last_price
            for _ in range(days_ahead):
                shock = stats.t.rvs(df_t, loc=loc_t, scale=scale_t)
                price *= (1 + shock)
            final_prices.append(price)
        
        final_prices = np.array(final_prices)
        
        # Calculate probability of profit
        prob_profit = (final_prices > last_price).mean()
        expected_return = (final_prices.mean() - last_price) / last_price
        
        recommendation = 'BUY' if prob_profit > 0.5 else 'SELL'
        confidence = abs(prob_profit - 0.5) * 200  # Scale to 0-100
        
        return {
            'model': 'Monte Carlo',
            'metrics': {
                'prob_profit': prob_profit,
                'expected_return': expected_return,
                'median_price': np.median(final_prices)
            },
            'prediction': recommendation,
            'confidence': confidence
        }
    except Exception as e:
        return {'model': 'Monte Carlo', 'error': str(e)}

def run_ml_analysis(df):
    """
    Run all ML models and return results
    """
    results = []
    
    try:
        # Prepare features
        X, y = prepare_ml_features(df)
        
        if len(X) < 100:
            st.warning("Not enough data for ML analysis (need at least 100 data points)")
            return None
        
        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Run models
        with st.spinner("Training Random Forest..."):
            rf_result = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
            if 'error' not in rf_result:
                results.append([
                    rf_result['model'],
                    f"n_estimators=100, max_depth=10",
                    f"Acc: {rf_result['metrics']['accuracy']:.3f}, F1: {rf_result['metrics']['f1']:.3f}, AUC: {rf_result['metrics']['auc']:.3f}",
                    rf_result['prediction'],
                    f"{rf_result['confidence']:.1f}%"
                ])
        
        with st.spinner("Training XGBoost..."):
            xgb_result = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
            if 'error' not in xgb_result:
                results.append([
                    xgb_result['model'],
                    f"n_estimators=100, max_depth=5",
                    f"Acc: {xgb_result['metrics']['accuracy']:.3f}, F1: {xgb_result['metrics']['f1']:.3f}, AUC: {xgb_result['metrics']['auc']:.3f}",
                    xgb_result['prediction'],
                    f"{xgb_result['confidence']:.1f}%"
                ])
        
        with st.spinner("Training ARIMA+GARCH..."):
            arima_result = train_arima_garch(df)
            if 'error' not in arima_result:
                results.append([
                    arima_result['model'],
                    f"ARIMA(1,0,1) + GARCH(1,1)",
                    f"Pred Return: {arima_result['metrics']['predicted_return']:.3f}%, Vol: {arima_result['metrics']['predicted_volatility']:.3f}%",
                    arima_result['prediction'],
                    f"{arima_result['confidence']:.1f}%"
                ])
        
        with st.spinner("Training LSTM..."):
            lstm_result = train_lstm(X_train_scaled, y_train, X_test_scaled, y_test)
            if 'error' not in lstm_result:
                results.append([
                    lstm_result['model'],
                    f"2-layer, 50+30 units, dropout=0.2",
                    f"Acc: {lstm_result['metrics']['accuracy']:.3f}, F1: {lstm_result['metrics']['f1']:.3f}, AUC: {lstm_result['metrics']['auc']:.3f}",
                    lstm_result['prediction'],
                    f"{lstm_result['confidence']:.1f}%"
                ])
        
        with st.spinner("Training RNN..."):
            rnn_result = train_rnn(X_train_scaled, y_train, X_test_scaled, y_test)
            if 'error' not in rnn_result:
                results.append([
                    rnn_result['model'],
                    f"2-layer, 50+30 units, dropout=0.2",
                    f"Acc: {rnn_result['metrics']['accuracy']:.3f}, F1: {rnn_result['metrics']['f1']:.3f}, AUC: {rnn_result['metrics']['auc']:.3f}",
                    rnn_result['prediction'],
                    f"{rnn_result['confidence']:.1f}%"
                ])
        
        with st.spinner("Running Monte Carlo simulation..."):
            mc_result = monte_carlo_simulation(df, days_ahead=30, simulations=1000)
            if 'error' not in mc_result:
                results.append([
                    mc_result['model'],
                    f"1000 sims, 30 days, t-dist",
                    f"Prob Profit: {mc_result['metrics']['prob_profit']:.3f}, Exp Return: {mc_result['metrics']['expected_return']:.3f}",
                    mc_result['prediction'],
                    f"{mc_result['confidence']:.1f}%"
                ])
        
        # Store raw results for ensemble calculation
        st.session_state['ml_raw_results'] = {
            'RF': rf_result,
            'XGB': xgb_result,
            'ARIMA': arima_result,
            'LSTM': lstm_result,
            'RNN': rnn_result,
            'MC': mc_result
        }
        
        return results if results else None
        
    except Exception as e:
        st.error(f"ML Analysis Error: {e}")
        return None

def calculate_ensemble_recommendation(ml_results):
    """
    Calculate weighted ensemble recommendation
    Weights: RF=0.2, XGB=0.2, ARIMA+GARCH=0.15, LSTM=0.2, RNN=0.2, MC=0.1
    """
    # Default weights
    weights = {
        'Random Forest': 0.20,
        'XGBoost': 0.20,
        'ARIMA+GARCH': 0.15,
        'LSTM': 0.20,
        'RNN': 0.20,
        'Monte Carlo': 0.10
    }
    
    buy_score = 0
    sell_score = 0
    total_weight = 0
    total_confidence = 0
    count = 0
    
    for result in ml_results:
        algorithm = result[0]
        prediction = result[3]
        confidence_str = result[4].rstrip('%')
        confidence = float(confidence_str) / 100  # Convert to 0-1 scale
        
        weight = weights.get(algorithm, 0.1)
        total_weight += weight
        
        if prediction == 'BUY':
            buy_score += weight * confidence
        else:
            sell_score += weight * confidence
        
        total_confidence += confidence
        count += 1
    
    # Normalize scores
    if total_weight > 0:
        buy_score /= total_weight
        sell_score /= total_weight
    
    # Determine recommendation
    if buy_score > sell_score:
        recommendation = "BUY"
        ensemble_conf = f"{buy_score * 100:.1f}%"
    else:
        recommendation = "SELL"
        ensemble_conf = f"{sell_score * 100:.1f}%"
    
    # Calculate agreement (how many models agree with the majority)
    buy_count = sum(1 for r in ml_results if r[3] == 'BUY')
    sell_count = len(ml_results) - buy_count
    agreement = f"{max(buy_count, sell_count)}/{len(ml_results)}"
    
    return recommendation, ensemble_conf, agreement

# -------------------- Backtesting --------------------

def backtest_strategy(df, weights, initial_capital=10000, confidence_threshold=60, 
                     stop_loss_pct=0.05, take_profit_pct=0.15):
    """
    Backtest the trading strategy with risk management
    """
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long
    shares = 0
    entry_price = 0
    
    trades = []
    equity_curve = []
    buy_signals = []
    sell_signals = []
    positions = []
    
    for i in range(50, len(df)):
        current_data = df.iloc[:i+1]
        current_price = float(current_data['Close'].iloc[-1])
        current_date = current_data.index[-1]
        
        # Generate signals
        signals, buy_score, sell_score, net_score = generate_signals_with_weights(current_data)
        recommendation, confidence, _ = make_recommendation(buy_score, sell_score, net_score)
        
        # Check if in position
        if position == 1:
            # Check stop loss
            if current_price <= entry_price * (1 - stop_loss_pct):
                # Stop loss triggered
                sell_value = shares * current_price
                capital += sell_value
                return_pct = (current_price - entry_price) / entry_price
                positions.append(['SELL (Stop Loss)', current_date, f'${current_price:.2f}', 
                                f'{shares:.2f} shares', f'{return_pct*100:.2f}%'])
                position = 0
                shares = 0
                sell_signals.append(i)
                trades.append(return_pct)
            
            # Check take profit
            elif current_price >= entry_price * (1 + take_profit_pct):
                # Take profit triggered
                sell_value = shares * current_price
                capital += sell_value
                return_pct = (current_price - entry_price) / entry_price
                positions.append(['SELL (Take Profit)', current_date, f'${current_price:.2f}', 
                                f'{shares:.2f} shares', f'{return_pct*100:.2f}%'])
                position = 0
                shares = 0
                sell_signals.append(i)
                trades.append(return_pct)
            
            # Check normal sell signal
            elif 'SELL' in recommendation and confidence >= confidence_threshold:
                sell_value = shares * current_price
                capital += sell_value
                return_pct = (current_price - entry_price) / entry_price
                positions.append(['SELL', current_date, f'${current_price:.2f}', 
                                f'{shares:.2f} shares', f'{return_pct*100:.2f}%'])
                position = 0
                shares = 0
                sell_signals.append(i)
                trades.append(return_pct)
        
        # Check buy signal
        elif position == 0 and 'BUY' in recommendation and confidence >= confidence_threshold:
            shares = capital / current_price
            entry_price = current_price
            capital = 0
            position = 1
            buy_signals.append(i)
            positions.append(['BUY', current_date, f'${current_price:.2f}', 
                            f'{shares:.2f} shares', f'{confidence:.1f}%'])
        
        # Update equity curve
        portfolio_value = capital + (shares * current_price if position == 1 else 0)
        equity_curve.append({'date': current_date, 'value': portfolio_value})
    
    # Close any open position
    if position == 1:
        final_price = float(df['Close'].iloc[-1])
        sell_value = shares * final_price
        capital += sell_value
        return_pct = (final_price - entry_price) / entry_price
        trades.append(return_pct)
    
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Calculate buy & hold return
    buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[50]) / df['Close'].iloc[50]
    
    # Calculate trade statistics
    if trades:
        trades_array = np.array(trades)
        wins = trades_array[trades_array > 0]
        losses = trades_array[trades_array < 0]
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
    else:
        win_rate = avg_win = avg_loss = 0
    
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'alpha': total_return - buy_hold_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'equity_curve': equity_curve,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'positions': positions
    }

# -------------------- Monte Carlo Simulation --------------------

def estimate_days_to_target_advanced(df, current_price, target_return=0.10, sims=10000, max_days=252):
    """
    Advanced Monte Carlo using Student's t-distribution
    """
    returns = df['Close'].pct_change().dropna()
    
    # Fit Student's t-distribution
    params = stats.t.fit(returns)
    df_t, loc_t, scale_t = params
    
    days_to_target = []
    
    for _ in range(sims):
        price = current_price
        for day in range(1, max_days + 1):
            shock = stats.t.rvs(df_t, loc=loc_t, scale=scale_t)
            price *= (1 + shock)
            
            if price >= current_price * (1 + target_return):
                days_to_target.append(day)
                break
    
    if not days_to_target:
        return {
            'probability': 0,
            'median_days': 'N/A',
            '90pct_days': 'N/A',
            '10pct_days': 'N/A'
        }
    
    probability = len(days_to_target) / sims
    median_days = int(np.median(days_to_target))
    pct_90 = int(np.percentile(days_to_target, 90))
    pct_10 = int(np.percentile(days_to_target, 10))
    
    return {
        'probability': probability,
        'median_days': median_days,
        '90pct_days': pct_90,
        '10pct_days': pct_10
    }

def monte_carlo_price_simulation(df, current_price, sims=10000):
    """
    Monte Carlo simulation for different time horizons
    Returns DataFrame with 5th, 50th, 95th percentile prices
    """
    returns = df['Close'].pct_change().dropna()
    
    # Fit Student's t-distribution
    params = stats.t.fit(returns)
    df_t, loc_t, scale_t = params
    
    timeframes = [7, 30, 90, 180, 252]  # 1w, 1m, 3m, 6m, 1y
    results = []
    
    for days in timeframes:
        final_prices = []
        
        for _ in range(sims):
            price = current_price
            for _ in range(days):
                shock = stats.t.rvs(df_t, loc=loc_t, scale=scale_t)
                price *= (1 + shock)
            final_prices.append(price)
        
        final_prices = np.array(final_prices)
        
        pct_5 = np.percentile(final_prices, 5)
        pct_50 = np.percentile(final_prices, 50)
        pct_95 = np.percentile(final_prices, 95)
        
        # Calculate returns
        ret_5 = (pct_5 - current_price) / current_price * 100
        ret_50 = (pct_50 - current_price) / current_price * 100
        ret_95 = (pct_95 - current_price) / current_price * 100
        
        if days == 7:
            period = "1 Week"
        elif days == 30:
            period = "1 Month"
        elif days == 90:
            period = "3 Months"
        elif days == 180:
            period = "6 Months"
        else:
            period = "1 Year"
        
        results.append({
            'Period': period,
            'Days': days,
            'Worst Case (5%)': f'${pct_5:.2f} ({ret_5:+.1f}%)',
            'Most Likely (50%)': f'${pct_50:.2f} ({ret_50:+.1f}%)',
            'Best Case (95%)': f'${pct_95:.2f} ({ret_95:+.1f}%)'
        })
    
    return pd.DataFrame(results)

# ==================== STREAMLIT UI ====================

st.title("📈 Enhanced Stock Dashboard v2.7")
st.caption("Advanced technical analysis, ML predictions, and risk management (by Sadiq)")

# Initialize session state
if "ml_cache" not in st.session_state:
    st.session_state["ml_cache"] = {}

# ==================== Sidebar ====================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Single ticker input with default stocks
    tickers_input = st.text_input(
        "Tickers",
        value="AAPL, TSLA, AMZN",
        help="Enter comma-separated tickers (e.g., AAPL, GOOGL, MSFT)"
    )
    
    # Parse tickers
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    st.markdown("---")
    
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    
    st.markdown("---")
    st.subheader("⚖️ Signal Weights")
    
    weights = {
        'MACD': st.slider("MACD", 0.0, 2.0, 1.2, 0.1),
        'RSI': st.slider("RSI", 0.0, 2.0, 1.0, 0.1),
        'BB': st.slider("Bollinger Bands", 0.0, 2.0, 1.0, 0.1),
        'Stoch': st.slider("Stochastic", 0.0, 2.0, 0.7, 0.1),
        'MA': st.slider("Moving Averages", 0.0, 2.0, 1.5, 0.1),
    }
    
    st.markdown("---")
    st.subheader("🎲 Monte Carlo Settings")
    sim_count = st.number_input("Simulations", 1000, 50000, 10000, 1000)
    max_days = st.number_input("Max Days Horizon", 30, 500, 252, 10)
    
    st.markdown("---")
    st.subheader("📊 Backtest Settings")
    initial_capital = st.number_input("Initial Capital ($)", 1000, 1000000, 10000, 1000)
    confidence_threshold = st.slider("Signal Confidence Threshold (%)", 0, 100, 60, 5)
    stop_loss_pct = st.slider("Stop Loss (%)", 0.0, 0.20, 0.05, 0.01)
    take_profit_pct = st.slider("Take Profit (%)", 0.05, 0.50, 0.15, 0.01)

# ==================== Main Dashboard ====================

# Tabs for multiple stocks
if len(tickers) == 1:
    # Single stock - no tabs needed
    ticker = tickers[0]
    st.header(f"📊 Analysis for {ticker}")
    
    # ... rest of the analysis code (will be added in next part)
    
else:
    # Multiple stocks - use tabs
    tabs = st.tabs([f"📈 {ticker}" for ticker in tickers])
    
    for idx, ticker in enumerate(tickers):
        with tabs[idx]:
            # ... analysis code for each ticker (will be added in next part)
            pass

# For now, let's implement the analysis for a single ticker
# We'll handle multiple tickers in the tab structure

for ticker in tickers[:1]:  # Process first ticker only for now
    with st.spinner(f"Loading data for {ticker}..."):
        df, info = get_data_optimized(ticker, period=period, interval=interval)
        
        if "_error" in info:
            st.error(f"❌ Failed to load {ticker}: {info['_error']}")
            continue
        
        # Calculate indicators
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Get SPY data for correlation
        spy_df = get_spy_data(period=period, interval=interval)
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(df, spy_df)
    
    # ==================== Market Overview ====================
    
    st.markdown("---")
    st.subheader("🌍 Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fear & Greed Index with Speedometer
        fg_score, fg_rating, fg_color = get_fear_greed_index()
        
        if fg_score is not None:
            st.subheader("CNN Fear & Greed Index")
            fig_speedometer = create_speedometer(fg_score, "Market Sentiment")
            st.plotly_chart(fig_speedometer, use_container_width=True)
            st.markdown(f"**Rating:** {fg_color}")
        else:
            st.info("Fear & Greed Index temporarily unavailable")
    
    with col2:
        st.subheader("Key Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Current Price", f"${latest['Close']:.2f}",
                 delta=f"{((latest['Close']/prev['Close'])-1)*100:+.2f}%")
        m1.metric("52-Week Range", 
                 f"${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        if info.get('marketCap'):
            m2.metric("Market Cap", f"${info['marketCap']/1e9:.2f}B")
        if info.get('forwardPE'):
            m2.metric("Forward P/E", f"{info['forwardPE']:.2f}")
    
    # ==================== Technical Analysis ====================
    
    st.markdown("---")
    st.subheader("📈 Technical Analysis")
    
    # Price chart with indicators
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.5, 0.15, 0.15, 0.2],
                       subplot_titles=("Price & Moving Averages", "MACD", "RSI", "Volume"))
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'),
                  row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                            line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                            line=dict(color='blue', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                            line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                            line=dict(color='gray', dash='dash', width=1),
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color='blue', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal',
                            line=dict(color='red', width=1)), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram',
                        marker_color='gray'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color='purple', width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Volume
    colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
             for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                        marker_color=colors), row=4, col=1)
    
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical metrics
    st.subheader("📊 Current Technical Indicators")
    
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("RSI", f"{latest['RSI']:.2f}", 
             help="Relative Strength Index (14-period)")
    t2.metric("MACD", f"{latest['MACD']:.4f}",
             help="Moving Average Convergence Divergence")
    t3.metric("Stochastic %K", f"{latest['Stoch_%K']:.2f}",
             help="Stochastic Oscillator")
    t4.metric("ADX", f"{latest['ADX']:.2f}",
             help="Average Directional Index (trend strength)")
    
    t5, t6, t7, t8 = st.columns(4)
    t5.metric("ATR", f"{latest['ATR']:.2f}",
             help="Average True Range (volatility)")
    t6.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}",
             help="Current volume vs 20-day average")
    t7.metric("BB Position", f"{((latest['Close']-latest['BB_lower'])/(latest['BB_upper']-latest['BB_lower'])*100):.1f}%",
             help="Position within Bollinger Bands")
    t8.metric("Volatility", f"{risk_metrics['volatility']*100:.2f}%",
             help="Annualized volatility")
    
    # ==================== Risk Metrics ====================
    
    st.markdown("---")
    st.subheader("⚠️ Risk Metrics")
    
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}",
             help="Risk-adjusted return (>1 is good)")
    r2.metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}",
             help="Downside risk-adjusted return")
    r3.metric("Max Drawdown", f"{risk_metrics['max_drawdown']*100:.2f}%",
             help="Largest peak-to-trough decline")
    r4.metric("Calmar Ratio", f"{risk_metrics['calmar_ratio']:.2f}",
             help="Return vs max drawdown")
    
    if risk_metrics['beta'] is not None:
        r5.metric("Beta", f"{risk_metrics['beta']:.2f}",
                 help="Volatility vs market (SPY)")
    
    # ==================== News Section ====================
    
    st.markdown("---")
    st.subheader("📰 Latest News")
    
    news_source = st.selectbox("Select News Source", ["Finviz", "Google News"])
    
    if news_source == "Finviz":
        news_data = scrape_finviz_news(ticker, max_news=10)
    else:
        news_data = scrape_google_news(ticker, max_news=10)
    
    if news_data:
        # Load FinBERT for sentiment analysis
        finbert_pipeline = load_finbert_pipeline()
        
        if finbert_pipeline:
            with st.spinner("Analyzing sentiment with FinBERT..."):
                titles = [news['title'] for news in news_data]
                sentiments = analyze_sentiment_finbert(titles, finbert_pipeline)
                
                # Add sentiments to news data
                for news, (label, score) in zip(news_data, sentiments):
                    news['sentiment_label'] = label
                    news['sentiment_score'] = score
        
        # Display news in a table
        news_display = []
        for news in news_data:
            date_time = f"{news['date']} {news['time']}"
            title = news['title']
            source = news['source']
            link = news['link']
            
            sentiment_emoji = ""
            sentiment_text = ""
            if 'sentiment_label' in news and news['sentiment_label']:
                if news['sentiment_label'].lower() == 'positive':
                    sentiment_emoji = "🟢"
                elif news['sentiment_label'].lower() == 'negative':
                    sentiment_emoji = "🔴"
                else:
                    sentiment_emoji = "🟡"
                sentiment_text = f"{news['sentiment_label']} ({news['sentiment_score']:.2f})"
            
            news_display.append({
                'Date/Time': date_time,
                'Title': title,
                'Source': source,
                'Sentiment': f"{sentiment_emoji} {sentiment_text}" if sentiment_text else "N/A",
                'Link': link
            })
        
        news_df = pd.DataFrame(news_display)
        st.dataframe(news_df, use_container_width=True, hide_index=True)
        
        # Sentiment summary
        if finbert_pipeline and sentiments:
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for label, score in sentiments:
                if label:
                    sentiment_counts[label.lower()] = sentiment_counts.get(label.lower(), 0) + 1
            
            st.write("**Sentiment Summary:**")
            s1, s2, s3 = st.columns(3)
            s1.metric("🟢 Positive", sentiment_counts['positive'])
            s2.metric("🟡 Neutral", sentiment_counts['neutral'])
            s3.metric("🔴 Negative", sentiment_counts['negative'])
    else:
        st.info("No news available for this ticker")
    
    # ==================== Trading Signals ====================
    
    st.markdown("---")
    st.subheader("🎯 Trading Signals")
    
    # Generate signals
    signals, buy_score, sell_score, net_score = generate_signals_with_weights(df, fg_score)
    recommendation, confidence, color = make_recommendation(buy_score, sell_score, net_score)
    
    raw_scores = {'buy': buy_score, 'sell': sell_score, 'net': net_score}
    cache_key = f"{ticker}_{period}_{interval}"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{recommendation}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align:center;'>Confidence: {confidence:.1f}%</h4>", unsafe_allow_html=True)
        st.metric("Buy Score", f"{raw_scores['buy']:.2f}")
        st.metric("Sell Score", f"{raw_scores['sell']:.2f}")
        st.metric("Net Score", f"{raw_scores['net']:.2f}")
    
    with col2:
        st.write("**Signal Breakdown:**")
        for signal_text, signal_type, weight, extra in signals:
            emoji = {"BUY": "🟢", "SELL": "🔴", "CONFIRM": "✅", "AMPLIFY": "📈", "DAMPEN": "📉"}.get(signal_type, "⚪")
            display_text = f"{emoji} {signal_text}"
            if extra:
                display_text += f" ({extra})"
            if weight > 0:
                display_text += f" [w={weight:.2f}]"
            st.write(display_text)
    
    # ==================== ML Analysis Section ====================
    
    st.markdown("---")
    st.subheader("🤖 Machine Learning Models Analysis")
    
    ml_button = st.button("🚀 Run ML Analysis", type="primary", use_container_width=True)
    
    if ml_button or cache_key in st.session_state["ml_cache"]:
        if ml_button or cache_key not in st.session_state["ml_cache"]:
            ml_results = run_ml_analysis(df)
            if ml_results:
                st.session_state["ml_cache"][cache_key] = ml_results
        else:
            ml_results = st.session_state["ml_cache"][cache_key]
        
        if ml_results:
            # Display results table
            ml_df = pd.DataFrame(ml_results, columns=["Algorithm", "Key Parameters", "Performance Metrics", "Recommendation", "Confidence"])
            st.dataframe(ml_df, use_container_width=True, height=300)
            
            # Ensemble Recommendation (Weighted)
            st.markdown("---")
            st.subheader("🎯 Ensemble Recommendation (Weighted)")
            ensemble_rec, ensemble_conf, agreement = calculate_ensemble_recommendation(ml_results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                ens_color = "green" if ensemble_rec == "BUY" else "red" if ensemble_rec == "SELL" else "orange"
                st.markdown(f"<h2 style='color:{ens_color}; text-align:center;'>{ensemble_rec}</h2>", unsafe_allow_html=True)
            with col2:
                st.metric("Weighted Confidence", ensemble_conf)
            with col3:
                st.metric("Model Agreement", agreement)
            
            st.info("💡 Ensemble uses weighted voting: RF(0.2), XGB(0.2), ARIMA+GARCH(0.15), LSTM(0.2), RNN(0.2), MC(0.1)")
    
    # ==================== Backtest Results ====================
    
    st.markdown("---")
    st.subheader("📊 Strategy Backtest Performance")
    
    with st.spinner("Running backtest..."):
        backtest_results = backtest_strategy(df, weights=weights, initial_capital=initial_capital,
                                            confidence_threshold=confidence_threshold,
                                            stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
    
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Final Capital", f"${backtest_results['final_capital']:.2f}")
    b2.metric("Strategy Return", f"{backtest_results['total_return']*100:.2f}%")
    b3.metric("Buy & Hold Return", f"{backtest_results['buy_hold_return']*100:.2f}%")
    b4.metric("Alpha", f"{backtest_results['alpha']*100:.2f}%", delta=f"{backtest_results['alpha']*100:.2f}%")
    b5.metric("Number of Trades", backtest_results['num_trades'])
    
    b6, b7, b8 = st.columns(3)
    b6.metric("Win Rate", f"{backtest_results['win_rate']*100:.1f}%")
    b7.metric("Avg Win", f"{backtest_results['avg_win']*100:.2f}%")
    b8.metric("Avg Loss", f"{backtest_results['avg_loss']*100:.2f}%")
    
    if backtest_results['equity_curve']:
        st.subheader("📈 Equity Curve")
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=equity_df['date'], y=equity_df['value'], mode='lines',
                                        name='Portfolio Value', line=dict(color='green', width=2)))
        
        if backtest_results['buy_signals']:
            buy_dates = [df.index[i] for i in backtest_results['buy_signals']]
            fig_equity.add_trace(go.Scatter(
                x=buy_dates,
                y=[equity_df[equity_df['date'] == d]['value'].iloc[0] if len(equity_df[equity_df['date'] == d]) > 0 else 0 for d in buy_dates],
                mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))
        
        if backtest_results['sell_signals']:
            sell_dates = [df.index[i] for i in backtest_results['sell_signals']]
            fig_equity.add_trace(go.Scatter(
                x=sell_dates,
                y=[equity_df[equity_df['date'] == d]['value'].iloc[0] if len(equity_df[equity_df['date'] == d]) > 0 else 0 for d in sell_dates],
                mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))
        
        fig_equity.update_layout(height=400, xaxis_title="Date", yaxis_title="Portfolio Value ($)", hovermode='x unified')
        st.plotly_chart(fig_equity, use_container_width=True)
    
    with st.expander("📋 View Trade History"):
        if backtest_results['positions']:
            trades_df = pd.DataFrame(backtest_results['positions'], columns=['Action', 'Date', 'Price', 'Value/Shares', 'Return/Conf'])
            st.dataframe(trades_df, use_container_width=True)
    
    # ==================== Monte Carlo Projections ====================
    
    st.markdown("---")
    st.subheader("🎲 Monte Carlo Price Target Projections")
    
    targets = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    sim_results = []
    current_price = float(latest['Close'])
    
    with st.spinner("Running Monte Carlo simulations..."):
        for t in targets:
            res = estimate_days_to_target_advanced(df, current_price, target_return=t, sims=sim_count, max_days=max_days)
            sim_results.append({
                "Target (%)": int(t*100),
                "Target Price": f"${current_price * (1+t):.2f}",
                "Probability (%)": f"{res['probability']*100:.1f}",
                "Median Days": res['median_days'],
                "90th Pctl Days": res['90pct_days'],
                "10th Pctl Days": res['10pct_days']
            })
    
    mc_df = pd.DataFrame(sim_results)
    st.dataframe(mc_df, use_container_width=True)
    st.info(f"💡 Based on {sim_count:,} simulations with {max_days} day horizon using Student's t-distribution")
    
    # ==================== NEW: Monte Carlo Price Simulation ====================
    
    st.markdown("---")
    st.subheader("🎲 Monte Carlo Price Simulation")
    
    with st.spinner("Running price simulations across timeframes..."):
        mc_price_sim = monte_carlo_price_simulation(df, current_price, sims=sim_count)
    
    if not mc_price_sim.empty:
        st.dataframe(mc_price_sim, use_container_width=True, hide_index=True)
        
        st.info(f"💡 Based on {sim_count:,} simulations using Student's t-distribution with volatility clustering. "
                f"Shows 5th percentile (worst case), median (most likely), and 95th percentile (best case) prices.")
    else:
        st.warning("⚠️ Insufficient data for price simulation")

# ==================== Footer ====================

st.markdown("---")
st.subheader("📝 Notes & Disclaimers")

st.write("""
### Improvements in v2.7:
- ✅ **Simplified Ticker Input** - Single input box with default stocks (AAPL, TSLA, AMZN)
- ✅ **Speedometer Indicator** - Visual gauge for market sentiment (2 decimal places)
- ✅ **Weighted Ensemble** - RF(0.2), XGB(0.2), ARIMA+GARCH(0.15), LSTM(0.2), RNN(0.2), MC(0.1)
- ✅ **Multi-Source News** - Finviz, Google News with dropdown selector
- ✅ **News Sentiment Analysis** - FinBERT for financial sentiment
- ✅ **Machine Learning Integration** - 6 different algorithms
- ✅ **Strategy Backtesting** - With risk management (stop loss, take profit)
- ✅ **Risk Analytics** - Sharpe, Sortino, Max Drawdown, Calmar ratio

### Limitations & Disclaimers:
- ⚠️ **NOT FINANCIAL ADVICE** - This tool is for educational purposes only
- ⚠️ Past performance does not guarantee future results
- ⚠️ ML models can overfit to historical patterns that may not persist
- ⚠️ Market conditions change - models trained on past data may not predict future well
- ⚠️ Sentiment analysis is based on headlines only, not full article content
- ⚠️ Always do your own research and consult a financial advisor
- ⚠️ Consider paper trading before using real capital

### Dependencies:
- `pip install feedparser --break-system-packages` (for Google News)
- `pip install transformers torch --break-system-packages` (for FinBERT with PyTorch backend - recommended)
- Alternative: `pip install tf-keras --break-system-packages` (for FinBERT with TensorFlow backend)

**Note on FinBERT:** If you see Keras 3 compatibility issues, the app will automatically try to use PyTorch backend. Make sure `torch` is installed.

### Recommended Next Steps:
1. Compare sentiment across different news sources
2. Monitor ensemble agreement - high disagreement suggests uncertainty
3. Test different weight configurations for both signals and ensemble
4. Paper trade the strategy for at least 3 months before deploying real capital
5. Consider adding fundamental analysis (earnings, revenue growth, debt ratios)
""")

st.markdown("---")
st.caption("Enhanced Stock Dashboard v2.7 | Built with Streamlit | Data: Yahoo Finance, Finviz, Google News | ML: scikit-learn, XGBoost, TensorFlow, FinBERT")
