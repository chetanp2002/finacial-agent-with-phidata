import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import os
from dotenv import load_dotenv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import talib
from sklearn.linear_model import LinearRegression
from textblob import TextBlob  # For sentiment analysis

# Load environment variables from .env
load_dotenv()

# Get API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Enhanced stock symbol mappings
COMMON_STOCKS = {
    'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'GOOGLE': 'GOOGL', 'MICROSOFT': 'MSFT',
    'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'META': 'META', 'NETFLIX': 'NFLX',
    'TCS': 'TCS.NS', 'RELIANCE': 'RELIANCE.NS', 'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS', 'HDFC': 'HDFCBANK.NS', 'TATAMOTORS': 'TATAMOTORS.NS',
    'ICICIBANK': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS', 'MARUTI': 'MARUTI.NS',
    'BHARTIARTL': 'BHARTIARTL.NS', 'HCLTECH': 'HCLTECH.NS', 'ITC': 'ITC.NS',
    'AXISBANK': 'AXISBANK.NS'
}

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stock-header {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.watchlist = set()
    st.session_state.analysis_history = []

# Get stock symbol
def get_symbol_from_name(stock_name):
    stock_name = stock_name.strip().upper()
    return COMMON_STOCKS.get(stock_name, stock_name)

# Fetch stock data
def get_stock_data(symbol, period="1y"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError("No historical data available")
        return stock.info, hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

# Plot stock price
def create_price_chart(hist_data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close']
    ))
    fig.update_layout(title=f'{symbol} Stock Price', xaxis_rangeslider_visible=False)
    return fig

# Display metrics
def display_metrics(info):
    col1, col2, col3 = st.columns(3)
    col1.metric("Market Cap", info.get('marketCap', 'N/A'))
    col2.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
    col3.metric("Dividend Yield", info.get('dividendYield', 'N/A'))

# Moving Average Convergence Divergence (MACD)
def macd_indicator(stock_symbol):
    st.title(f"MACD Indicator for {stock_symbol}")
    _, hist = get_stock_data(stock_symbol)
    
    if hist is not None:
        # Calculate MACD using talib
        hist['MACD'], hist['MACD_signal'], hist['MACD_hist'] = talib.MACD(
            hist['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_signal'], mode='lines', name='MACD Signal'))
        fig.update_layout(title=f"{stock_symbol} MACD Indicator")
        st.plotly_chart(fig)

# Stock Price Prediction (Linear Regression Model)
def predict_stock_price(stock_symbol):
    st.title(f"Stock Price Prediction for {stock_symbol}")
    
    _, hist = get_stock_data(stock_symbol)
    
    if hist is not None:
        hist['Date'] = hist.index
        hist['Date'] = hist['Date'].map(pd.Timestamp.toordinal)
        
        # Prepare data for Linear Regression model
        X = hist[['Date']]
        y = hist['Close']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict the next day's price
        next_day = pd.Timestamp.today() + pd.Timedelta(days=1)
        next_day_ordinal = next_day.toordinal()
        
        predicted_price = model.predict([[next_day_ordinal]])[0]
        
        st.write(f"Predicted Price for {stock_symbol} on {next_day.date()}: ₹{predicted_price:.2f}")

# Stock Performance Summary (with comparison table)
def stock_performance_summary(stock_symbol):
    st.title(f"Stock Performance Summary for {stock_symbol}")
    
    _, hist = get_stock_data(stock_symbol)
    
    if hist is not None:
        # Calculate performance metrics
        price_change = hist['Close'][-1] - hist['Close'][0]
        price_change_percent = (price_change / hist['Close'][0]) * 100
        volatility = hist['Close'].std()
        average_volume = hist['Volume'].mean()
        
        # Create a table to display the comparison
        data = {
            "Metric": ["Price Change", "Price Change (%)", "Volatility (Std. Dev.)", "Average Volume"],
            "Value": [
                f"₹{price_change:.2f}", 
                f"{price_change_percent:.2f}%", 
                f"₹{volatility:.2f}", 
                f"{average_volume:.0f} shares"
            ]
        }
        
        comparison_df = pd.DataFrame(data)
        st.table(comparison_df)
        
        # Plot stock price trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Stock Price'))
        fig.update_layout(title=f"{stock_symbol} Stock Price Trend")
        st.plotly_chart(fig)

# Volume Analysis
def volume_analysis(stock_symbol):
    st.title(f"Volume Analysis for {stock_symbol}")
    
    _, hist = get_stock_data(stock_symbol)
    
    if hist is not None:
        # Analyze volume for unusual activity
        hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()  # 20-day moving average
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Volume'], mode='lines', name='Volume'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Volume_MA'], mode='lines', name='20-Day Moving Avg. Volume'))
        fig.update_layout(title=f"Volume Analysis: {stock_symbol}")
        st.plotly_chart(fig)

# Sentiment Analysis of Stock News (Basic Implementation)
def sentiment_analysis(stock_symbol):
    st.title(f"Sentiment Analysis of News for {stock_symbol}")
    
    # Example news headlines (for demonstration purposes)
    headlines = [
        "NVIDIA Stock Surges After Positive Earnings Report",
        "Apple Faces Declining Sales in China Amid Trade Tensions",
        "Tesla Reports Record Deliveries in Q3"
    ]
    
    sentiment_scores = []
    for headline in headlines:
        blob = TextBlob(headline)
        sentiment_scores.append(blob.sentiment.polarity)
    
    average_sentiment = np.mean(sentiment_scores)
    
    sentiment_label = "Positive" if average_sentiment > 0 else "Negative" if average_sentiment < 0 else "Neutral"
    
    st.write(f"Average Sentiment: {sentiment_label} (Score: {average_sentiment:.2f})")
    
    # Plot sentiment analysis chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=headlines, y=sentiment_scores, name="Sentiment Score"))
    fig.update_layout(title="Sentiment Analysis of News Headlines")
    st.plotly_chart(fig)

# Relative Strength Index (RSI) Indicator
def rsi_indicator(stock_symbol):
    st.title(f"RSI Indicator for {stock_symbol}")
    
    _, hist = get_stock_data(stock_symbol)
    
    if hist is not None:
        # Calculate RSI using talib
        hist['RSI'] = talib.RSI(hist['Close'], timeperiod=14)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI'))
        fig.update_layout(title=f"{stock_symbol} RSI Indicator", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

# Stock Comparison Function
def stock_comparison(symbol1, symbol2):
    st.title(f"Stock Comparison: {symbol1} vs {symbol2}")
    
    _, hist1 = get_stock_data(symbol1)
    _, hist2 = get_stock_data(symbol2)
    
    if hist1 is not None and hist2 is not None:
        # Compare closing prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist1.index, y=hist1['Close'], mode='lines', name=f'{symbol1} Close Price'))
        fig.add_trace(go.Scatter(x=hist2.index, y=hist2['Close'], mode='lines', name=f'{symbol2} Close Price'))
        fig.update_layout(title=f"Stock Comparison: {symbol1} vs {symbol2}")
        st.plotly_chart(fig)

# Stock Recommendation based on basic features
def stock_recommendation(info):
    st.title("Stock Recommendation Based on Basic Metrics")
    
    recommendation = "Hold"
    market_cap = info.get('marketCap', 0)
    pe_ratio = info.get('trailingPE', 0)
    
    if market_cap < 10e9:
        recommendation = "Sell"
    elif market_cap > 10e9 and pe_ratio < 25:
        recommendation = "Buy"
    
    st.write(f"Recommendation: {recommendation}")

# Main function to display the stock analysis interface
def main():
    st.sidebar.title("Stock Market Analysis")
    
    symbol_name = st.sidebar.text_input("Enter Stock Name or Symbol (e.g., AAPL for Apple):")
    if symbol_name:
        symbol = get_symbol_from_name(symbol_name)
        st.sidebar.write(f"Selected Stock: {symbol}")
        
        analysis_options = ['MACD', 'Stock Prediction', 'Performance Summary', 'Volume Analysis', 'Sentiment Analysis', 'RSI', 'Stock Comparison', 'Stock Recommendation']
        choice = st.sidebar.selectbox('Choose Analysis Type:', analysis_options)
        
        if choice == 'MACD':
            macd_indicator(symbol)
        elif choice == 'Stock Prediction':
            predict_stock_price(symbol)
        elif choice == 'Performance Summary':
            stock_performance_summary(symbol)
        elif choice == 'Volume Analysis':
            volume_analysis(symbol)
        elif choice == 'Sentiment Analysis':
            sentiment_analysis(symbol)
        elif choice == 'RSI':
            rsi_indicator(symbol)
        elif choice == 'Stock Comparison':
            comparison_symbol = st.sidebar.text_input("Enter Stock for Comparison:")
            if comparison_symbol:
                stock_comparison(symbol, get_symbol_from_name(comparison_symbol))
        elif choice == 'Stock Recommendation':
            info, _ = get_stock_data(symbol)
            if info:
                stock_recommendation(info)

# Run the app
if __name__ == "__main__":
    main()
