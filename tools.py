import os
import requests
import yfinance as yf
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


def _to_float(val):
    if hasattr(val, 'item'):
        return val.item()
    return float(val)


@tool
def get_stock_analysis(ticker: str) -> str:
    """Analyzes the stock price and technical indicators (RSI, MA20, MA50, Bollinger Bands)."""
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")

        if hist.empty:
            return f"Error: No data found for '{ticker}'"
        close = hist['Close']
        current_price = _to_float(close.iloc[-1])
        ma20 = _to_float(close.rolling(20).mean().iloc[-1])
        ma50 = _to_float(close.rolling(50).mean().iloc[-1])

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = _to_float((100 - 100 / (1 + rs)).iloc[-1])

        bb_ma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = _to_float(bb_ma.iloc[-1] + bb_std.iloc[-1] * 2)
        bb_lower = _to_float(bb_ma.iloc[-1] - bb_std.iloc[-1] * 2)

        month_change = ((current_price - _to_float(close.iloc[0])) / _to_float(close.iloc[0])) * 100

        volume = hist['Volume'].iloc[-1]
        avg_volume = hist['Volume'].mean()
        vol_ratio = _to_float(volume) / _to_float(avg_volume)

        info = stock.info
        pe = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', None)
        mc_str = f"${market_cap/1e9:.1f}B" if market_cap else "N/A"

        rsi_label = "(Overbought ⬆️)" if rsi > 70 else "(Oversold ⬇️)" if rsi < 30 else "(Neutral →)"
        trend = "Bullish 📈" if current_price > ma50 else "Bearish 📉"

        return f"""📊 {ticker} Technical Analysis:
• Current Price: ${current_price:.2f} | Monthly Change: {month_change:+.2f}%
• Trend: {trend}
• MA20: ${ma20:.2f} | MA50: ${ma50:.2f}
• RSI(14): {rsi:.1f} {rsi_label}
• Bollinger Bands: ${bb_lower:.2f} – ${bb_upper:.2f}
• Volume Ratio: {vol_ratio:.2f}x (vs. average)
• P/E: {pe} | Market Cap: {mc_str}"""
    except Exception as e:
        return f"Technical Analysis Error: {str(e)}"
@tool
def get_stock_recommendation(ticker: str) -> str:
    """Provides a BUY/SELL/HOLD recommendation for a stock based on technical indicators."""
    try:
        ticker = ticker.strip().upper()
        hist = yf.Ticker(ticker).history(period="3mo")
        if hist.empty:
            return f"Error: No data found for '{ticker}'"

        close = hist['Close']
        current = _to_float(close.iloc[-1])
        ma50 = _to_float(close.rolling(50).mean().iloc[-1])
        ma20 = _to_float(close.rolling(20).mean().iloc[-1])

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = _to_float((100 - 100 / (1 + gain / loss)).iloc[-1])

        signals = []
        score = 0

        if current > ma50:
            score += 1
            signals.append("✅ The price is above the MA50.")
        else:
            score -= 1
            signals.append("❌ The price is below the MA50.")

        if current > ma20:
            score += 1
            signals.append("✅ The price is above the MA20.")
        else:
            score -= 1
            signals.append("❌ The price is below the MA20.")

        if rsi < 70:
            score += 1
            signals.append(f"✅ RSI {rsi:.1f} - not overbought")
        else:
            score -= 1
            signals.append(f"❌ RSI {rsi:.1f} - Overbought")

        if rsi > 30:
            score += 0
        else:
            score += 1
            signals.append(f"✅ RSI {rsi:.1f} - Oversold (buying opportunity).")

        if score >= 2:
            rec = "BUY 🟢"
        elif score <= -1:
            rec = "SELL 🔴"
        else:
            rec = "HOLD 🟡"

        signals_str = "\n".join(f"  {s}" for s in signals)
        return f"💡 {ticker} recommendation: {rec}\n\nSignals:\n{signals_str}"
    except Exception as e:
        return f"Recommendation error: {str(e)}"


@tool
def compare_stocks(ticker1: str, ticker2: str) -> str:
    """Compares the technical analysis of two stocks: price, change, RSI, volume, P/E."""
    try:
        ticker1, ticker2 = ticker1.strip().upper(), ticker2.strip().upper()
        results = {}
        for t in [ticker1, ticker2]:
            hist = yf.Ticker(t).history(period="1mo")
            info = yf.Ticker(t).info
            if hist.empty:
                return f"Error: No data found for '{t}"
            close = hist['Close']
            current = _to_float(close.iloc[-1])
            change = ((current - _to_float(close.iloc[0])) / _to_float(close.iloc[0])) * 100
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = _to_float((100 - 100 / (1 + gain / loss)).iloc[-1])
            avg_vol = _to_float(hist['Volume'].mean())
            pe = info.get('trailingPE', None)
            mc = info.get('marketCap', None)
            results[t] = {
                'price': current, 'change': change, 'rsi': rsi,
                'avg_vol': avg_vol, 'pe': pe, 'mc': mc
            }

        def fmt(t):
            r = results[t]
            mc_str = f"${r['mc']/1e9:.1f}B" if r['mc'] else "N/A"
            pe_str = f"{r['pe']:.1f}" if r['pe'] else "N/A"
            return (f"{t}:\n"
                    f"  Price: ${r['price']:.2f} | Change: {r['change']:+.2f}%\n"
                    f"  RSI: {r['rsi']:.1f} | P/E: {pe_str}\n"
                    f"  Market Capitalization: {mc_str}\n"
                    f"  Average Volume: {r['avg_vol']:,.0f}")

        winner = ticker1 if results[ticker1]['change'] > results[ticker2]['change'] else ticker2
        return f"🔄 Comparison: {ticker1} vs {ticker2}\n{'─'*35}\n{fmt(ticker1)}\n{'─'*35}\n{fmt(ticker2)}\n{'─'*35}\n🏆 Better Performer: {winner}"
    except Exception as e:
        return f"Comparison error: {str(e)}"


@tool
def get_market_news(query: str) -> str:
    """Gets the latest 10 news articles from financial news sources."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return "⚠️ NEWSAPI_KEY is not configured. Please set it in your environment variables."
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}&pageSize=10&sortBy=publishedAt"
            f"&language=en&apiKey={api_key}"
        )
        data = requests.get(url, timeout=10).json()
        if data.get('status') != 'ok':
            return f"API error: {data.get('message', 'Unknown error')}"
        articles = data.get('articles', [])
        if not articles:
            return f"No news found for '{query}'"
        news = f"📰 '{query}' related news:\n\n"
        for i, a in enumerate(articles, 1):
            source = a.get('source', {}).get('name', 'Unknown')
            desc = (a.get('description') or '')[:120]
            news += f"{i}. **{a['title']}**\n   📌 {source} — {a['publishedAt'][:10]}\n   {desc}\n\n"
        return news
    except Exception as e:
        return f"Error fetching news: {str(e)}"


tools = [
    get_stock_analysis,
    get_market_news,
    compare_stocks,
    get_stock_recommendation,
]