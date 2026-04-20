import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import tools
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Stock AI Assistant",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark terminal background */
.stApp {
    background-color: #0d0f14;
    color: #c8d0e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2230;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #111318;
    border-bottom: 1px solid #1e2230;
    padding: 0 1rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    padding: 0.7rem 1.4rem;
    color: #5a6480;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #5a6480;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    color: #e0e8ff;
}

/* Info/success/warning boxes */
.custom-card {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    line-height: 1.7;
}
.card-buy  { border-left: 3px solid #3fb950; }
.card-sell { border-left: 3px solid #f85149; }
.card-hold { border-left: 3px solid #d29922; }
.card-info { border-left: 3px solid #58a6ff; }

/* News card */
.news-card {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: #58a6ff; }
.news-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #c8d0e0;
    margin-bottom: 0.25rem;
}
.news-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #5a6480;
}
.news-desc {
    font-size: 0.82rem;
    color: #8896b3;
    margin-top: 0.4rem;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 8px;
}

/* Inputs */
input, textarea, .stTextInput > div > div > input {
    background-color: #13161e !important;
    border: 1px solid #1e2230 !important;
    color: #c8d0e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 6px !important;
}

/* Buttons */
.stButton > button {
    background: #1a1f2e;
    border: 1px solid #2a3050;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    border-radius: 6px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #1e2540;
    border-color: #58a6ff;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #13161e !important;
    border: 1px solid #1e2230 !important;
    color: #c8d0e0 !important;
}

/* Section title */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a6480;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2230;
}

/* Compare table */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
.compare-table th {
    background: #1a1f2e;
    color: #5a6480;
    padding: 0.6rem 1rem;
    text-align: left;
    font-weight: 400;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    text-transform: uppercase;
}
.compare-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #1e2230;
    color: #c8d0e0;
}
.compare-table tr:hover td { background: #13161e; }
.green { color: #3fb950; }
.red   { color: #f85149; }
.yellow { color: #d29922; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #1e2230; border-radius: 2px; }

/* Portfolio badge */
.portfolio-badge {
    display: inline-block;
    background: #1a1f2e;
    border: 1px solid #2a3050;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 1rem; border-bottom: 1px solid #1e2230; margin-bottom: 1.5rem;">
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#5a6480; letter-spacing:0.2em; text-transform:uppercase;">
        STOCK AI ASSISTANT
    </span>
    <h1 style="font-family:'IBM Plex Mono',monospace; font-size:1.6rem; color:#e0e8ff; margin:0.3rem 0 0; font-weight:600; letter-spacing:-0.02em;">
        Moliyaviy Analitik Terminal
    </h1>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def _to_float(val):
    if hasattr(val, 'item'):
        return val.item()
    return float(val)


with st.sidebar:
    st.markdown('<div class="section-title">Portfolio Kuzatish</div>', unsafe_allow_html=True)

    portfolio_input = st.text_input(
        "Tickerlar (vergul bilan)",
        "AAPL,GOOGL,MSFT,TSLA",
        label_visibility="collapsed",
        placeholder="AAPL,GOOGL,MSFT..."
    )

    if portfolio_input:
        tickers_list = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]
        with st.spinner(""):
            for t in tickers_list:
                try:
                    d = yf.download(t, period="5d", progress=False)
                    if not d.empty:
                        cur = _to_float(d['Close'].iloc[-1])
                        prev = _to_float(d['Close'].iloc[-2]) if len(d) > 1 else _to_float(d['Close'].iloc[0])
                        chg = (cur - prev) / prev * 100
                        color = "#3fb950" if chg >= 0 else "#f85149"
                        arrow = "▲" if chg >= 0 else "▼"
                        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            background:#13161e;border:1px solid #1e2230;border-radius:6px;
            padding:0.5rem 0.8rem;margin:0.3rem 0;">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#e0e8ff;font-weight:600;">{t}</span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;">
    <span style="color:#8896b3;">${cur:.2f}</span>
    <span style="color:{color};margin-left:0.4rem;">{arrow}{abs(chg):.2f}%</span>
  </span>
</div>""", unsafe_allow_html=True)
                except:
                    pass

    st.markdown("---")
    st.markdown('<div class="section-title">Sozlamalar</div>', unsafe_allow_html=True)

    from dotenv import dotenv_values
    cfg = dotenv_values(".env")
    for key, label in [("OPENAI_API_KEY", "OpenAI API"), ("NEWSAPI_KEY", "NewsAPI")]:
        ok = key in cfg and cfg[key]
        dot = "🟢" if ok else "🔴"
        st.markdown(f'<span style="font-family:monospace;font-size:0.78rem;">{dot} {label}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.72rem;color:#3a4060;line-height:1.6;">
⚠️ Faqat ta'limiy maqsadda.<br>
Professional konsultant bilan maslahatlashing.
</div>""", unsafe_allow_html=True)


# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Grafik", "📰 Yangiliklar", "🤖 AI Analitik", "📈 Tahlil"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Grafik
# ══════════════════════════════════════════════════════════════════════════════
PERIOD_MAP = {
    "1 soat":  ("1d",  "1m"),
    "6 soat":  ("1d",  "5m"),
    "12 soat": ("1d",  "15m"),
    "24 soat": ("2d",  "30m"),
    "1 hafta": ("5d",  "1h"),
    "1 oy":    ("1mo", "1d"),
    "1 yil":   ("1y",  "1wk"),
}

with tab1:
    st.markdown('<div class="section-title">Real-Time Grafik Tahlil</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([4, 2, 1])
    with col_a:
        graph_ticker = st.text_input("Grafik uchun ticker", "AAPL", key="g_ticker",
                                     label_visibility="collapsed", placeholder="Ticker kiriting...")
    with col_b:
        period_label = st.selectbox("Davr", list(PERIOD_MAP.keys()),
                                    index=5, key="g_period", label_visibility="collapsed")
    with col_c:
        chart_type = st.selectbox("Turi", ["Sham", "Chiziq"], key="g_type", label_visibility="collapsed")

    if graph_ticker and graph_ticker.strip():
        t = graph_ticker.strip().upper()
        period_yf, interval_yf = PERIOD_MAP[period_label]
        try:
            with st.spinner(f"{t} yuklanmoqda..."):
                data = yf.download(t, period=period_yf, interval=interval_yf, progress=False)

            if data.empty:
                st.error(f"'{t}' uchun ma'lumot topilmadi")
            else:
                fig = go.Figure()
                if chart_type == "Sham":
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'].values.flatten(),
                        high=data['High'].values.flatten(),
                        low=data['Low'].values.flatten(),
                        close=data['Close'].values.flatten(),
                        name=t,
                        increasing_line_color='#3fb950',
                        decreasing_line_color='#f85149',
                        increasing_fillcolor='#3fb950',
                        decreasing_fillcolor='#f85149',
                    ))
                else:
                    close_vals = data['Close'].values.flatten()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=close_vals,
                        mode='lines',
                        name=t,
                        line=dict(color='#58a6ff', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(88,166,255,0.06)',
                    ))
                    # MA overlay
                    close_series = data['Close'].squeeze()
                    if len(close_series) >= 20:
                        ma20 = close_series.rolling(20).mean()
                        fig.add_trace(go.Scatter(
                            x=data.index, y=ma20.values,
                            mode='lines', name='MA20',
                            line=dict(color='#d29922', width=1, dash='dot'),
                            opacity=0.8
                        ))

                fig.update_layout(
                    paper_bgcolor='#0d0f14',
                    plot_bgcolor='#0d0f14',
                    font=dict(family='IBM Plex Mono', color='#5a6480', size=11),
                    height=460,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(gridcolor='#1a1f2e', showgrid=True, zeroline=False,
                               rangeslider=dict(visible=False)),
                    yaxis=dict(gridcolor='#1a1f2e', showgrid=True, zeroline=False,
                               side='right'),
                    hovermode='x unified',
                    hoverlabel=dict(bgcolor='#13161e', font_color='#c8d0e0',
                                    font_family='IBM Plex Mono'),
                    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8896b3')),
                    title=dict(text=f"{t} — {period_label}", font=dict(color='#e0e8ff', size=13)),
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Grafik xatolik: {str(e)}")

    # ── Statistika (bir yoki bir nechta ticker) ──
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Statistika</div>', unsafe_allow_html=True)
    stats_input = st.text_input("Statistika uchun tickerlar (vergul bilan)", "AAPL,MSFT,GOOGL",
                                 key="stats_tickers", label_visibility="collapsed",
                                 placeholder="AAPL,MSFT,GOOGL...")
    stats_period = st.selectbox("Statistika davri", list(PERIOD_MAP.keys()),
                                 index=5, key="stats_period", label_visibility="collapsed")

    if stats_input:
        stats_list = [x.strip().upper() for x in stats_input.split(",") if x.strip()]
        s_period, s_interval = PERIOD_MAP[stats_period]
        cols_stat = st.columns(len(stats_list))
        for idx, t in enumerate(stats_list):
            with cols_stat[idx]:
                try:
                    d = yf.download(t, period=s_period, interval=s_interval, progress=False)
                    if not d.empty:
                        cur = _to_float(d['Close'].iloc[-1])
                        start = _to_float(d['Close'].iloc[0])
                        high = _to_float(d['High'].max())
                        low = _to_float(d['Low'].min())
                        chg = (cur - start) / start * 100
                        vol = _to_float(d['Volume'].mean())
                        st.metric(t, f"${cur:.2f}", f"{chg:+.2f}%",
                                  delta_color="normal" if chg >= 0 else "inverse")
                        st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a6480;line-height:1.8;margin-top:0.3rem;">
  ↑ Yuqori: <span style="color:#3fb950;">${high:.2f}</span><br>
  ↓ Past:   <span style="color:#f85149;">${low:.2f}</span><br>
  ≋ Hajm:   {vol:,.0f}
</div>""", unsafe_allow_html=True)
                except:
                    st.caption(f"{t}: xatolik")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Yangiliklar
# ══════════════════════════════════════════════════════════════════════════════
import os, requests

with tab2:
    st.markdown('<div class="section-title">Bozor Yangiliklari</div>', unsafe_allow_html=True)

    col_n1, col_n2 = st.columns([5, 1])
    with col_n1:
        news_query = st.text_input("Qidiruv", "stock market", key="nq",
                                    label_visibility="collapsed",
                                    placeholder="Kompaniya yoki kalit so'z kiriting...")
    with col_n2:
        fetch_btn = st.button("Qidirish →", use_container_width=True)

    if fetch_btn or "last_news_query" not in st.session_state:
        query_to_use = news_query or "stock market"
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            st.error("⚠️ NEWSAPI_KEY konfiguratsiya qilinmagan. .env faylini tekshiring.")
        else:
            with st.spinner("Yangiliklar yuklanmoqda..."):
                try:
                    url = (
                        f"https://newsapi.org/v2/everything"
                        f"?q={query_to_use}&pageSize=10&sortBy=publishedAt"
                        f"&language=en&apiKey={api_key}"
                    )
                    resp = requests.get(url, timeout=10).json()
                    articles = resp.get('articles', [])
                    st.session_state["last_news_articles"] = articles
                    st.session_state["last_news_query"] = query_to_use
                except Exception as e:
                    st.error(f"Yangilik yuklashda xatolik: {e}")

    articles = st.session_state.get("last_news_articles", [])
    if articles:
        st.markdown(f'<div style="font-family:monospace;font-size:0.72rem;color:#5a6480;margin-bottom:1rem;">— {len(articles)} ta natija: "{st.session_state.get("last_news_query","")}"</div>', unsafe_allow_html=True)
        for i, a in enumerate(articles):
            source = a.get('source', {}).get('name', 'Unknown')
            date = a['publishedAt'][:10]
            title = a.get('title', '')
            desc = (a.get('description') or '')[:160]
            url_link = a.get('url', '#')
            st.markdown(f"""
<div class="news-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
    <div>
      <div class="news-title">{i+1}. {title}</div>
      <div class="news-meta">📌 {source} &nbsp;·&nbsp; 📅 {date}</div>
      <div class="news-desc">{desc}</div>
    </div>
    <a href="{url_link}" target="_blank" style="flex-shrink:0;font-family:'IBM Plex Mono',monospace;
       font-size:0.7rem;color:#58a6ff;text-decoration:none;white-space:nowrap;">O'qish →</a>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI Analitik
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">AI Moliya Analitik</div>', unsafe_allow_html=True)

    import os as _os
    _oai_key = _os.getenv("OPENAI_API_KEY")
    if not _oai_key:
        st.error("⚠️ OPENAI_API_KEY konfiguratsiya qilinmagan.")
    else:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=_oai_key
        )

        _system = """Siz tajribali moliya analitikasisiz. Asosan aksiyalar va moliyaviy bozorlar haqida maslahat berasiz.

Qoidalar:
- Faqat moliyaviy savollarga javob bering
- Mavjud toollardan ma'lumot oling va ularga tayanib o'zingiz qaror qabul qiling
- Tahlilni quyidagi tizimda bering: texnik holat → signal → tavsiya → risk
- Har doim risk ogohlantirishi qo'shing
- O'zbek tilida javob bering
- Javobni qisqa, aniq va professional usulda yozing"""

        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", _system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Suggested questions
        st.markdown('<div style="margin-bottom:1rem;">', unsafe_allow_html=True)
        sugg_cols = st.columns(3)
        suggestions = [
            "AAPL uchun tavsiya bering",
            "TSLA vs NVDA qiyosla",
            "MSFT texnik tahlil",
        ]
        for i, s in enumerate(suggestions):
            with sugg_cols[i]:
                if st.button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state._pending_input = s
        st.markdown('</div>', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pending = st.session_state.pop("_pending_input", None)
        user_input = st.chat_input("Moliya savoli bering...") or pending

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Tahlil qilinmoqda..."):
                    try:
                        resp = agent_executor.invoke({
                            "input": user_input,
                            "chat_history": [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages[:-1]
                            ]
                        })
                        out = resp.get("output", "Javob olinmadi")
                    except Exception as e:
                        out = f"⚠️ Xatolik: {str(e)}"
                st.markdown(out)

            st.session_state.messages.append({"role": "assistant", "content": out})

        if st.session_state.messages:
            if st.button("🗑️ Suhbatni tozalash", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Tahlil
# ══════════════════════════════════════════════════════════════════════════════
def fetch_full_info(ticker: str):
    """Aksiya haqida to'liq ma'lumot olish."""
    try:
        t = ticker.strip().upper()
        stock = yf.Ticker(t)
        hist = stock.history(period="3mo")
        info = stock.info
        if hist.empty:
            return None, None

        close = hist['Close']
        current = _to_float(close.iloc[-1])
        open_p = _to_float(hist['Open'].iloc[-1])
        high52 = _to_float(hist['High'].max())
        low52 = _to_float(hist['Low'].min())
        vol = _to_float(hist['Volume'].iloc[-1])
        avg_vol = _to_float(hist['Volume'].mean())
        ma20 = _to_float(close.rolling(20).mean().iloc[-1])
        ma50 = _to_float(close.rolling(50).mean().iloc[-1])

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = _to_float((100 - 100 / (1 + gain / loss)).iloc[-1])

        chg_1m = (current - _to_float(close.iloc[0])) / _to_float(close.iloc[0]) * 100

        return {
            'ticker': t,
            'current': current,
            'open': open_p,
            'high52': high52,
            'low52': low52,
            'vol': vol,
            'avg_vol': avg_vol,
            'ma20': ma20,
            'ma50': ma50,
            'rsi': rsi,
            'chg_1m': chg_1m,
            'pe': info.get('trailingPE'),
            'pb': info.get('priceToBook'),
            'ps': info.get('priceToSalesTrailing12Months'),
            'eps': info.get('trailingEps'),
            'div': info.get('dividendYield'),
            'beta': info.get('beta'),
            'mc': info.get('marketCap'),
            'sector': info.get('sector', 'N/A'),
            'name': info.get('shortName', t),
            'rec': info.get('recommendationKey', 'N/A').upper(),
            'target': info.get('targetMeanPrice'),
        }, None
    except Exception as e:
        return None, str(e)


with tab4:
    st.markdown('<div class="section-title">Chuqur Tahlil</div>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1, 1])

    # ── Bitta aksiya tahlili ─────────────────────────────────────────────────
    with col_t1:
        st.markdown('<div style="font-size:0.8rem;color:#8896b3;margin-bottom:0.6rem;">Bitta Aksiya Tahlili</div>', unsafe_allow_html=True)
        single_ticker = st.text_input("", "AAPL", key="single_t",
                                       label_visibility="collapsed",
                                       placeholder="Ticker kiriting...")
        analyze_btn = st.button("Tahlil →", key="btn_single", use_container_width=True)

        if analyze_btn and single_ticker:
            with st.spinner(f"{single_ticker.upper()} tahlil qilinmoqda..."):
                d, err = fetch_full_info(single_ticker)

            if err:
                st.error(f"Xatolik: {err}")
            elif d:
                # Price block
                chg_color = "#3fb950" if d['chg_1m'] >= 0 else "#f85149"
                arrow = "▲" if d['chg_1m'] >= 0 else "▼"
                rec_color = {"BUY": "#3fb950", "STRONG_BUY": "#3fb950",
                             "HOLD": "#d29922", "SELL": "#f85149",
                             "STRONG_SELL": "#f85149"}.get(d['rec'], "#8896b3")

                st.markdown(f"""
<div class="custom-card card-info">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#5a6480;">{d['name']}</div>
  <div style="font-size:2rem;font-weight:700;color:#e0e8ff;margin:0.2rem 0;">${d['current']:.2f}</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:{chg_color};">
    {arrow} {abs(d['chg_1m']):.2f}% (3 oy)
  </div>
</div>""", unsafe_allow_html=True)

                # Metrics table
                mc_str = f"${d['mc']/1e9:.2f}B" if d['mc'] else "N/A"
                target_str = f"${d['target']:.2f}" if d['target'] else "N/A"
                upside = f"{(d['target']/d['current']-1)*100:+.1f}%" if d['target'] else "—"

                rows = [
                    ("Sektor", d['sector']),
                    ("Bozor kapitali", mc_str),
                    ("P/E", f"{d['pe']:.1f}" if d['pe'] else "N/A"),
                    ("P/B", f"{d['pb']:.2f}" if d['pb'] else "N/A"),
                    ("P/S", f"{d['ps']:.2f}" if d['ps'] else "N/A"),
                    ("EPS", f"${d['eps']:.2f}" if d['eps'] else "N/A"),
                    ("Dividend", f"{d['div']*100:.2f}%" if d['div'] else "0%"),
                    ("Beta", f"{d['beta']:.2f}" if d['beta'] else "N/A"),
                    ("52H Yuqori", f"${d['high52']:.2f}"),
                    ("52H Past", f"${d['low52']:.2f}"),
                    ("MA20", f"${d['ma20']:.2f}"),
                    ("MA50", f"${d['ma50']:.2f}"),
                    ("RSI(14)", f"{d['rsi']:.1f}"),
                    ("O'rtacha hajm", f"{d['avg_vol']:,.0f}"),
                    ("Maqsad narx", f"{target_str} ({upside})"),
                    ("Tavsiya", f'<span style="color:{rec_color};">{d["rec"]}</span>'),
                ]

                table_rows = "".join(
                    f"<tr><td style='color:#5a6480;'>{k}</td><td style='text-align:right;'>{v}</td></tr>"
                    for k, v in rows
                )
                st.markdown(f"""
<div class="custom-card" style="padding:0;overflow:hidden;margin-top:0.6rem;">
<table class="compare-table">{table_rows}</table>
</div>""", unsafe_allow_html=True)

                # RSI gauge style
                rsi_label = "Overbought 🔴" if d['rsi'] > 70 else "Oversold 🟢" if d['rsi'] < 30 else "Neutral 🟡"
                rsi_pct = min(max(d['rsi'], 0), 100)
                rsi_clr = "#f85149" if d['rsi'] > 70 else "#3fb950" if d['rsi'] < 30 else "#d29922"
                st.markdown(f"""
<div class="custom-card" style="margin-top:0.6rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a6480;margin-bottom:0.4rem;">
    RSI(14): {d['rsi']:.1f} — {rsi_label}
  </div>
  <div style="background:#1a1f2e;border-radius:4px;height:6px;overflow:hidden;">
    <div style="width:{rsi_pct}%;background:{rsi_clr};height:100%;border-radius:4px;"></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Qiyoslash ───────────────────────────────────────────────────────────
    with col_t2:
        st.markdown('<div style="font-size:0.8rem;color:#8896b3;margin-bottom:0.6rem;">Ikkita Aksiyani Solishtirish</div>', unsafe_allow_html=True)
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            cmp_t1 = st.text_input("", "AAPL", key="cmp1",
                                    label_visibility="collapsed", placeholder="Ticker 1")
        with cmp_col2:
            cmp_t2 = st.text_input("", "MSFT", key="cmp2",
                                    label_visibility="collapsed", placeholder="Ticker 2")
        cmp_btn = st.button("Solishtir →", key="btn_cmp", use_container_width=True)

        if cmp_btn and cmp_t1 and cmp_t2:
            with st.spinner("Ma'lumotlar yuklanmoqda..."):
                d1, e1 = fetch_full_info(cmp_t1)
                d2, e2 = fetch_full_info(cmp_t2)

            if e1 or e2:
                st.error(f"Xatolik: {e1 or e2}")
            elif d1 and d2:
                def better(v1, v2, lower_is_better=False):
                    if v1 is None or v2 is None:
                        return "#8896b3", "#8896b3"
                    if lower_is_better:
                        return ("#3fb950", "#f85149") if v1 < v2 else ("#f85149", "#3fb950")
                    return ("#3fb950", "#f85149") if v1 > v2 else ("#f85149", "#3fb950")

                chg_c1, chg_c2 = better(d1['chg_1m'], d2['chg_1m'])
                rsi_c1, rsi_c2 = better(d1['rsi'], d2['rsi'], lower_is_better=True)
                pe_c1, pe_c2   = better(d1['pe'], d2['pe'], lower_is_better=True)
                mc_c1, mc_c2   = better(d1['mc'], d2['mc'])

                def cmp_row(label, v1, v2, c1="#e0e8ff", c2="#e0e8ff", fmt=lambda x: str(x)):
                    _v1 = fmt(v1) if v1 is not None else "N/A"
                    _v2 = fmt(v2) if v2 is not None else "N/A"
                    return f"""<tr>
<td style='color:#5a6480;width:34%;'>{label}</td>
<td style='text-align:center;color:{c1};width:33%;'>{_v1}</td>
<td style='text-align:center;color:{c2};width:33%;'>{_v2}</td>
</tr>"""

                rows_cmp = (
                    f"<tr style='background:#1a1f2e;'>"
                    f"<th style='width:34%;'></th>"
                    f"<th style='text-align:center;width:33%;color:#e0e8ff;'>{d1['ticker']}</th>"
                    f"<th style='text-align:center;width:33%;color:#e0e8ff;'>{d2['ticker']}</th>"
                    f"</tr>"
                    + cmp_row("Narx", d1['current'], d2['current'], fmt=lambda x: f"${x:.2f}")
                    + cmp_row("3 oy o'zgarish", d1['chg_1m'], d2['chg_1m'], chg_c1, chg_c2, fmt=lambda x: f"{x:+.2f}%")
                    + cmp_row("Bozor kapitali", d1['mc'], d2['mc'], mc_c1, mc_c2, fmt=lambda x: f"${x/1e9:.1f}B")
                    + cmp_row("P/E", d1['pe'], d2['pe'], pe_c1, pe_c2, fmt=lambda x: f"{x:.1f}")
                    + cmp_row("P/B", d1['pb'], d2['pb'], fmt=lambda x: f"{x:.2f}")
                    + cmp_row("EPS", d1['eps'], d2['eps'], fmt=lambda x: f"${x:.2f}")
                    + cmp_row("Beta", d1['beta'], d2['beta'], fmt=lambda x: f"{x:.2f}")
                    + cmp_row("Dividend", d1['div'], d2['div'], fmt=lambda x: f"{x*100:.2f}%" if x else "0%")
                    + cmp_row("RSI(14)", d1['rsi'], d2['rsi'], rsi_c1, rsi_c2, fmt=lambda x: f"{x:.1f}")
                    + cmp_row("MA20", d1['ma20'], d2['ma20'], fmt=lambda x: f"${x:.2f}")
                    + cmp_row("MA50", d1['ma50'], d2['ma50'], fmt=lambda x: f"${x:.2f}")
                    + cmp_row("52H Yuqori", d1['high52'], d2['high52'], fmt=lambda x: f"${x:.2f}")
                    + cmp_row("52H Past", d1['low52'], d2['low52'], fmt=lambda x: f"${x:.2f}")
                    + cmp_row("Maqsad narx", d1['target'], d2['target'], fmt=lambda x: f"${x:.2f}")
                    + cmp_row("Sektor", d1['sector'], d2['sector'])
                )

                st.markdown(f"""
<div class="custom-card" style="padding:0;overflow:hidden;">
<table class="compare-table">{rows_cmp}</table>
</div>""", unsafe_allow_html=True)

                # Winner summary
                scores = {d1['ticker']: 0, d2['ticker']: 0}
                for pair in [(d1['chg_1m'], d2['chg_1m']), (d2['pe'], d1['pe']), (d1['mc'], d2['mc'])]:
                    v1, v2 = pair
                    if v1 is not None and v2 is not None:
                        if v1 > v2:
                            scores[d1['ticker']] += 1
                        else:
                            scores[d2['ticker']] += 1

                winner = max(scores, key=scores.get)
                w_data = d1 if winner == d1['ticker'] else d2
                up = f"({(w_data['target']/w_data['current']-1)*100:+.1f}%)" if w_data['target'] else ""

                st.markdown(f"""
<div class="custom-card card-buy" style="margin-top:0.6rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#5a6480;">TAHLIL XULOSASI</div>
  <div style="margin-top:0.4rem;font-size:0.88rem;">
    Texnik ko'rsatkichlar bo'yicha <span style="color:#3fb950;font-weight:600;">{winner}</span>
    ustunlik qilmoqda. Maqsad narx:
    <span style="color:#58a6ff;">{f"${w_data['target']:.2f} {up}" if w_data['target'] else "N/A"}</span>
  </div>
  <div style="font-size:0.75rem;color:#5a6480;margin-top:0.4rem;">
    ⚠️ Bu faqat texnik tahlil. Investitsiyadan oldin tadqiqot o'tkazing.
  </div>
</div>""", unsafe_allow_html=True)