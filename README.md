# Stock AI Agent

An intelligent stock analysis assistant powered by AI, combining real-time market data with advanced technical indicators to provide actionable trading insights.

## 🎯 Features

- **Real-time Stock Data**: Fetch current and historical stock data using yfinance
- **Technical Analysis**: Calculate and analyze key indicators:
  - RSI (Relative Strength Index)
  - Moving Averages (20-day, 50-day)
  - Bollinger Bands
  - Volume Analysis
- **AI-Powered Insights**: OpenAI-powered agent that interprets technical data and provides investment recommendations
- **Interactive Dashboard**: Beautiful Streamlit interface with dark theme for comfortable viewing
- **Professional Visualizations**: Interactive charts powered by Plotly
- **Multi-Symbol Support**: Analyze multiple stocks simultaneously

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Active internet connection (for real-time stock data)

## 🚀 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Stock AI Agent"
   ```

2. **Create a virtual environment (if not already created):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install:
   ```bash
   pip install streamlit yfinance plotly langchain langchain-openai python-dotenv
   ```

5. **Set up environment variables:**
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🎮 Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

3. **Interact with the AI Assistant:**
   - Enter stock tickers (e.g., AAPL, GOOGL, MSFT)
   - Ask questions about stock performance and analysis
   - View technical indicators and visualizations
   - Get AI-powered recommendations

## 📁 Project Structure

```
Stock AI Agent/
├── app.py              # Main Streamlit application
├── tools.py            # LangChain tools for stock analysis
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (create this)
├── .gitignore         # Git ignore rules
└── venv/              # Virtual environment
```

## 🔧 Core Components

### `app.py`
The main application file that:
- Sets up the Streamlit interface with dark theme
- Initializes the OpenAI language model
- Creates the agent executor with stock analysis tools
- Manages chat history and user interactions
- Displays results and visualizations

### `tools.py`
Implements LangChain tools including:
- **get_stock_analysis()**: Comprehensive technical analysis
  - Current price and price change
  - Technical indicators (RSI, Moving Averages, Bollinger Bands)
  - Volume analysis
  - Price-to-Earnings ratio
  - Market sentiment indicators

## 📊 Technical Indicators Explained

- **RSI (Relative Strength Index)**: Measures momentum (0-100 scale)
- **Moving Averages**: Track price trends over specific periods
- **Bollinger Bands**: Show volatility and potential reversal points
- **Volume Ratio**: Compares current volume to average

## 🔐 Security Notes

- Keep your `.env` file out of version control
- Never share your OpenAI API key
- Use a `.gitignore` file to exclude `.env` and `venv/` directories

## 🐛 Troubleshooting

**Issue**: "ModuleNotFoundError" when running the app
- **Solution**: Ensure virtual environment is activated and all dependencies are installed

**Issue**: "Invalid OpenAI API key"
- **Solution**: Check your `.env` file and verify your API key is correct

**Issue**: Stock ticker not found
- **Solution**: Verify the ticker symbol is correct (e.g., AAPL, not APPLE)

## 📦 Dependencies

- `streamlit` - Web application framework
- `yfinance` - Yahoo Finance API wrapper
- `plotly` - Interactive visualizations
- `langchain` - LLM orchestration framework
- `langchain-openai` - OpenAI integration
- `pandas` - Data manipulation
- `python-dotenv` - Environment variable management

## 🔄 Future Enhancements

- Real-time price alerts
- Portfolio tracking
- Multi-timeframe analysis
- Custom indicator combinations
- Export analysis reports
- User authentication and saved preferences

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Support

For issues or questions, please check the troubleshooting section or verify your setup matches the installation requirements.
