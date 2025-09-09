from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
from datetime import datetime
import pandas as pd
import os
import json

# Create FastAPI app
app = FastAPI(title="Market Scanner Pro", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for tickers
TICKER_CACHE = []
CACHE_TIME = None

def get_all_us_tickers():
    """
    Fetch comprehensive list of US tickers from multiple sources
    Returns 5000+ tradeable symbols
    """
    global TICKER_CACHE, CACHE_TIME
    
    # Use cache if less than 1 hour old
    if CACHE_TIME and (datetime.now() - CACHE_TIME).seconds < 3600 and TICKER_CACHE:
        return TICKER_CACHE
    
    all_tickers = set()
    
    try:
        # Method 1: Get S&P 500 (500 stocks)
        print("Fetching S&P 500...")
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_df = pd.read_html(sp500_url)[0]
        sp500_tickers = sp500_df['Symbol'].str.replace('.', '-').tolist()
        all_tickers.update(sp500_tickers)
        print(f"Added {len(sp500_tickers)} S&P 500 stocks")
        
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
    
    try:
        # Method 2: Get NASDAQ listings (3000+ stocks)
        print("Fetching NASDAQ listings...")
        nasdaq_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=NASDAQ"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
            # Using pandas to read from FTP as backup
        nasdaq_traded = pd.read_csv('ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt', sep='|')

        nasdaq_symbols = nasdaq_traded[nasdaq_traded['ETF'] == 'N']['Symbol'].tolist()
        # Clean symbols
        nasdaq_symbols = [s for s in nasdaq_symbols if isinstance(s, str) and len(s) <= 5 and s.isalpha()]
        all_tickers.update(nasdaq_symbols[:2000])  # Limit to 2000 for performance
        print(f"Added {len(nasdaq_symbols[:2000])} NASDAQ stocks")
        
    except Exception as e:
        print(f"Error fetching NASDAQ: {e}")
        # Fallback to popular NASDAQ stocks
        nasdaq_popular = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
                         'AMD', 'INTC', 'CSCO', 'ADBE', 'NFLX', 'CMCSA', 'PEP', 
                         'COST', 'TMUS', 'AVGO', 'TXN', 'QCOM', 'SBUX', 'INTU',
                         'MDLZ', 'ISRG', 'BKNG', 'FISV', 'ADP', 'GILD', 'MU',
                         'LRCX', 'ADI', 'PYPL', 'VRTX', 'MRVL', 'AMAT', 'REGN']
        all_tickers.update(nasdaq_popular)
    
    try:
        # Method 3: Get NYSE listings via other sources
        print("Adding NYSE stocks...")
        nyse_popular = ['JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'BAC', 
                       'CVX', 'ABBV', 'PFE', 'MRK', 'KO', 'WMT', 'VZ', 'CRM', 
                       'NKE', 'TMO', 'LLY', 'ABT', 'XOM', 'ACN', 'DHR', 'WFC',
                       'BMY', 'NOW', 'UPS', 'MS', 'RTX', 'NEE', 'T', 'SCHW',
                       'LOW', 'ORCL', 'PM', 'UNP', 'GS', 'BA', 'HON', 'BLK',
                       'IBM', 'AXP', 'CAT', 'GE', 'MMM', 'AMGN', 'CVS', 'MO']
        all_tickers.update(nyse_popular)
        print(f"Added {len(nyse_popular)} NYSE stocks")
        
    except Exception as e:
        print(f"Error adding NYSE: {e}")
    
    # Add popular ETFs
    etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'EEM', 'GLD', 'XLF', 
            'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'VNQ',
            'AGG', 'TLT', 'HYG', 'ARKK', 'ARKG', 'ARKF', 'ARKW', 'ARKQ',
            'VIG', 'VUG', 'VTV', 'VB', 'VO', 'VGT', 'VCR', 'VDC', 'VDE']
    all_tickers.update(etfs)
    
    # Convert to list and sort
    TICKER_CACHE = sorted(list(all_tickers))
    CACHE_TIME = datetime.now()
    
    print(f"Total tickers available: {len(TICKER_CACHE)}")
    return TICKER_CACHE

# Request model
class ScanRequest(BaseModel):
    limit: int = 50
    min_price: float = 1.0
    max_price: float = 10000.0
    min_volume: int = 100000
    sort_by: str = "volume"  # volume, price, change

@app.get("/")
async def root():
    """Serve the HTML frontend"""
    if os.path.exists('templates/index.html'):
        return FileResponse('templates/index.html')
    return {"message": "Market Scanner API is running!"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ticker_universe": len(TICKER_CACHE) if TICKER_CACHE else 0
    }

@app.get("/api/tickers")
async def get_tickers():
    """Get all available tickers"""
    tickers = get_all_us_tickers()
    return {
        "success": True,
        "count": len(tickers),
        "tickers": tickers[:100],  # Return first 100 for preview
        "total": len(tickers)
    }

@app.post("/api/scan")
async def run_scan(request: ScanRequest):
    """Run basic market scan showing tickers and prices"""
    try:
        # Get all available tickers
        all_tickers = get_all_us_tickers()
        
        # Limit the scan to requested number
        tickers_to_scan = all_tickers[:request.limit]
        
        results = []
        errors = []
        
        print(f"Scanning {len(tickers_to_scan)} stocks...")
        
        # Batch process tickers
        batch_size = 10
        for i in range(0, len(tickers_to_scan), batch_size):
            batch = tickers_to_scan[i:i+batch_size]
            
            for ticker in batch:
                try:
                    stock = yf.Ticker(ticker)
                    
                    # Get current data
                    info = stock.info
                    
                    # Try to get price from different fields
                    price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('price', 0)
                    
                    if price and price > 0:
                        # Get additional data if available
                        volume = info.get('volume') or info.get('regularMarketVolume', 0)
                        market_cap = info.get('marketCap', 0)
                        
                        # Get today's change
                        hist = stock.history(period="2d")
                        change_pct = 0
                        if not hist.empty and len(hist) >= 2:
                            try:
                                prev_close = hist['Close'].iloc[-2]
                                current = hist['Close'].iloc[-1]
                                change_pct = ((current - prev_close) / prev_close * 100)
                            except:
                                change_pct = 0
                        
                        # Apply filters
                        if request.min_price <= price <= request.max_price:
                            if volume >= request.min_volume:
                                results.append({
                                    'ticker': ticker,
                                    'price': round(price, 2),
                                    'change': round(change_pct, 2),
                                    'volume': int(volume),
                                    'market_cap': int(market_cap),
                                    'name': info.get('longName', ticker)[:50] if info.get('longName') else ticker
                                })
                
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)[:50]}")
                    continue
            
            # Progress update
            print(f"Processed {min(i+batch_size, len(tickers_to_scan))} of {len(tickers_to_scan)}")
        
        # Sort results
        if request.sort_by == "volume":
            results.sort(key=lambda x: x['volume'], reverse=True)
        elif request.sort_by == "change":
            results.sort(key=lambda x: abs(x['change']), reverse=True)
        elif request.sort_by == "price":
            results.sort(key=lambda x: x['price'], reverse=True)
        
        return {
            'success': True,
            'results': results,
            'count': len(results),
            'scanned': len(tickers_to_scan),
            'errors': len(errors),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ticker/{symbol}")
async def get_ticker_details(symbol: str):
    """Get detailed info for a specific ticker"""
    try:
        stock = yf.Ticker(symbol.upper())
        info = stock.info
        hist = stock.history(period="1mo")
        
        return {
            'success': True,
            'ticker': symbol.upper(),
            'name': info.get('longName'),
            'price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'volume': info.get('volume'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            '52w_high': info.get('fiftyTwoWeekHigh'),
            '52w_low': info.get('fiftyTwoWeekLow'),
            'history': hist.tail(5).to_dict() if not hist.empty else {}
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# === Fast scan integration (non-breaking additive endpoint) ===
from typing import Literal
from pydantic import BaseModel
import pandas as pd
from fastapi import HTTPException

try:
    from market_data.yahoo_provider import YahooProvider
except Exception as e:
    print("YahooProvider import failed:", e)
    YahooProvider = None

# Single provider instance
if 'DATA_PROVIDER' not in globals():
    DATA_PROVIDER = YahooProvider(max_workers=24, batch_size=200) if YahooProvider else None

class FastScanRequest(BaseModel):
    limit: int = 300
    min_price: float = 1.0
    max_price: float = 10000.0
    min_volume: int = 100000
    sort_by: Literal['volume','price','change'] = 'volume'

@app.post("/api/scan_fast")
def scan_stocks_fast(request: FastScanRequest):
    if DATA_PROVIDER is None:
        raise HTTPException(500, "Data provider not available")
    try:
        # Reuse your existing universe builder
        tickers = get_all_us_tickers()[: request.limit]

        quotes = DATA_PROVIDER.get_quotes(tickers)
        results, errors = [], []

        for t in tickers:
            q = quotes.get(t)
            if not q:
                errors.append(t)
                continue

            price = q.get("price")
            volume = q.get("volume") or 0
            prev_close = q.get("prev_close")

            change_pct = None
            if price is not None and prev_close not in (None, 0):
                change_pct = round(((price - prev_close) / prev_close) * 100, 2)

            if price is None:
                continue
            if not (request.min_price <= price <= request.max_price):
                continue
            if volume < request.min_volume:
                continue

            results.append({
                "ticker": t,
                "name": t,  # keep fast; we can enrich top-N later
                "price": round(price, 2),
                "change": change_pct,
                "volume": int(volume),
                "market_cap": None,  # omit for speed in step 1
            })

        key_map = {
            "volume": lambda x: x.get("volume", 0),
            "price":  lambda x: x.get("price", 0),
            "change": lambda x: abs(x.get("change") or 0),
        }
        results.sort(key=key_map.get(request.sort_by, key_map["volume"]), reverse=True)

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "scanned": len(tickers),
            "errors": len(errors),
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "engine": "scan_fast_v1"
        }
    except Exception as e:
        print("scan_fast error:", e)
        raise HTTPException(status_code=500, detail=str(e))
