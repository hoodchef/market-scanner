#!/usr/bin/env python3
"""
MARKET SCANNER BACKEND SERVER
============================
FastAPI backend for the web-based market scanner
Provides REST API and WebSocket connections for real-time updates
"""
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
import uvicorn
from typing import Dict, List, Optional
from pydantic import BaseModel
import aiohttp
import concurrent.futures
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Market Scanner API", version="1.0.0")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ScanRequest(BaseModel):
    exchange: str = "all"
    min_price: float = 5.0
    max_price: float = 500.0
    min_volume: int = 1000000
    timeframe: str = "1d"
    patterns: List[str] = ["momentum", "breakout", "volume", "reversal", "strength"]

class ScanResult(BaseModel):
    ticker: str
    price: float
    change: float
    volume: int
    patterns: List[str]
    strength: float
    timestamp: str

# Global variables
active_connections: List[WebSocket] = []
scanner_instance = None
cache = {}
scan_running = False

class MarketScanner:
    """Core scanner logic from previous implementation"""
    
    def __init__(self):
        self.spy_data = None
        self.last_scan = None
        
    async def get_all_tickers(self, exchange: str = "all") -> List[str]:
        """Fetch tickers based on exchange selection"""
        tickers = []
        
        try:
            if exchange in ["all", "nasdaq"]:
                # NASDAQ tickers
                nasdaq_100 = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'GOOGL', 
                             'GOOG', 'AVGO', 'PEP', 'COST', 'ADBE', 'CSCO', 'CMCSA', 
                             'INTC', 'AMD', 'NFLX', 'TMUS', 'QCOM', 'TXN']
                tickers.extend(nasdaq_100)
            
            if exchange in ["all", "nyse"]:
                # NYSE major stocks
                nyse_stocks = ['JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 
                              'DIS', 'BAC', 'XOM', 'CVX', 'ABBV', 'KO', 'PFE']
                tickers.extend(nyse_stocks)
            
            # Add ETFs
            etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'XLF', 'XLK', 'XLE']
            tickers.extend(etfs)
            
            # For demo, limit to manageable number
            return list(set(tickers))[:100]
            
        except Exception as e:
            print(f"Error fetching tickers: {e}")
            return tickers[:50] if tickers else self.get_default_tickers()
    
    def get_default_tickers(self) -> List[str]:
        """Fallback ticker list"""
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'GOOGL', 
                'AMD', 'NFLX', 'JPM', 'V', 'JNJ', 'WMT', 'SPY', 'QQQ']
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if len(df) < 20:
            return df
        
        # Basic indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Price change
        df['Change_Pct'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def scan_ticker(self, ticker: str, params: ScanRequest) -> Optional[Dict]:
        """Scan individual ticker for patterns"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1mo", interval=params.timeframe)
            
            if df.empty or len(df) < 5:
                return None
            
            df = self.calculate_indicators(df)
            latest = df.iloc[-1]
            
            # Check price and volume filters
            if not (params.min_price <= latest['Close'] <= params.max_price):
                return None
            if latest['Volume'] < params.min_volume:
                return None
            
            # Pattern detection
            patterns_found = []
            strength = 0
            
            # Momentum pattern
            if "momentum" in params.patterns:
                if len(df) > 1 and abs(df['Change_Pct'].iloc[-1]) > 3:
                    patterns_found.append("momentum")
                    strength += 20
            
            # Volume surge pattern
            if "volume" in params.patterns:
                if 'Volume_Avg' in df.columns and latest['Volume'] > df['Volume_Avg'].iloc[-1] * 2:
                    patterns_found.append("volume")
                    strength += 20
            
            # Breakout pattern
            if "breakout" in params.patterns:
                if 'SMA_20' in df.columns and latest['Close'] > df['SMA_20'].iloc[-1] * 1.02:
                    patterns_found.append("breakout")
                    strength += 25
            
            # Reversal pattern
            if "reversal" in params.patterns:
                if 'RSI' in df.columns:
                    if latest['RSI'] < 30:
                        patterns_found.append("reversal")
                        strength += 15
                    elif latest['RSI'] > 70:
                        patterns_found.append("reversal")
                        strength += 15
            
            # Relative strength pattern
            if "strength" in params.patterns and self.spy_data is not None:
                ticker_return = (latest['Close'] / df['Close'].iloc[0] - 1) * 100
                spy_return = (self.spy_data['Close'].iloc[-1] / self.spy_data['Close'].iloc[0] - 1) * 100
                if ticker_return > spy_return + 5:
                    patterns_found.append("strength")
                    strength += 30
            
            if patterns_found:
                return {
                    "ticker": ticker,
                    "price": round(latest['Close'], 2),
                    "change": round(df['Change_Pct'].iloc[-1], 2) if 'Change_Pct' in df.columns else 0,
                    "volume": int(latest['Volume']),
                    "patterns": patterns_found,
                    "strength": min(100, strength + 40),  # Base strength
                    "timestamp": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
            return None
    
    async def run_scan(self, params: ScanRequest) -> List[Dict]:
        """Run complete market scan"""
        # Get SPY data for relative strength
        spy = yf.Ticker("SPY")
        self.spy_data = spy.history(period="1mo", interval=params.timeframe)
        
        # Get tickers to scan
        tickers = await self.get_all_tickers(params.exchange)
        
        # Scan all tickers asynchronously
        results = []
        for ticker in tickers:
            result = await self.scan_ticker(ticker, params)
            if result:
                results.append(result)
                # Send live update via WebSocket
                await notify_clients({
                    "type": "scan_progress",
                    "ticker": ticker,
                    "result": result
                })
        
        # Sort by strength
        results.sort(key=lambda x: x['strength'], reverse=True)
        
        return results

# WebSocket manager
async def notify_clients(data: Dict):
    """Send updates to all connected WebSocket clients"""
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)

# API Routes
@app.get("/")
async def root():
    """Serve the HTML frontend"""
    return FileResponse('templates/index.html')

@app.post("/api/scan")
async def run_market_scan(request: ScanRequest):
    """Run a market scan with specified parameters"""
    global scan_running, scanner_instance
    
    if scan_running:
        raise HTTPException(status_code=429, detail="Scan already in progress")
    
    try:
        scan_running = True
        
        if not scanner_instance:
            scanner_instance = MarketScanner()
        
        # Notify clients scan is starting
        await notify_clients({
            "type": "scan_started",
            "timestamp": datetime.now().isoformat()
        })
        
        # Run the scan
        results = await scanner_instance.run_scan(request)
        
        # Cache results
        cache['last_scan'] = {
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'params': request.dict()
        }
        
        # Notify clients scan is complete
        await notify_clients({
            "type": "scan_complete",
            "results": results[:10],  # Send top 10
            "total": len(results),
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    
    finally:
        scan_running = False

@app.get("/api/tickers/{exchange}")
async def get_tickers(exchange: str):
    """Get list of tickers for specific exchange"""
    scanner = MarketScanner()
    tickers = await scanner.get_all_tickers(exchange)
    return {"tickers": tickers, "count": len(tickers)}

@app.get("/api/ticker/{symbol}")
async def get_ticker_data(symbol: str):
    """Get detailed data for a specific ticker"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        history = stock.history(period="1mo")
        
        return {
            "symbol": symbol,
            "name": info.get('longName', symbol),
            "price": info.get('currentPrice', 0),
            "change": info.get('regularMarketChangePercent', 0),
            "volume": info.get('volume', 0),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "history": history.tail(30).to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")

@app.get("/api/results/latest")
async def get_latest_results():
    """Get the latest scan results from cache"""
    if 'last_scan' in cache:
        return cache['last_scan']
    return {"results": [], "message": "No scan results available"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to market scanner",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            
            # Handle different message types
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif message.get("type") == "subscribe":
                # Handle subscription to specific tickers
                ticker = message.get("ticker")
                # Implement ticker-specific updates
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(active_connections),
        "scan_running": scan_running
    }

# Background task for live monitoring
async def live_monitoring():
    """Background task that runs continuous monitoring"""
    scanner = MarketScanner()
    
    while True:
        try:
            # Run mini scan every 5 minutes during market hours
            now = datetime.now()
            if 9 <= now.hour < 16 and now.weekday() < 5:  # Market hours
                
                # Quick scan of top movers
                params = ScanRequest(
                    patterns=["momentum", "volume"],
                    min_volume=5000000
                )
                
                # Scan subset of active stocks
                hot_tickers = ['NVDA', 'TSLA', 'AMD', 'AAPL', 'SPY', 'QQQ']
                
                for ticker in hot_tickers:
                    result = await scanner.scan_ticker(ticker, params)
                    if result and result['patterns']:
                        # Send alert to all connected clients
                        await notify_clients({
                            "type": "alert",
                            "ticker": result['ticker'],
                            "patterns": result['patterns'],
                            "price": result['price'],
                            "change": result['change'],
                            "message": f"{ticker} showing {', '.join(result['patterns'])} pattern",
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Wait 5 minutes before next check
            await asyncio.sleep(300)
            
        except Exception as e:
            print(f"Live monitoring error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup"""
    asyncio.create_task(live_monitoring())

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
