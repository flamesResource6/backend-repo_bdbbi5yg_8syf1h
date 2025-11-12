"""
Database Schemas

Custom schemas for the trading app. Each Pydantic model corresponds to a MongoDB
collection (lowercased class name).
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Symbol(BaseModel):
    """Tracked market symbols (e.g., BTCUSDT, AAPL)."""
    ticker: str = Field(..., description="Symbol ticker, e.g., BTCUSDT or AAPL")
    market: str = Field(..., description="Market type: crypto | stocks | forex")
    timeframe: str = Field(..., description="Default timeframe, e.g., 1h, 1d")
    notes: Optional[str] = Field(None, description="Optional notes")

class Strategy(BaseModel):
    """Saved strategy configurations."""
    name: str = Field(..., description="Strategy name")
    params: Dict[str, Any] = Field(..., description="Parameter dictionary")
    description: Optional[str] = Field(None, description="Optional description")

class Trade(BaseModel):
    """Paper trades executed by the engine."""
    asset: str
    timeframe: str
    side: str = Field(..., pattern="^(buy|sell)$")
    entry_time: int
    entry_price: float
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    qty: float = 1.0
    pnl: Optional[float] = None
    strategy_name: Optional[str] = None

class Backtest(BaseModel):
    """Backtest result summary and parameters."""
    asset: str
    timeframe: str
    strategy_name: str
    params: Dict[str, Any]
    trades: List[Dict[str, Any]]
    stats: Dict[str, Any]
    started_at: int
    ended_at: int

# Example schemas from template kept for reference (not used by app)
class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = Field(None, ge=0, le=120)
    is_active: bool = True

class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float = Field(..., ge=0)
    category: str
    in_stock: bool = True
