"""
数据获取模块 - Market Data Fetcher
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class MarketDataFetcher:
    """市场数据获取器"""
    
    def __init__(self):
        """初始化数据获取器"""
        self.data_cache = {}
        
    def get_option_chain(self, underlying_symbol: str, expiry_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取期权链数据
        
        Args:
            underlying_symbol: 标的代码
            expiry_date: 到期日期
            
        Returns:
            期权链数据DataFrame
        """
        # 模拟期权链数据
        option_chain = pd.DataFrame({
            'symbol': [f'{underlying_symbol}_CALL_{i}' for i in range(10)] + 
                     [f'{underlying_symbol}_PUT_{i}' for i in range(10)],
            'type': ['CALL'] * 10 + ['PUT'] * 10,
            'strike': list(range(95, 105)) + list(range(95, 105)),
            'expiry': [expiry_date or (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')] * 20,
            'bid': np.random.uniform(1, 10, 20),
            'ask': np.random.uniform(1.1, 10.5, 20),
            'last': np.random.uniform(1, 10, 20),
            'volume': np.random.randint(100, 10000, 20),
            'open_interest': np.random.randint(500, 50000, 20),
            'implied_volatility': np.random.uniform(0.15, 0.35, 20),
            'delta': np.random.uniform(-0.8, 0.8, 20),
            'gamma': np.random.uniform(0.01, 0.1, 20),
            'theta': np.random.uniform(-0.1, -0.01, 20),
            'vega': np.random.uniform(0.1, 0.5, 20),
        })
        
        return option_chain
    
    def get_underlying_price(self, symbol: str) -> float:
        """
        获取标的资产当前价格
        
        Args:
            symbol: 标的代码
            
        Returns:
            当前价格
        """
        # 模拟价格
        return 100.0 + np.random.uniform(-5, 5)
    
    def get_historical_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbol: 代码
            days: 天数
            
        Returns:
            历史数据DataFrame
        """
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # 生成模拟的历史价格数据
        prices = 100 + np.cumsum(np.random.randn(days) * 2)
        
        historical_data = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.uniform(-1, 1, days),
            'high': prices + np.random.uniform(0, 2, days),
            'low': prices - np.random.uniform(0, 2, days),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days)
        })
        
        return historical_data
    
    def get_market_indicators(self, symbol: str) -> Dict:
        """
        获取市场指标
        
        Args:
            symbol: 代码
            
        Returns:
            市场指标字典
        """
        historical_data = self.get_historical_data(symbol)
        
        # 计算技术指标
        close_prices = historical_data['close'].values
        
        # 移动平均
        ma20 = np.mean(close_prices[-20:])
        ma60 = np.mean(close_prices[-60:]) if len(close_prices) >= 60 else np.mean(close_prices)
        
        # RSI
        rsi = self._calculate_rsi(close_prices, period=14)
        
        # 波动率
        volatility = np.std(np.diff(close_prices) / close_prices[:-1]) * np.sqrt(252)
        
        return {
            'ma20': ma20,
            'ma60': ma60,
            'rsi': rsi,
            'volatility': volatility,
            'current_price': close_prices[-1]
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        计算RSI指标
        
        Args:
            prices: 价格数组
            period: 周期
            
        Returns:
            RSI值
        """
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
