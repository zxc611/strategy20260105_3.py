"""
信号生成模块 - Trading Signal Generator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config import Config
from data_fetcher import MarketDataFetcher


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(self, config: Config, data_fetcher: MarketDataFetcher):
        """
        初始化信号生成器
        
        Args:
            config: 配置对象
            data_fetcher: 数据获取器
        """
        self.config = config
        self.data_fetcher = data_fetcher
        
    def generate_signals(self, symbol: str) -> Dict:
        """
        生成交易信号
        
        Args:
            symbol: 标的代码
            
        Returns:
            信号字典
        """
        # 获取市场指标
        indicators = self.data_fetcher.get_market_indicators(symbol)
        
        # 趋势信号
        trend_signal = self._analyze_trend(indicators)
        
        # 动量信号
        momentum_signal = self._analyze_momentum(indicators)
        
        # 波动率信号
        volatility_signal = self._analyze_volatility(indicators)
        
        # 综合信号
        combined_signal = self._combine_signals(trend_signal, momentum_signal, volatility_signal)
        
        return {
            'symbol': symbol,
            'trend_signal': trend_signal,
            'momentum_signal': momentum_signal,
            'volatility_signal': volatility_signal,
            'combined_signal': combined_signal,
            'signal_strength': self._calculate_signal_strength(trend_signal, momentum_signal, volatility_signal),
            'recommended_strategy': self._recommend_option_strategy(combined_signal, indicators)
        }
    
    def _analyze_trend(self, indicators: Dict) -> str:
        """
        分析趋势
        
        Args:
            indicators: 市场指标
            
        Returns:
            趋势信号 (BULLISH/BEARISH/NEUTRAL)
        """
        ma20 = indicators.get('ma20', 0)
        ma60 = indicators.get('ma60', 0)
        current_price = indicators.get('current_price', 0)
        
        if current_price > ma20 > ma60:
            return 'BULLISH'
        elif current_price < ma20 < ma60:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _analyze_momentum(self, indicators: Dict) -> str:
        """
        分析动量
        
        Args:
            indicators: 市场指标
            
        Returns:
            动量信号 (STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL)
        """
        rsi = indicators.get('rsi', 50)
        
        if rsi > self.config.RSI_OVERBOUGHT:
            return 'STRONG_SELL'
        elif rsi > 60:
            return 'SELL'
        elif rsi < self.config.RSI_OVERSOLD:
            return 'STRONG_BUY'
        elif rsi < 40:
            return 'BUY'
        else:
            return 'NEUTRAL'
    
    def _analyze_volatility(self, indicators: Dict) -> str:
        """
        分析波动率
        
        Args:
            indicators: 市场指标
            
        Returns:
            波动率信号 (HIGH/NORMAL/LOW)
        """
        volatility = indicators.get('volatility', 0)
        
        if volatility > self.config.VOLATILITY_THRESHOLD * 1.5:
            return 'HIGH'
        elif volatility < self.config.VOLATILITY_THRESHOLD * 0.5:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def _combine_signals(self, trend: str, momentum: str, volatility: str) -> str:
        """
        综合多个信号
        
        Args:
            trend: 趋势信号
            momentum: 动量信号
            volatility: 波动率信号
            
        Returns:
            综合信号 (BUY/SELL/HOLD)
        """
        score = 0
        
        # 趋势权重
        if trend == 'BULLISH':
            score += 2
        elif trend == 'BEARISH':
            score -= 2
        
        # 动量权重
        if momentum in ['STRONG_BUY', 'BUY']:
            score += 1
        elif momentum in ['STRONG_SELL', 'SELL']:
            score -= 1
        
        # 波动率调整
        if volatility == 'HIGH':
            score = score * 0.7  # 高波动率降低信号强度
        
        if score >= 2:
            return 'BUY'
        elif score <= -2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_signal_strength(self, trend: str, momentum: str, volatility: str) -> float:
        """
        计算信号强度
        
        Args:
            trend: 趋势信号
            momentum: 动量信号
            volatility: 波动率信号
            
        Returns:
            信号强度 (0-1)
        """
        strength = 0.5  # 基础强度
        
        # 趋势贡献
        if trend in ['BULLISH', 'BEARISH']:
            strength += 0.2
        
        # 动量贡献
        if momentum in ['STRONG_BUY', 'STRONG_SELL']:
            strength += 0.2
        elif momentum in ['BUY', 'SELL']:
            strength += 0.1
        
        # 波动率调整
        if volatility == 'NORMAL':
            strength += 0.1
        
        return min(1.0, strength)
    
    def _recommend_option_strategy(self, signal: str, indicators: Dict) -> str:
        """
        推荐期权策略
        
        Args:
            signal: 综合信号
            indicators: 市场指标
            
        Returns:
            推荐策略
        """
        volatility = indicators.get('volatility', 0)
        
        if signal == 'BUY':
            if volatility < self.config.VOLATILITY_THRESHOLD * 0.8:
                return 'LONG_CALL'  # 买入看涨期权
            else:
                return 'BULL_CALL_SPREAD'  # 牛市看涨价差
        elif signal == 'SELL':
            if volatility < self.config.VOLATILITY_THRESHOLD * 0.8:
                return 'LONG_PUT'  # 买入看跌期权
            else:
                return 'BEAR_PUT_SPREAD'  # 熊市看跌价差
        else:
            if volatility > self.config.VOLATILITY_THRESHOLD * 1.2:
                return 'STRADDLE'  # 跨式策略
            else:
                return 'IRON_CONDOR'  # 铁秃鹰策略
