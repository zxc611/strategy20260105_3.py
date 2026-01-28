# 虚拟 pythongo 包，用于策略调试
from .base import Field, BaseParams, BaseState, BaseStrategy
from .classdef import KLineData, TickData, OrderData, TradeData
from .utils import KLineGenerator

# 平台API模拟函数
def get_option_contracts():
    """模拟获取期权合约"""
    class MockContract:
        def __init__(self, symbol, underlying):
            self.symbol = symbol
            self.underlying = underlying
            self.exchange = "CFFEX"
            self.product = "option"
    
    return [
        MockContract("IO2503-P-3200", "IC2503"),
        MockContract("IO2503-C-3200", "IC2503"),
        MockContract("IO2503-P-3300", "IC2503"),
        MockContract("IO2503-C-3300", "IC2503")
    ]

# 导出所有类和函数
__all__ = [
    'Field', 'BaseParams', 'BaseState', 'BaseStrategy',
    'KLineData', 'TickData', 'OrderData', 'TradeData',
    'KLineGenerator', 'get_option_contracts'
]
