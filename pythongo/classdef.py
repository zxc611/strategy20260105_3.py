#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟 pythongo.classdef 模块 - 用于独立调试环境
无限易平台专有模块的临时替代品，用于在普通 Python 环境中运行和测试策略
"""

class KLineData:
    """K线数据类"""
    def __init__(self, datetime=None, open=0.0, high=0.0, low=0.0, close=0.0, volume=0, pre_close=0.0):
        self.datetime = datetime
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.pre_close = pre_close

class TickData:
    """Tick数据类"""
    def __init__(self):
        self.datetime = None
        self.last_price = 0.0
        self.volume = 0

class OrderData:
    """订单数据类"""
    def __init__(self):
        self.order_id = ""
        self.price = 0.0
        self.volume = 0

class TradeData:
    """成交数据类"""
    def __init__(self):
        self.trade_id = ""
        self.price = 0.0
        self.volume = 0

# 导出所有类
__all__ = ['KLineData', 'TickData', 'OrderData', 'TradeData']
