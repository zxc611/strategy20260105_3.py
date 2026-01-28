#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟 pythongo.base 模块 - 用于独立调试环境
无限易平台专有模块的临时替代品，用于在普通 Python 环境中运行和测试策略
"""

# Field装饰器模拟
class Field:
    def __init__(self, default=None, title=None, value=None, name=None, default_factory=None, **kwargs):
        self.default = default
        self.title = title
        if value is not None:
            self.value = value
        elif default is not None:
            self.value = default
        elif default_factory is not None:
            self.value = default_factory()
        else:
            self.value = None
        self.name = name

    def __get__(self, instance, owner):
        return self.value

# BaseParams基类
class BaseParams:
    def __init__(self):
        # 手动初始化字段值
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, Field):
                setattr(self, name, attr.value)

# BaseState基类
class BaseState:
    def __init__(self):
        # 手动初始化字段值
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, Field):
                setattr(self, name, attr.value)

# BaseStrategy基类
class BaseStrategy:
    def __init__(self):
        # 在真实平台中，params和state会通过Field装饰器自动处理
        # 在这里我们手动初始化
        self.params = Params() if 'Params' in globals() else None
        self.state = State() if 'State' in globals() else None

    def on_kline_generated(self, kline_data):
        pass
        
    def on_init(self, *args, **kwargs):
        pass
        
    def on_start(self, *args, **kwargs):
        pass
        
    def on_stop(self, *args, **kwargs):
        pass
        """K线生成时的回调方法，子类可以重写此方法"""
        pass

    def send_order(self, exchange, instrument_id, volume, price, order_direction, offset_flag="Open", **kwargs):
        """模拟发单方法"""
        import uuid
        order_id = str(uuid.uuid4())
        print(f"[Simulation-Base-Fixed] 发单成功: {exchange}.{instrument_id} {order_direction} {offset_flag} {volume}@{price} -> OrderID: {order_id}")
        return {
            "success": True,
            "order_id": order_id,
            "order_sys_id": order_id,
            "message": "模拟委托成功"
        }

    def output(self, msg, **kwargs):
        """模拟输出"""
        print(f"[Strategy] {msg}")

# 数据类模拟
class KLineData:
    def __init__(self, datetime=None, open=0.0, high=0.0, low=0.0, close=0.0, volume=0, pre_close=0.0):
        self.datetime = datetime
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.pre_close = pre_close

class TickData:
    def __init__(self):
        self.datetime = None
        self.last_price = 0.0
        self.volume = 0

class OrderData:
    def __init__(self):
        self.order_id = ""
        self.price = 0.0
        self.volume = 0

class TradeData:
    def __init__(self):
        self.trade_id = ""
        self.price = 0.0
        self.volume = 0

# K线生成器类
class KLineGenerator:
    def __init__(self, callback=None, real_time_callback=None, exchange=None, instrument_id=None, style=None):
        pass

    def push_history_data(self):
        pass

    def tick_to_kline(self, tick):
        pass

# 导出所有类和函数
__all__ = [
    'Field', 'BaseParams', 'BaseState', 'BaseStrategy',
    'KLineData', 'TickData', 'OrderData', 'TradeData',
    'KLineGenerator'
]
