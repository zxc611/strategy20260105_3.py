#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟 pythongo.core 模块 - 用于独立调试环境
无限易平台专有模块的临时替代品，用于在普通 Python 环境中运行和测试策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class MarketCenter:
    """市场数据中心模拟类"""
    
    def __init__(self):
        print("MarketCenter 初始化 - 虚拟实现")
        pass
        
    def get_kline_data(self, exchange, instrument_id, style, start_time, end_time):
        """获取K线数据的虚拟实现"""
        print(f"MarketCenter.get_kline_data 被调用: 合约 {instrument_id}, 周期 {style}, 时间 {start_time} 到 {end_time}")
        
        # 创建模拟的K线数据
        class MockKLine:
            def __init__(self, datetime_val, open_price, high, low, close, volume, open_interest, pre_close):
                self.datetime = datetime_val
                self.open = open_price
                self.high = high
                self.low = low
                self.close = close
                self.volume = volume
                self.open_interest = open_interest
                self.pre_close = pre_close
        
        # 生成模拟数据
        kline_data = []
        current_time = start_time
        interval = self._get_interval(style)
        
        while current_time <= end_time:
            # 随机生成价格数据
            base_price = 3400.0
            variation = np.random.uniform(-5, 5)
            open_price = base_price + variation
            close_price = open_price + np.random.uniform(-3, 3)
            high_price = max(open_price, close_price) + np.random.uniform(0, 2)
            low_price = min(open_price, close_price) - np.random.uniform(0, 2)
            
            kline = MockKLine(
                current_time,
                open_price,
                high_price,
                low_price,
                close_price,
                np.random.randint(100, 1000),
                np.random.randint(1000, 5000),
                base_price - np.random.uniform(0, 10)
            )
            
            kline_data.append(kline)
            current_time += interval
            
        print(f"MarketCenter.get_kline_data 返回 {len(kline_data)} 条K线数据")
        return kline_data
        
    def _get_interval(self, style):
        """获取K线周期对应的时间间隔"""
        if style == 'M1':
            return timedelta(minutes=1)
        elif style == 'M5':
            return timedelta(minutes=5)
        elif style == 'M15':
            return timedelta(minutes=15)
        elif style == 'H1':
            return timedelta(hours=1)
        elif style == 'D1':
            return timedelta(days=1)
        else:
            return timedelta(minutes=1)
            
    def get_contract_info(self, exchange, instrument_id):
        """获取合约信息的虚拟实现"""
        print(f"MarketCenter.get_contract_info 被调用: 合约 {instrument_id}")
        return {
            'instrument_id': instrument_id,
            'exchange': exchange,
            'name': f"模拟合约 {instrument_id}",
            'product_type': 'OPTION' if 'C' in instrument_id or 'P' in instrument_id else 'FUTURES',
            'strike_price': 3400.0 if 'C' in instrument_id or 'P' in instrument_id else None,
            'option_type': 'CALL' if 'C' in instrument_id else 'PUT' if 'P' in instrument_id else None,
            'expiry_date': '20251225' if '2512' in instrument_id else '20260125',
            'underlying': instrument_id.replace('IO', 'IF') if instrument_id.startswith('IO') else instrument_id
        }

    def get_instruments_by_product(self, exchange, product_id):
        """Mock implementation of get_instruments_by_product"""
        print(f"MarketCenter.get_instruments_by_product called for {exchange} {product_id}")
        
        class MockInstrument:
            def __init__(self, instrument_id, product_id, exchange_id):
                self.instrument_id = instrument_id
                self.product_id = product_id
                self.exchange_id = exchange_id
                # Add commonly accessed attributes
                self.exchange = exchange_id
                self.product = product_id
                self.name = instrument_id
                self.multiple = 10
                self.price_tick = 1.0
                self.strike_price = 0.0
                self.option_type = None
                self.underlying_id = ""
                
                # Simple parsing for mock option logic
                if "C" in instrument_id or "P" in instrument_id:
                    # Very rough parsing, assuming suffix is price
                    import re
                    m = re.search(r"([CP])(\d+)$", instrument_id)
                    if m:
                        # Use standard type indicators compatible with strategy
                        # Strategy expects "C"/"P" in option_type validation or "1"/"2" in OptionType dictionary check
                        self.option_type = "C" if m.group(1) == "C" else "P"
                        self.OptionType = "1" if m.group(1) == "C" else "2"  # Redundant field for strategy compatibility
                        self.strike_price = float(m.group(2))
                        # Assume underlying is prefix
                        self.underlying_id = instrument_id[:m.start()]

        pid = product_id.upper()
        res = []
        
        # Generate underlying futures (e.g., CU2603)
        future_id = f"{pid}2603"
        res.append(MockInstrument(future_id, pid, exchange))
        
        # Generate options (e.g., CU2603C65000)
        # We need strikes around the mock price 3400 set in get_kline_data to make them ATM/OTM
        # But for CU usually price is 60000+, so 3400 is weird. 
        # However, get_kline_data uses 3400. So let's use strikes around 3400 to match data.
        base_strike = 3400
        for i in range(-5, 6):
            strike = base_strike + i * 100
            # Format depends on exchange, but strategy uses contains C/P check
            # Construct simplified ID: CU2603C3400
            res.append(MockInstrument(f"{future_id}C{strike}", pid, exchange))
            res.append(MockInstrument(f"{future_id}P{strike}", pid, exchange))
            
        return res

    def get_real_options_data(self, date, futures_code):
        """获取真实期权数据的虚拟实现 - 直接读取本地CSV文件"""
        try:
            # 构造CSV文件路径
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', f'options_{futures_code}_{date[:6]}.csv')
            print(f"MarketCenter.get_real_options_data: 尝试读取文件: {file_path}")
            
            if os.path.exists(file_path):
                # 读取CSV文件
                options_data = pd.read_csv(file_path)
                print(f"MarketCenter.get_real_options_data: 成功读取文件，共 {len(options_data)} 条记录")
                
                # 格式化日期字段
                options_data['date'] = options_data['date'].astype(str)
                if '-' not in date:
                    options_data['date'] = options_data['date'].str.replace('-', '')
                
                # 过滤指定日期的数据
                filtered_data = options_data[options_data['date'] == date]
                print(f"MarketCenter.get_real_options_data: 过滤后数据条数: {len(filtered_data)}")
                
                if not filtered_data.empty:
                    return filtered_data
            
            print("MarketCenter.get_real_options_data: 未找到本地CSV文件或无对应日期数据")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"MarketCenter.get_real_options_data: 获取期权数据失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

# 导出所有类和函数
__all__ = ['MarketCenter']
