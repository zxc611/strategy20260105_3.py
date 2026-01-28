from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 移除所有第三方库依赖，只使用Python标准库

# 保持原有函数接口以实现向后兼容
def calculate_signal_strength(option: Dict, futures_close: float, futures_prev_close: float, option_width: float, options_data: List[Dict]) -> float:
    try:
        price_change = option['close'] - option['prev_close']
        expiry_date = datetime.strptime(option['expiry_date'], '%Y-%m-%d')
        current_date = datetime.strptime(option['date'], '%Y-%m-%d')
        remaining_days = (expiry_date - current_date).days
        if option['option_type'] == 'call':
            otm_degree = (option['strike_price'] - futures_prev_close) / futures_prev_close * 100
        else:
            otm_degree = (futures_prev_close - option['strike_price']) / futures_prev_close * 100
        avg_volume = _get_average_volume(option, options_data)
        volume_intensity = option['volume'] / max(1, avg_volume)
        signal_strength = (
            (price_change / option_width) * 0.4 + 
            volume_intensity * 0.3 + 
            (1 / remaining_days) * 0.2 + 
            (1 / max(1, abs(otm_degree))) * 0.1
        )
        return max(0, signal_strength)
    except Exception as e:
        return 0.0

def _get_average_volume(option: Dict, options_data: List[Dict]) -> float:
    try:
        option_code = option['code']
        # 过滤出相同期权代码的数据
        option_data = [item for item in options_data if item['code'] == option_code]
        # 过滤出日期早于当前期权日期的数据
        option_data = [item for item in option_data if item['date'] < option['date']]
        # 按日期降序排序并取前5条
        option_data.sort(key=lambda x: x['date'], reverse=True)
        option_data = option_data[:5]
        if len(option_data) > 0:
            total_volume = sum(item['volume'] for item in option_data)
            return total_volume / len(option_data)
        else:
            return option['volume']
    except Exception as e:
        return option['volume']

def generate_signal_for_option(option: Dict, futures_close: float, date: str, option_width: float, signal_strength: float, calculate_remaining_days_func, avg_volume: float, futures_data: List[Dict]) -> Optional[Dict]:
    try:
        remaining_days = calculate_remaining_days_func(option['expiry_date'], date)
        if remaining_days < 3 or remaining_days > 45:
            return None
        price_change = option['close'] - option['prev_close']
        if abs(price_change) <= option_width * 1.5:
            return None
        if option['volume'] <= avg_volume * 0.8:
            return None
        direction = 'buy' if price_change > 0 else 'sell'
        if direction == 'buy':
            target_price = option['close'] + (price_change * 2.0)
        else:
            target_price = option['close'] - (abs(price_change) * 2.0)
        if direction == 'buy':
            stop_loss_price = option['close'] - (option_width * 3.0)
        else:
            stop_loss_price = option['close'] + (option_width * 3.0)
        base_quantity = max(1, int(option['volume'] / max(1, avg_volume)))
        quantity = min(10, max(1, base_quantity))
        # 获取对应的期货代码
        futures_code = next(item['code'] for item in futures_data if item['date'] == date)
        signal = {
            'date': date,
            'option_code': option['code'],
            'option_type': option['option_type'],
            'strike_price': option['strike_price'],
            'expiry_date': option['expiry_date'],
            'entry_price': option['close'],
            'target_price': target_price,
            'stop_loss_price': stop_loss_price,
            'quantity': quantity,
            'direction': direction,
            'signal_strength': signal_strength,
            'futures_close': futures_close,
            'futures_code': futures_code,
            'remaining_days': remaining_days,
            'volume': option['volume'],
            'avg_volume': avg_volume,
            'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return signal
    except Exception as e:
        return None

def is_out_of_money(option_type: str, strike_price: float, futures_price: float) -> bool:
    """
    判断是否为虚值期权
    
    参数:
        option_type: 期权类型 ('call'或'put')
        strike_price: 行权价
        futures_price: 标的期货价格
        
    返回:
        bool: 是否为虚值期权
    """
    if strike_price is None or futures_price is None:
        return False
    
    if option_type not in ['call', 'put']:
        return False
    
    if option_type == 'call':
        return futures_price < strike_price
    else:  # put
        return futures_price > strike_price

def calculate_remaining_days(expiry_date: str, current_date: str) -> int:
    expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
    current = datetime.strptime(current_date, '%Y-%m-%d')
    return max(0, (expiry - current).days)

def calculate_option_width(futures_close: float, futures_prev_close: float) -> float:
    if futures_close is None or futures_prev_close is None:
        return 0.01
    if futures_close < 1000:
        width = 0.01
    elif futures_close < 5000:
        width = 0.05
    elif futures_close < 10000:
        width = 0.1
    elif futures_close < 50000:
        width = 0.5
    else:
        width = 1.0
    return width