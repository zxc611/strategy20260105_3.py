"""
工具函数模块 - Utilities
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List
import os


def setup_logger(log_file: str = 'options_strategy.log', log_level: str = 'INFO') -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_file: 日志文件名
        log_level: 日志级别
        
    Returns:
        日志记录器
    """
    logger = logging.getLogger('OptionsStrategy')
    logger.setLevel(getattr(logging, log_level))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level))
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def calculate_days_to_expiry(expiry_date: str) -> int:
    """
    计算到期天数
    
    Args:
        expiry_date: 到期日期 (YYYY-MM-DD)
        
    Returns:
        到期天数
    """
    expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
    today = datetime.now()
    return (expiry - today).days


def is_market_open(current_time: datetime = None) -> bool:
    """
    检查市场是否开盘
    
    Args:
        current_time: 当前时间
        
    Returns:
        是否开盘
    """
    if current_time is None:
        current_time = datetime.now()
    
    # 检查是否为工作日
    if current_time.weekday() >= 5:  # 周六、周日
        return False
    
    # 检查时间
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
    
    return market_open <= current_time <= market_close


def format_currency(amount: float) -> str:
    """
    格式化货币
    
    Args:
        amount: 金额
        
    Returns:
        格式化的字符串
    """
    return f"¥{amount:,.2f}"


def format_percent(value: float) -> str:
    """
    格式化百分比
    
    Args:
        value: 数值
        
    Returns:
        格式化的字符串
    """
    return f"{value:.2f}%"


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.03) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率列表
        risk_free_rate: 无风险利率
        
    Returns:
        夏普比率
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    import numpy as np
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    return (avg_return - risk_free_rate) / std_return


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    计算最大回撤
    
    Args:
        equity_curve: 净值曲线
        
    Returns:
        最大回撤
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    import numpy as np
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0
        max_dd = max(max_dd, drawdown)
    
    return max_dd


def save_trade_history(trades: List[Dict], filename: str = 'trade_history.csv'):
    """
    保存交易历史
    
    Args:
        trades: 交易记录列表
        filename: 文件名
    """
    import pandas as pd
    
    df = pd.DataFrame(trades)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"交易历史已保存到 {filename}")


def load_trade_history(filename: str = 'trade_history.csv') -> List[Dict]:
    """
    加载交易历史
    
    Args:
        filename: 文件名
        
    Returns:
        交易记录列表
    """
    import pandas as pd
    
    if not os.path.exists(filename):
        return []
    
    df = pd.read_csv(filename, encoding='utf-8-sig')
    return df.to_dict('records')
