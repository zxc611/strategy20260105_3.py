"""
配置文件 - Options Trading Strategy Configuration
"""

class Config:
    """交易策略配置"""
    
    # 交易参数
    INITIAL_CAPITAL = 1000000  # 初始资金
    MAX_POSITION_SIZE = 0.2  # 单个持仓最大占比
    MAX_TOTAL_POSITION = 0.8  # 总持仓最大占比
    
    # 风险管理
    STOP_LOSS_PERCENT = 0.15  # 止损比例
    TAKE_PROFIT_PERCENT = 0.30  # 止盈比例
    MAX_DAILY_LOSS = 0.05  # 单日最大亏损比例
    
    # 期权参数
    MIN_DAYS_TO_EXPIRY = 7  # 最小到期天数
    MAX_DAYS_TO_EXPIRY = 90  # 最大到期天数
    PREFERRED_DELTA_RANGE = (0.3, 0.7)  # 优选Delta范围
    MIN_OPEN_INTEREST = 100  # 最小持仓量
    
    # 交易信号参数
    VOLATILITY_THRESHOLD = 0.25  # 波动率阈值
    TREND_PERIOD = 20  # 趋势周期
    RSI_PERIOD = 14  # RSI周期
    RSI_OVERBOUGHT = 70  # RSI超买阈值
    RSI_OVERSOLD = 30  # RSI超卖阈值
    
    # 执行参数
    MAX_SLIPPAGE = 0.02  # 最大滑点
    ORDER_TIMEOUT = 30  # 订单超时时间(秒)
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "options_strategy.log"
    
    # 交易时间
    MARKET_OPEN = "09:30:00"
    MARKET_CLOSE = "15:00:00"
