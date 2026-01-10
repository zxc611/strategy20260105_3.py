"""
风险管理模块 - Risk Management
"""

from typing import Dict, List, Optional
from config import Config


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Config):
        """
        初始化风险管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.daily_pnl = 0.0
        self.positions = {}
        
    def check_position_limit(self, position_value: float, total_capital: float) -> bool:
        """
        检查持仓限制
        
        Args:
            position_value: 持仓价值
            total_capital: 总资金
            
        Returns:
            是否允许开仓
        """
        position_ratio = position_value / total_capital
        return position_ratio <= self.config.MAX_POSITION_SIZE
    
    def check_total_position_limit(self, total_position_value: float, total_capital: float) -> bool:
        """
        检查总持仓限制
        
        Args:
            total_position_value: 总持仓价值
            total_capital: 总资金
            
        Returns:
            是否允许开仓
        """
        total_ratio = total_position_value / total_capital
        return total_ratio <= self.config.MAX_TOTAL_POSITION
    
    def check_daily_loss_limit(self, current_pnl: float, total_capital: float) -> bool:
        """
        检查单日亏损限制
        
        Args:
            current_pnl: 当前盈亏
            total_capital: 总资金
            
        Returns:
            是否触发止损
        """
        loss_ratio = abs(current_pnl) / total_capital
        if current_pnl < 0 and loss_ratio >= self.config.MAX_DAILY_LOSS:
            return False  # 触发止损，不允许继续交易
        return True
    
    def calculate_position_size(self, entry_price: float, total_capital: float, 
                               volatility: float = 0.2) -> int:
        """
        计算持仓数量
        
        Args:
            entry_price: 入场价格
            total_capital: 总资金
            volatility: 波动率
            
        Returns:
            建议持仓数量
        """
        # 基于凯利公式的仓位管理
        max_position_value = total_capital * self.config.MAX_POSITION_SIZE
        
        # 考虑波动率调整
        volatility_adjustment = max(0.5, 1 - volatility)
        adjusted_position_value = max_position_value * volatility_adjustment
        
        position_size = int(adjusted_position_value / entry_price)
        
        return max(1, position_size)  # 至少1手
    
    def check_stop_loss(self, entry_price: float, current_price: float, 
                       position_type: str) -> bool:
        """
        检查止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_type: 持仓类型 (LONG/SHORT)
            
        Returns:
            是否需要止损
        """
        if position_type == 'LONG':
            loss_ratio = (entry_price - current_price) / entry_price
        else:  # SHORT
            loss_ratio = (current_price - entry_price) / entry_price
        
        return loss_ratio >= self.config.STOP_LOSS_PERCENT
    
    def check_take_profit(self, entry_price: float, current_price: float, 
                         position_type: str) -> bool:
        """
        检查止盈
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_type: 持仓类型 (LONG/SHORT)
            
        Returns:
            是否需要止盈
        """
        if position_type == 'LONG':
            profit_ratio = (current_price - entry_price) / entry_price
        else:  # SHORT
            profit_ratio = (entry_price - current_price) / entry_price
        
        return profit_ratio >= self.config.TAKE_PROFIT_PERCENT
    
    def validate_option_criteria(self, option_data: Dict) -> bool:
        """
        验证期权选择标准
        
        Args:
            option_data: 期权数据
            
        Returns:
            是否符合标准
        """
        # 检查持仓量
        if option_data.get('open_interest', 0) < self.config.MIN_OPEN_INTEREST:
            return False
        
        # 检查Delta范围
        delta = abs(option_data.get('delta', 0))
        if not (self.config.PREFERRED_DELTA_RANGE[0] <= delta <= self.config.PREFERRED_DELTA_RANGE[1]):
            return False
        
        return True
    
    def update_daily_pnl(self, pnl: float):
        """
        更新当日盈亏
        
        Args:
            pnl: 盈亏金额
        """
        self.daily_pnl += pnl
    
    def reset_daily_pnl(self):
        """重置当日盈亏"""
        self.daily_pnl = 0.0
