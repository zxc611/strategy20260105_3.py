"""
仓位管理模块 - Position Manager
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


class Position:
    """持仓类"""
    
    def __init__(self, symbol: str, option_type: str, quantity: int, 
                 entry_price: float, strike: float, expiry: str):
        """
        初始化持仓
        
        Args:
            symbol: 期权代码
            option_type: 期权类型 (CALL/PUT)
            quantity: 数量
            entry_price: 入场价格
            strike: 行权价
            expiry: 到期日
        """
        self.symbol = symbol
        self.option_type = option_type
        self.quantity = quantity
        self.entry_price = entry_price
        self.strike = strike
        self.expiry = expiry
        self.entry_time = datetime.now()
        self.current_price = entry_price
        
    def update_price(self, price: float):
        """更新当前价格"""
        self.current_price = price
        
    def get_pnl(self) -> float:
        """计算盈亏"""
        return (self.current_price - self.entry_price) * self.quantity
    
    def get_pnl_percent(self) -> float:
        """计算盈亏百分比"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'option_type': self.option_type,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'strike': self.strike,
            'expiry': self.expiry,
            'entry_time': self.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'pnl': self.get_pnl(),
            'pnl_percent': self.get_pnl_percent()
        }


class PositionManager:
    """仓位管理器"""
    
    def __init__(self, initial_capital: float):
        """
        初始化仓位管理器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        
    def add_position(self, position: Position) -> bool:
        """
        添加持仓
        
        Args:
            position: 持仓对象
            
        Returns:
            是否成功添加
        """
        required_cash = position.entry_price * position.quantity
        
        if required_cash > self.cash:
            return False
        
        self.positions[position.symbol] = position
        self.cash -= required_cash
        
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[Dict]:
        """
        平仓
        
        Args:
            symbol: 期权代码
            exit_price: 平仓价格
            
        Returns:
            平仓记录
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.update_price(exit_price)
        
        # 计算盈亏
        pnl = position.get_pnl()
        
        # 回收资金
        self.cash += exit_price * position.quantity
        
        # 记录平仓
        closed_record = {
            **position.to_dict(),
            'exit_price': exit_price,
            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'holding_period': (datetime.now() - position.entry_time).days
        }
        
        self.closed_positions.append(closed_record)
        
        # 移除持仓
        del self.positions[symbol]
        
        return closed_record
    
    def update_positions(self, price_data: Dict[str, float]):
        """
        更新所有持仓价格
        
        Args:
            price_data: 价格数据字典 {symbol: price}
        """
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position.update_price(price_data[symbol])
    
    def get_total_value(self) -> float:
        """获取总资产价值"""
        position_value = sum(
            pos.current_price * pos.quantity 
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return self.get_total_value() - self.initial_capital
    
    def get_position_value(self) -> float:
        """获取持仓总价值"""
        return sum(
            pos.current_price * pos.quantity 
            for pos in self.positions.values()
        )
    
    def get_positions_summary(self) -> pd.DataFrame:
        """获取持仓摘要"""
        if not self.positions:
            return pd.DataFrame()
        
        return pd.DataFrame([pos.to_dict() for pos in self.positions.values()])
    
    def get_closed_positions_summary(self) -> pd.DataFrame:
        """获取已平仓摘要"""
        if not self.closed_positions:
            return pd.DataFrame()
        
        return pd.DataFrame(self.closed_positions)
    
    def get_performance_metrics(self) -> Dict:
        """获取绩效指标"""
        total_value = self.get_total_value()
        total_pnl = self.get_total_pnl()
        
        # 计算胜率
        if self.closed_positions:
            winning_trades = sum(1 for pos in self.closed_positions if pos['pnl'] > 0)
            win_rate = winning_trades / len(self.closed_positions) * 100
            
            # 平均盈亏
            avg_pnl = sum(pos['pnl'] for pos in self.closed_positions) / len(self.closed_positions)
        else:
            win_rate = 0
            avg_pnl = 0
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'position_value': self.get_position_value(),
            'total_pnl': total_pnl,
            'return_percent': (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'win_rate': win_rate,
            'avg_pnl_per_trade': avg_pnl
        }
