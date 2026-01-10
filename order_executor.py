"""
订单执行模块 - Order Executor
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging


class Order:
    """订单类"""
    
    def __init__(self, symbol: str, order_type: str, quantity: int, 
                 price: float, strategy: str):
        """
        初始化订单
        
        Args:
            symbol: 期权代码
            order_type: 订单类型 (BUY/SELL)
            quantity: 数量
            price: 价格
            strategy: 策略名称
        """
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.strategy = strategy
        self.status = 'PENDING'
        self.created_time = datetime.now()
        self.filled_time = None
        self.filled_price = None
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'price': self.price,
            'strategy': self.strategy,
            'status': self.status,
            'created_time': self.created_time.strftime('%Y-%m-%d %H:%M:%S'),
            'filled_time': self.filled_time.strftime('%Y-%m-%d %H:%M:%S') if self.filled_time else None,
            'filled_price': self.filled_price
        }


class OrderExecutor:
    """订单执行器"""
    
    def __init__(self, max_slippage: float = 0.02):
        """
        初始化订单执行器
        
        Args:
            max_slippage: 最大滑点
        """
        self.max_slippage = max_slippage
        self.orders: List[Order] = []
        self.logger = logging.getLogger(__name__)
        
    def create_order(self, symbol: str, order_type: str, quantity: int, 
                    price: float, strategy: str) -> Order:
        """
        创建订单
        
        Args:
            symbol: 期权代码
            order_type: 订单类型
            quantity: 数量
            price: 价格
            strategy: 策略名称
            
        Returns:
            订单对象
        """
        order = Order(symbol, order_type, quantity, price, strategy)
        self.orders.append(order)
        
        self.logger.info(f"创建订单: {order_type} {quantity} {symbol} @ {price}")
        
        return order
    
    def execute_order(self, order: Order, market_price: float) -> bool:
        """
        执行订单
        
        Args:
            order: 订单对象
            market_price: 市场价格
            
        Returns:
            是否执行成功
        """
        # 检查滑点
        if order.order_type == 'BUY':
            max_acceptable_price = order.price * (1 + self.max_slippage)
            if market_price > max_acceptable_price:
                order.status = 'REJECTED'
                self.logger.warning(f"订单拒绝: 买入价格 {market_price} 超过最大可接受价格 {max_acceptable_price}")
                return False
            filled_price = min(market_price, max_acceptable_price)
        else:  # SELL
            min_acceptable_price = order.price * (1 - self.max_slippage)
            if market_price < min_acceptable_price:
                order.status = 'REJECTED'
                self.logger.warning(f"订单拒绝: 卖出价格 {market_price} 低于最小可接受价格 {min_acceptable_price}")
                return False
            filled_price = max(market_price, min_acceptable_price)
        
        # 执行订单
        order.status = 'FILLED'
        order.filled_price = filled_price
        order.filled_time = datetime.now()
        
        self.logger.info(f"订单执行: {order.order_type} {order.quantity} {order.symbol} @ {filled_price}")
        
        return True
    
    def cancel_order(self, order: Order):
        """
        取消订单
        
        Args:
            order: 订单对象
        """
        if order.status == 'PENDING':
            order.status = 'CANCELLED'
            self.logger.info(f"订单取消: {order.symbol}")
    
    def get_pending_orders(self) -> List[Order]:
        """获取待执行订单"""
        return [order for order in self.orders if order.status == 'PENDING']
    
    def get_filled_orders(self) -> List[Order]:
        """获取已执行订单"""
        return [order for order in self.orders if order.status == 'FILLED']
    
    def get_order_statistics(self) -> Dict:
        """获取订单统计"""
        total_orders = len(self.orders)
        filled_orders = len([o for o in self.orders if o.status == 'FILLED'])
        rejected_orders = len([o for o in self.orders if o.status == 'REJECTED'])
        cancelled_orders = len([o for o in self.orders if o.status == 'CANCELLED'])
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'rejected_orders': rejected_orders,
            'cancelled_orders': cancelled_orders,
            'fill_rate': (filled_orders / total_orders * 100) if total_orders > 0 else 0
        }
