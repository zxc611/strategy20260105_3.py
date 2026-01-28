"""
期权交易策略智能体简化系统 - 无PyTorch版本
文件名：option_trading_agent_simple.py
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import json
import pickle
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import random
import warnings
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

"""
第二部分：日志配置
"""

def setup_logging(log_dir="logs"):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"trading_agent_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

"""
第三部分：配置类初始化
"""

class AgentConfig:
    """智能体配置"""
    def __init__(self):
        # 网络参数
        self.state_dim = 30
        self.action_dim = 6
        self.hidden_dim = 256
        
        # 训练参数
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.replay_buffer_size = 200000
        self.warmup_steps = 5000
        
        # SAC参数
        self.alpha = 0.2
        self.target_entropy = -self.action_dim
        self.alpha_lr = 3e-4
        self.autotune_alpha = True
        
        # 训练控制
        self.max_episodes = 2000
        self.max_steps = 500
        self.eval_freq = 20
        self.save_freq = 50
        self.gradient_steps = 1
        
        # 期权交易参数
        self.initial_capital = 100000
        self.max_position = 10
        self.transaction_cost = 0.001
        self.margin_requirement = 0.2
        self.max_drawdown_limit = 0.2
        
        # 数据参数
        self.lookback_window = 20
        self.feature_columns = [
            'underlying_price', 'volume', 'open_interest', 'iv', 'delta', 
            'gamma', 'theta', 'vega', 'moneyness', 'days_to_expiry',
            'bid_ask_spread', 'volume_ratio', 'skew', 'term_structure', 
            'risk_free_rate', 'dividend_yield', 'implied_volatility',
            'historical_volatility', 'put_call_ratio', 'volume_imbalance'
        ]
        
        # 新增参数
        self.max_position_size = 0.1
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10
        self.max_portfolio_vega = 5000
        self.max_portfolio_theta = -1000

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
            for k, v in config_dict.items():
                if hasattr(self, k):
                    setattr(self, k, v)

"""
第四部分：简化的期权交易环境
"""

class OptionTradingEnv(gym.Env):
    """期权交易环境"""
    def __init__(self, config, data_generator=None, mode='train'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.data_generator = data_generator
        
        # 状态空间和动作空间定义
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(config.state_dim,), 
            dtype=np.float32
        )
        
        # 动作空间：0-5对应不同操作
        self.action_space = spaces.Discrete(config.action_dim)
        
        # 账户状态
        self.cash = config.initial_capital
        self.positions = {}  # 持仓
        self.closed_positions = []  # 已平仓记录
        self.current_step = 0
        
        # 绩效跟踪
        self.max_portfolio_value = config.initial_capital
        self.drawdown = 0.0
        self.stop_loss_triggered = False
        
        # 数据
        self.current_market_data = None
        self.option_chain = None
        
        # 交易历史
        self.trade_history = []
        self.value_history = []
        self.portfolio_greeks = {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0
        }
        
        self.reset()

    def reset(self):
        """重置环境"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.closed_positions = []
        self.current_step = 0
        self.max_portfolio_value = self.config.initial_capital
        self.drawdown = 0.0
        self.stop_loss_triggered = False
        self.trade_history = []
        self.value_history = []
        self.portfolio_greeks = {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0
        }
        
        self._generate_mock_data()
        return self._get_state()

    def step(self, action):
        """执行动作并返回下一个状态"""
        try:
            # 保存当前组合价值
            current_value = self._calculate_portfolio_value()
            self.value_history.append(current_value)
            
            # 解析动作
            parsed_action = self._parse_action(action)
            
            # 执行交易
            trades = self._execute_trades(parsed_action)
            
            # 更新持仓
            self._update_positions()
            
            # 更新希腊值
            self._update_portfolio_greeks()
            
            # 检查止损
            self._check_stop_loss()
            
            # 检查是否平仓
            self._check_positions_for_closing()
            
            # 计算奖励
            reward = self._calculate_reward(trades)
            
            # 获取下一个状态
            next_state = self._get_state()
            
            # 检查是否结束
            done = self._check_done()
            
            # 生成信息
            info = {
                'portfolio_value': current_value,
                'cash': self.cash,
                'positions': len(self.positions),
                'drawdown': self.drawdown,
                'trades': trades,
                'portfolio_greeks': self.portfolio_greeks.copy(),
                'step': self.current_step
            }
            
            self.current_step += 1
            return next_state, reward, done, info
            
        except Exception as e:
            logger.error(f"执行步骤时出错: {str(e)}")
            return self._get_state(), -1.0, True, {'error': str(e)}

    def _parse_action(self, action):
        """解析动作"""
        actions = {
            0: {'type': 'hold', 'direction': None, 'quantity': 0},
            1: {'type': 'buy', 'direction': 'call', 'quantity': 1},
            2: {'type': 'sell', 'direction': 'call', 'quantity': 1},
            3: {'type': 'buy', 'direction': 'put', 'quantity': 1},
            4: {'type': 'sell', 'direction': 'put', 'quantity': 1},
            5: {'type': 'hedge', 'direction': 'delta', 'quantity': 0}
        }
        return actions.get(action, {'type': 'hold'})

    def _execute_trades(self, action):
        """执行交易"""
        trades = []
        
        if action['type'] == 'hold':
            return trades
            
        # 计算可用于交易的资金
        portfolio_value = self._calculate_portfolio_value()
        trade_value = portfolio_value * self.config.max_position_size
        
        # 选择要交易的期权
        option = self._select_option_for_trade(action, trade_value)
        if not option:
            return trades
        
        # 计算交易量
        if action['type'] == 'buy':
            max_quantity = int((self.cash * (1 - self.config.margin_requirement)) / 
                             (option['price'] * 100))  # 期权合约通常是100份
        else:  # sell
            max_quantity = int(trade_value / (option['price'] * 100))
            
        quantity = min(max_quantity, action['quantity'])
        if quantity <= 0:
            return trades
            
        # 执行交易
        trade = self._place_order(option, quantity, action['type'], action['direction'])
        if trade:
            trades.append(trade)
            self.trade_history.append(trade)
            
        return trades

    def _select_option_for_trade(self, action, trade_value):
        """选择要交易的期权"""
        if not self.option_chain:
            return None
            
        # 为了简化，我们选择最接近平值的期权
        target_moneyness = 1.0
        best_option = None
        min_distance = float('inf')
        
        for option in self.option_chain:
            # 只考虑指定方向的期权
            if action['direction'] == 'call' and option['type'] != 'call':
                continue
            if action['direction'] == 'put' and option['type'] != 'put':
                continue
                
            # 计算与目标moneyness的距离
            distance = abs(option['moneyness'] - target_moneyness)
            if distance < min_distance:
                min_distance = distance
                best_option = option
                
        return best_option

    def _place_order(self, option, quantity, action_type, direction):
        """下订单"""
        option_id = f"{option['underlying']}_{option['expiry']}_{option['strike']}_{option['type']}"
        price = option['price']
        total_cost = price * quantity * 100  # 100是期权合约乘数
        
        # 计算交易成本
        cost = total_cost * self.config.transaction_cost
        
        if action_type == 'buy':
            # 检查资金是否足够
            if self.cash < (total_cost + cost):
                logger.warning(f"资金不足，无法购买 {quantity} 份 {option_id}")
                return None
                
            # 更新现金
            self.cash -= (total_cost + cost)
            
            # 添加到持仓
            if option_id in self.positions:
                self.positions[option_id]['quantity'] += quantity
            else:
                self.positions[option_id] = {
                    'option': option.copy(),
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': self.current_step,
                    'unrealized_pnl': 0.0,
                    'stop_loss_price': price * (1 - self.config.stop_loss_pct),
                    'take_profit_price': price * (1 + self.config.take_profit_pct)
                }
                
        elif action_type == 'sell':
            # 在简化版本中，我们假设可以卖空
            # 更新现金
            self.cash += (total_cost - cost)  # 扣除交易成本
            
            # 添加到持仓（负数量表示卖空）
            if option_id in self.positions:
                self.positions[option_id]['quantity'] -= quantity
            else:
                self.positions[option_id] = {
                    'option': option.copy(),
                    'quantity': -quantity,
                    'entry_price': price,
                    'entry_time': self.current_step,
                    'unrealized_pnl': 0.0,
                    'stop_loss_price': price * (1 + self.config.stop_loss_pct),  # 卖空的止损价格更高
                    'take_profit_price': price * (1 - self.config.take_profit_pct)  # 卖空的止盈价格更低
                }
                
        return {
            'option_id': option_id,
            'action': action_type,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'total_cost': total_cost,
            'time': self.current_step
        }

    def _update_positions(self):
        """更新持仓"""
        for option_id, position in self.positions.items():
            # 查找当前期权价格
            current_price = None
            for option in self.option_chain:
                if option_id == f"{option['underlying']}_{option['expiry']}_{option['strike']}_{option['type']}":
                    current_price = option['price']
                    break
                    
            if current_price is None:
                continue  # 期权可能已经到期
                
            # 更新未实现盈亏
            position['unrealized_pnl'] = (current_price - position['entry_price']) * \
                                       position['quantity'] * 100
            
            # 更新期权数据
            position['option']['price'] = current_price

    def _update_portfolio_greeks(self):
        """更新组合希腊值"""
        self.portfolio_greeks = {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0
        }
        
        for position in self.positions.values():
            option = position['option']
            quantity = position['quantity']
            
            # 更新组合希腊值
            self.portfolio_greeks['delta'] += option['delta'] * quantity * 100
            self.portfolio_greeks['gamma'] += option['gamma'] * quantity * 100
            self.portfolio_greeks['theta'] += option['theta'] * quantity * 100
            self.portfolio_greeks['vega'] += option['vega'] * quantity * 100

    def _check_positions_for_closing(self):
        """检查是否需要平仓"""
        positions_to_close = []
        
        for option_id, position in self.positions.items():
            option = position['option']
            current_price = option['price']
            
            # 检查止盈止损
            if position['quantity'] > 0:  # 多头
                if current_price <= position['stop_loss_price']:
                    positions_to_close.append((option_id, 'stop_loss'))
                elif current_price >= position['take_profit_price']:
                    positions_to_close.append((option_id, 'take_profit'))
            else:  # 空头
                if current_price >= position['stop_loss_price']:
                    positions_to_close.append((option_id, 'stop_loss'))
                elif current_price <= position['take_profit_price']:
                    positions_to_close.append((option_id, 'take_profit'))
                    
            # 检查是否到期
            if option['days_to_expiry'] <= 0:
                positions_to_close.append((option_id, 'expired'))
                
        # 执行平仓
        for option_id, reason in positions_to_close:
            self._close_position(option_id, self.positions[option_id], reason)

    def _close_position(self, option_id, position, reason):
        """平仓"""
        if option_id not in self.positions:
            return
            
        option = position['option']
        quantity = position['quantity']
        entry_price = position['entry_price']
        current_price = option['price']
        
        # 计算实现盈亏
        realized_pnl = (current_price - entry_price) * quantity * 100
        
        # 更新现金
        if quantity > 0:  # 平多头
            self.cash += (current_price * quantity * 100 - 
                        current_price * quantity * 100 * self.config.transaction_cost)
        else:  # 平空头
            self.cash -= (current_price * abs(quantity) * 100 + 
                        current_price * abs(quantity) * 100 * self.config.transaction_cost)
        
        # 记录平仓
        closed_position = {
            'option_id': option_id,
            'option': option.copy(),
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': current_price,
            'entry_time': position['entry_time'],
            'exit_time': self.current_step,
            'realized_pnl': realized_pnl,
            'reason': reason
        }
        
        self.closed_positions.append(closed_position)
        
        # 从持仓中移除
        del self.positions[option_id]
        
        logger.info(f"平仓 {option_id}: 数量={quantity}, 盈亏={realized_pnl:.2f}, 原因={reason}")

    def _calculate_reward(self, trades):
        """计算奖励"""
        current_value = self._calculate_portfolio_value()
        
        # 基本奖励是组合价值的变化
        if len(self.value_history) > 1:
            reward = (current_value - self.value_history[-2]) / self.value_history[-2]
        else:
            reward = 0.0
            
        # 惩罚回撤
        self.drawdown = max(0.0, 1.0 - current_value / self.max_portfolio_value)
        reward -= self.drawdown * 200
        
        # 惩罚过大的希腊值风险
        greek_penalty = 0.0
        if abs(self.portfolio_greeks['vega']) > self.config.max_portfolio_vega:
            greek_penalty += 0.1
        if self.portfolio_greeks['theta'] < self.config.max_portfolio_theta:
            greek_penalty += 0.1
            
        reward -= greek_penalty * 50
        
        # 交易成本惩罚
        if trades:
            total_cost = sum(t.get('cost', 0) for t in trades)
            cost_penalty = total_cost / current_value if current_value > 0 else 0
            reward -= cost_penalty * 100
            
        # 适度分散奖励
        position_count = len(self.positions)
        if 0 < position_count <= 5:
            reward += 0.5
            
        return reward

    def _check_stop_loss(self):
        """检查是否触发止损"""
        current_value = self._calculate_portfolio_value()
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        self.drawdown = 1.0 - current_value / self.max_portfolio_value
        
        if self.drawdown >= self.config.max_drawdown_limit:
            self.stop_loss_triggered = True
            logger.warning(f"触发止损: 回撤 {self.drawdown:.2%}")

    def _check_done(self):
        """检查是否结束"""
        if self.stop_loss_triggered:
            return True
            
        if self.current_step >= self.config.max_steps:
            return True
            
        current_value = self._calculate_portfolio_value()
        if current_value <= self.config.initial_capital * 0.1:
            return True
            
        return False

    def _calculate_portfolio_value(self):
        """计算组合价值"""
        value = self.cash
        
        for position in self.positions.values():
            option = position['option']
            quantity = position['quantity']
            value += option['price'] * quantity * 100
            
        return value

    def _get_state(self):
        """获取当前状态"""
        state = []
        
        # 市场数据特征
        if self.current_market_data:
            state.extend([
                self.current_market_data['underlying_price'] / 100,  # 归一化
                self.current_market_data['volume'] / 1000000,
                self.current_market_data['open_interest'] / 1000000,
                self.current_market_data['iv'] * 100,
                self.current_market_data['risk_free_rate'] * 100,
                self.current_market_data['dividend_yield'] * 100,
                self.current_market_data['implied_volatility'] * 100,
                self.current_market_data['historical_volatility'] * 100,
                self.current_market_data['put_call_ratio'],
                self.current_market_data['volume_imbalance']
            ])
        else:
            state.extend([0]*10)
            
        # 账户特征
        portfolio_value = self._calculate_portfolio_value()
        state.extend([
            portfolio_value / self.config.initial_capital,  # 归一化
            self.cash / portfolio_value if portfolio_value > 0 else 0,
            len(self.positions) / self.config.max_position,
            self.drawdown,
            self.stop_loss_triggered * 1.0
        ])
        
        # 组合希腊值
        state.extend([
            self.portfolio_greeks['delta'] / 10000,  # 归一化
            self.portfolio_greeks['gamma'] / 1000,
            self.portfolio_greeks['theta'] / -10000,
            self.portfolio_greeks['vega'] / 10000
        ])
        
        # 近期回报
        if len(self.value_history) >= 5:
            recent_returns = [
                (self.value_history[i] - self.value_history[i-1]) / max(self.value_history[i-1], 1e-6)
                for i in range(max(1, len(self.value_history)-5), len(self.value_history))
            ]
            state.extend([
                np.mean(recent_returns) if recent_returns else 0,
                np.std(recent_returns) if recent_returns else 0
            ])
        else:
            state.extend([0, 0])
            
        # 填充到指定维度
        state_array = np.array(state, dtype=np.float32)
        if len(state_array) < self.config.state_dim:
            padding = np.zeros(self.config.state_dim - len(state_array), dtype=np.float32)
            state_array = np.concatenate([state_array, padding])
        elif len(state_array) > self.config.state_dim:
            state_array = state_array[:self.config.state_dim]
            
        return state_array

    def _generate_mock_data(self):
        """生成模拟数据"""
        # 模拟标的资产价格
        underlying_price = 100 + np.random.normal(0, 2)
        
        # 模拟市场数据
        self.current_market_data = {
            'underlying': 'AAPL',
            'underlying_price': underlying_price,
            'volume': random.randint(1000000, 10000000),
            'open_interest': random.randint(500000, 5000000),
            'iv': np.random.uniform(0.1, 0.5),
            'risk_free_rate': np.random.uniform(0.01, 0.05),
            'dividend_yield': np.random.uniform(0.0, 0.03),
            'implied_volatility': np.random.uniform(0.15, 0.45),
            'historical_volatility': np.random.uniform(0.1, 0.4),
            'put_call_ratio': np.random.uniform(0.5, 1.5),
            'volume_imbalance': np.random.uniform(-0.5, 0.5)
        }
        
        # 生成期权链
        self.option_chain = []
        strikes = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 21)  # 21个行权价
        
        for strike in strikes:
            # 计算moneyness
            moneyness = underlying_price / strike
            
            # 随机生成到期日（10-90天）
            days_to_expiry = random.randint(10, 90)
            
            # 计算隐含波动率（平值附近较低，实值和虚值较高）
            iv = np.random.uniform(0.15, 0.45) * (1 + 0.1 * abs(moneyness - 1.0))
            
            # 生成看涨期权
            call_option = {
                'underlying': 'AAPL',
                'expiry': (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d'),
                'strike': strike,
                'type': 'call',
                'price': np.random.uniform(max(0, underlying_price - strike) * 0.5, 
                                         max(0, underlying_price - strike) * 1.5),
                'bid': np.random.uniform(0, 10),
                'ask': np.random.uniform(0.1, 11),
                'volume': random.randint(0, 1000),
                'open_interest': random.randint(0, 10000),
                'iv': iv,
                'delta': np.random.uniform(0.1, 0.9) if strike < underlying_price else np.random.uniform(0.1, 0.5),
                'gamma': np.random.uniform(0.01, 0.1),
                'theta': -np.random.uniform(0.01, 0.1),
                'vega': np.random.uniform(0.1, 1.0),
                'moneyness': moneyness,
                'days_to_expiry': days_to_expiry,
                'bid_ask_spread': np.random.uniform(0.01, 0.5),
                'volume_ratio': np.random.uniform(0.1, 2.0),
                'skew': np.random.uniform(-0.5, 0.5),
                'term_structure': np.random.uniform(-0.5, 0.5),
                'volume_imbalance': np.random.uniform(-0.5, 0.5)
            }
            self.option_chain.append(call_option)
            
            # 生成看跌期权
            put_option = {
                'underlying': 'AAPL',
                'expiry': (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d'),
                'strike': strike,
                'type': 'put',
                'price': np.random.uniform(max(0, strike - underlying_price) * 0.5, 
                                         max(0, strike - underlying_price) * 1.5),
                'bid': np.random.uniform(0, 10),
                'ask': np.random.uniform(0.1, 11),
                'volume': random.randint(0, 1000),
                'open_interest': random.randint(0, 10000),
                'iv': iv,
                'delta': -np.random.uniform(0.9, 0.1) if strike > underlying_price else -np.random.uniform(0.5, 0.1),
                'gamma': np.random.uniform(0.01, 0.1),
                'theta': -np.random.uniform(0.01, 0.1),
                'vega': np.random.uniform(0.1, 1.0),
                'moneyness': moneyness,
                'days_to_expiry': days_to_expiry,
                'bid_ask_spread': np.random.uniform(0.01, 0.5),
                'volume_ratio': np.random.uniform(0.1, 2.0),
                'skew': np.random.uniform(-0.5, 0.5),
                'term_structure': np.random.uniform(-0.5, 0.5),
                'volume_imbalance': np.random.uniform(-0.5, 0.5)
            }
            self.option_chain.append(put_option)

    def render(self, mode='human'):
        """渲染环境"""
        portfolio_value = self._calculate_portfolio_value()
        
        if mode == 'human':
            print(f"\n步骤 {self.current_step}:")
            print(f"组合价值: ${portfolio_value:,.2f}")
            print(f"现金: ${self.cash:,.2f}")
            print(f"持仓数: {len(self.positions)}")
            print(f"最大回撤: {self.drawdown:.2%}")
            print(f"组合希腊值: delta={self.portfolio_greeks['delta']:.2f}, "
                  f"gamma={self.portfolio_greeks['gamma']:.4f}, "
                  f"theta={self.portfolio_greeks['theta']:.2f}, "
                  f"vega={self.portfolio_greeks['vega']:.2f}")
            
            if self.positions:
                print("\n当前持仓:")
                for i, (option_id, position) in enumerate(self.positions.items()):
                    print(f"  {i+1}. {option_id}: 数量={position['quantity']}, "
                          f"成本=${position['entry_price']:.2f}, "
                          f"当前=${position['option']['price']:.2f}, "
                          f"盈亏=${position['unrealized_pnl']:.2f}")
        
        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions),
            'drawdown': self.drawdown,
            'portfolio_greeks': self.portfolio_greeks.copy()
        }

"""
第五部分：简化的智能体（随机策略）
"""

class SimpleRandomAgent:
    """简化的随机策略智能体"""
    def __init__(self, config):
        self.config = config
        self.action_dim = config.action_dim
        
    def select_action(self, state, deterministic=False, explore=True):
        """选择动作"""
        # 如果deterministic为True，选择持有
        if deterministic:
            return 0
            
        # 随机选择动作
        return random.randint(0, self.action_dim - 1)

"""
第六部分：训练器
"""

class Trainer:
    """训练器"""
    def __init__(self, config, env, agent):
        self.config = config
        self.env = env
        self.agent = agent
        
        # 训练统计
        self.episode_rewards = []
        self.episode_portfolio_values = []
        self.episode_drawdowns = []
        self.episode_lengths = []
        self.episode_trades = []
        
        self.start_time = None

    def train(self):
        """训练智能体"""
        self.start_time = time.time()
        logger.info("开始训练期权交易智能体...")
        
        for episode in range(self.config.max_episodes):
            # 重置环境
            state = self.env.reset()
            episode_reward = 0
            episode_trades = 0
            
            for step in range(self.config.max_steps):
                # 选择动作
                action = self.agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 累积奖励
                episode_reward += reward
                
                # 统计交易次数
                if info.get('trades'):
                    episode_trades += len(info['trades'])
                    
                # 更新状态
                state = next_state
                
                if done:
                    break
                    
            # 记录统计信息
            final_value = self.env._calculate_portfolio_value()
            self.episode_rewards.append(episode_reward)
            self.episode_portfolio_values.append(final_value)
            self.episode_drawdowns.append(self.env.drawdown)
            self.episode_lengths.append(step + 1)
            self.episode_trades.append(episode_trades)
            
            # 定期输出进度
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                avg_value = np.mean(self.episode_portfolio_values[-10:]) if len(self.episode_portfolio_values) >= 10 else np.mean(self.episode_portfolio_values)
                avg_drawdown = np.mean(self.episode_drawdowns[-10:]) if len(self.episode_drawdowns) >= 10 else np.mean(self.episode_drawdowns)
                avg_trades = np.mean(self.episode_trades[-10:]) if len(self.episode_trades) >= 10 else np.mean(self.episode_trades)
                
                elapsed_time = time.time() - self.start_time
                remaining_episodes = self.config.max_episodes - episode - 1
                estimated_remaining_time = elapsed_time / (episode + 1) * remaining_episodes
                
                logger.info(f"Episode {episode:4d}/{self.config.max_episodes} | "
                           f"平均奖励: {avg_reward:.2f} | "
                           f"平均组合价值: ${avg_value:,.2f} | "
                           f"平均回撤: {avg_drawdown*100:.2f}% | "
                           f"平均交易次数: {avg_trades:.1f} | "
                           f"已用时间: {elapsed_time:.0f}s | "
                           f"剩余时间: {estimated_remaining_time:.0f}s")
            
            # 定期评估
            if episode % self.config.eval_freq == 0 and episode > 0:
                self._perform_evaluation(episode)
        
        return self._finalize_training()

    def _perform_evaluation(self, episode):
        """执行评估"""
        logger.info(f"评估 Episode {episode}...")
        
        total_reward = 0
        total_value = 0
        total_drawdown = 0
        
        for _ in range(5):  # 评估5次
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.max_steps):
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
                    
            total_reward += episode_reward
            total_value += self.env._calculate_portfolio_value()
            total_drawdown += self.env.drawdown
            
        avg_reward = total_reward / 5
        avg_value = total_value / 5
        avg_drawdown = total_drawdown / 5
        
        logger.info(f"评估结果: 平均奖励={avg_reward:.2f}, "
                   f"平均组合价值=${avg_value:,.2f}, "
                   f"平均回撤={avg_drawdown*100:.2f}%")

    def _finalize_training(self):
        """完成训练"""
        logger.info("训练完成！")
        
        # 计算最终统计
        final_stats = {
            'total_episodes': self.config.max_episodes,
            'total_steps': sum(self.episode_lengths),
            'avg_episode_reward': np.mean(self.episode_rewards),
            'avg_portfolio_value': np.mean(self.episode_portfolio_values),
            'max_portfolio_value': max(self.episode_portfolio_values),
            'min_portfolio_value': min(self.episode_portfolio_values),
            'avg_drawdown': np.mean(self.episode_drawdowns),
            'max_drawdown': max(self.episode_drawdowns),
            'avg_trades_per_episode': np.mean(self.episode_trades),
            'total_training_time': time.time() - self.start_time
        }
        
        logger.info("最终统计:")
        for key, value in final_stats.items():
            if key == 'total_training_time':
                logger.info(f"  {key}: {value:.2f}秒")
            elif 'value' in key or 'capital' in key:
                logger.info(f"  {key}: ${value:,.2f}")
            elif 'drawdown' in key:
                logger.info(f"  {key}: {value*100:.2f}%")
            else:
                logger.info(f"  {key}: {value}")
                
        return final_stats

    def evaluate(self, num_episodes=10, render=False):
        """评估智能体"""
        logger.info(f"开始评估，共 {num_episodes} 个episode...")
        
        total_reward = 0
        total_value = 0
        total_drawdown = 0
        total_trades = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_trades = 0
            
            for step in range(self.config.max_steps):
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if info.get('trades'):
                    episode_trades += len(info['trades'])
                    
                if render and episode == 0:  # 只渲染第一个episode
                    self.env.render()
                    time.sleep(0.1)  # 放慢速度以便观察
                    
                state = next_state
                
                if done:
                    break
                    
            total_reward += episode_reward
            total_value += self.env._calculate_portfolio_value()
            total_drawdown += self.env.drawdown
            total_trades += episode_trades
            
        # 计算平均统计
        avg_reward = total_reward / num_episodes
        avg_value = total_value / num_episodes
        avg_drawdown = total_drawdown / num_episodes
        avg_trades = total_trades / num_episodes
        
        logger.info(f"评估结果 ({num_episodes} episodes):")
        logger.info(f"  平均奖励: {avg_reward:.2f}")
        logger.info(f"  平均组合价值: ${avg_value:,.2f}")
        logger.info(f"  平均回撤: {avg_drawdown*100:.2f}%")
        logger.info(f"  平均交易次数: {avg_trades:.1f}")
        
        return {
            'avg_reward': avg_reward,
            'avg_portfolio_value': avg_value,
            'avg_drawdown': avg_drawdown,
            'avg_trades': avg_trades
        }

"""
第七部分：主函数
"""

def quick_test():
    """快速测试"""
    print("开始快速测试期权交易智能体...")
    
    # 创建必要的目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 初始化配置
    config = AgentConfig()
    config.max_episodes = 10  # 减少测试时间
    config.max_steps = 100
    
    # 创建环境
    env = OptionTradingEnv(config, mode='train')
    
    # 创建智能体
    agent = SimpleRandomAgent(config)
    
    # 创建训练器
    trainer = Trainer(config, env, agent)
    
    # 训练智能体
    stats = trainer.train()
    
    # 评估智能体
    eval_stats = trainer.evaluate(num_episodes=3, render=True)
    
    print("\n测试完成！")
    print(f"最终组合价值: ${stats['avg_portfolio_value']:,.2f}")
    print(f"平均回撤: {stats['avg_drawdown']*100:.2f}%")
    print(f"平均交易次数: {stats['avg_trades_per_episode']:.1f}")
    
    return stats, eval_stats

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs("plots", exist_ok=True)
    
    # 运行快速测试
    stats, eval_stats = quick_test()
    
    print("\n期权交易智能体简化版运行成功！")
    print("这个简化版本使用随机策略，不依赖PyTorch。")
    print("您可以通过修改SimpleRandomAgent类来实现更复杂的策略。")

if __name__ == "__main__":
    main()
