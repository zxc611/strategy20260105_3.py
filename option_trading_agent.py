"""
期权交易策略智能体完整系统 - 完善版
文件名：option_trading_agent_fixed.py
第一部分：导入模块
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
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
        
        # 风险管理参数
        self.max_position_size = 0.1
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10
        self.max_portfolio_vega = 5000
        self.max_portfolio_theta = -1000

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def load(self, path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            setattr(self, k, v)
"""
第十一部分：ActorNetwork网络结构
"""

class ActorNetwork(nn.Module):
    """策略网络（Actor） - 改进版"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
        )
        
        # 输出层
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 初始化
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state, deterministic=False, with_logprob=True):
        """前向传播"""
        features = self.feature_net(state)
        
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)
        
        # 创建分布
        pi_distribution = Normal(mu, std)
        
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
            
        # 使用tanh限制动作范围
        pi_action = torch.tanh(pi_action)
        
        if with_logprob:
            # 计算log概率（考虑tanh变换）
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - torch.log(1 + torch.exp(-2 * pi_action)))).sum(axis=1)
        else:
            logp_pi = None
            
        return pi_action, logp_pi, mu, log_std

    def get_action(self, state, deterministic=False):
        """获取动作"""
        with torch.no_grad():
            action, _, _, _ = self.forward(
                state, deterministic=deterministic, with_logprob=False
            )
        return action
"""
第十五部分：CriticNetwork网络结构
"""

class CriticNetwork(nn.Module):
    """价值网络（Critic） - 改进版"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Q1网络
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2网络
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        """前向传播"""
        sa = torch.cat([state, action], dim=1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2
"""
第十八部分：优先级经验回放缓冲区初始化
"""

class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    def __init__(self, capacity, state_dim, action_dim, device, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        # 初始化缓冲区
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # 优先级
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        idx = self.ptr
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        # 设置初始优先级为最大优先级
        self.priorities[idx] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """采样经验（带优先级）"""
        if self.size == 0:
            return None
            
        # 计算采样概率
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        idxs = np.random.choice(self.size, batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (self.size * probs[idxs]) ** (-self.beta)
        weights /= weights.max()
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = {
            'states': torch.FloatTensor(self.states[idxs]).to(self.device),
            'actions': torch.FloatTensor(self.actions[idxs]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[idxs]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[idxs]).to(self.device),
            'dones': torch.FloatTensor(self.dones[idxs]).to(self.device),
            'indices': idxs,
            'weights': torch.FloatTensor(weights).to(self.device)
        }
        
        return batch

    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.size
"""
第二十二部分：期权交易环境初始化
"""

class OptionTradingEnv(gym.Env):
    """期权交易环境 - 完善版"""
    
    def __init__(self, config, data_generator=None, mode='train'):
        super().__init__()
        
        self.config = config
        self.mode = mode
        
        # 动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.action_dim,), dtype=np.float32
        )
        
        # 状态空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(config.state_dim,), dtype=np.float32
        )

        # 交易状态
        self.cash = config.initial_capital
        self.positions = {}  # 持仓
        self.closed_positions = []  # 已平仓记录
        self.current_step = 0
        
        # 风险管理
        self.max_portfolio_value = config.initial_capital
        self.drawdown = 0.0
        self.stop_loss_triggered = False
        
        # 数据
        self.data_generator = data_generator
        self.current_market_data = None
        self.option_chain = None
        
        # 交易记录
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
        self.stop_loss_triggered = False
        self.max_portfolio_value = self.config.initial_capital
        
        # 获取初始数据
        if self.data_generator:
            self.current_market_data, self.option_chain = self.data_generator.get_initial_state()
        else:
            self.current_market_data, self.option_chain = self._generate_mock_data()
        
        # 初始化历史记录
        self.trade_history = []
        self.value_history = [self._calculate_portfolio_value()]
        self._update_portfolio_greeks()
        
        logger.info(f"环境重置，初始资本: ${self.cash:,.2f}")
        return self._get_state()
    
    def step(self, action):
        """执行一步"""
        try:
            # 解析动作
            parsed_action = self._parse_action(action)
            
            # 检查止损
            if self._check_stop_loss():
                self.stop_loss_triggered = True
                logger.warning(f"止损触发，当前回撤: {self.drawdown*100:.2f}%")
            
            # 执行交易
            trades = self._execute_trades(parsed_action) if not self.stop_loss_triggered else []
            
            # 平仓检查
            self._check_positions_for_closing()
            
            # 更新到下一步
            self.current_step += 1
            
            # 获取新数据
            if self.data_generator:
                self.current_market_data, self.option_chain = self.data_generator.get_next_state()
            else:
                self.current_market_data, self.option_chain = self._generate_mock_data()
            
            # 更新持仓价值
            self._update_positions()
            
            # 更新希腊值
            self._update_portfolio_greeks()
            
            # 计算奖励
            reward = self._calculate_reward(trades)
            
            # 获取新状态
            new_state = self._get_state()
            
            # 检查是否结束
            done = self._check_done()
            
            # 记录
            portfolio_value = self._calculate_portfolio_value()
            self.value_history.append(portfolio_value)
            
            # 更新最大价值和回撤
            self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
            self.drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            
            # 信息
            info = {
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions),
                'trades': trades,
                'step': self.current_step,
                'drawdown': self.drawdown,
                'portfolio_greeks': self.portfolio_greeks.copy()
            }
            
            return new_state, reward, done, info
            
        except Exception as e:
            logger.error(f"Step执行出错: {str(e)}")
            # 返回安全状态
            return self._get_state(), -10.0, True, {'error': str(e)}
    
    def _parse_action(self, action):
        """解析动作"""
        action_dict = {
            'position_size': (action[0] + 1) / 2,  # [0, 1] 仓位大小
            'option_type_bias': np.tanh(action[1]),  # [-1, 1] 期权类型偏好
            'expiry_preference': (action[2] + 1) / 2,  # [0, 1] 到期日偏好
            'strike_preference': np.tanh(action[3]),  # [-1, 1] 行权价偏好
            'hedge_ratio': (action[4] + 1) / 2,  # [0, 1] 对冲比例
            'risk_appetite': (action[5] + 1) / 2  # [0, 1] 风险偏好
        }
        return action_dict
    
    def _execute_trades(self, action):
        """执行交易（支持买入和卖出）"""
        trades = []
        
        # 计算目标仓位
        target_position_value = self._calculate_portfolio_value() * action['position_size'] * self.config.max_position_size
        
        # 计算当前持仓价值
        current_position_value = sum(
            pos['quantity'] * pos['current_price'] 
            for pos in self.positions.values()
        )
        
        # 确定交易方向
        trade_value = target_position_value - current_position_value
        
        if abs(trade_value) < self.config.initial_capital * 0.01:  # 最小交易阈值
            return trades
        
        # 选择期权
        selected_option, quantity = self._select_option_for_trade(action, trade_value)
        
        if selected_option and quantity != 0:
            # 执行交易
            trade_success, trade_info = self._place_order(selected_option, quantity)
            
            if trade_success:
                trades.append(trade_info)
                logger.debug(f"执行交易: {trade_info}")
        
        # Delta对冲
        if action['hedge_ratio'] > 0.5 and abs(self.portfolio_greeks['delta']) > 1000:
            hedge_trade = self._execute_delta_hedge(action['hedge_ratio'])
            if hedge_trade:
                trades.append(hedge_trade)
        
        return trades

    def _select_option_for_trade(self, action, trade_value):
        """选择期权进行交易"""
        if not self.option_chain:
            return None, 0
        
        # 筛选期权
        filtered_options = []
        for option in self.option_chain:
            # 根据偏好筛选
            if action['option_type_bias'] > 0 and option['type'] == 'PUT':
                continue
            if action['option_type_bias'] < 0 and option['type'] == 'CALL':
                continue
            
            # 到期日筛选
            expiry_score = 1.0 - abs(action['expiry_preference'] - (option['days_to_expiry'] / 90))
            if expiry_score < 0.3:
                continue
            
            filtered_options.append(option)
        
        if not filtered_options:
            return None, 0
        
        # 评分选择
        scores = []
        for option in filtered_options:
            score = self._calculate_option_score(option, action)
            scores.append(score)
        
        # 选择最佳期权
        best_idx = np.argmax(scores)
        selected_option = filtered_options[best_idx]
        
        # 计算交易数量
        option_price = selected_option['price']
        quantity = int(trade_value / option_price)
        
        # 限制数量
        max_quantity = int(self.cash / option_price * 0.5)  # 最多使用50%现金
        quantity = np.clip(quantity, -max_quantity, max_quantity)
        
        return selected_option, quantity

    def _calculate_option_score(self, option, action):
        """计算期权得分"""
        score = 0.0
        
        # 隐含波动率因素
        iv_rank = (option['iv'] - 0.2) / 0.3  # 假设IV范围0.2-0.5
        if action['risk_appetite'] > 0.7:  # 高风险偏好
            score += iv_rank * 20
        else:  # 低风险偏好
            score += (1 - iv_rank) * 20
        
        # 时间价值衰减
        if option['days_to_expiry'] > 7:  # 避免接近到期
            theta_score = -option['theta'] / option['price'] * 100
            score += theta_score * 10
        
        # 流动性因素
        if 'volume' in option:
            score += np.log1p(option['volume']) * 5
        
        # 价差因素
        if 'bid_ask_spread' in option:
            spread_pct = option['bid_ask_spread'] / option['price']
            score -= spread_pct * 100
        
        return score

    def _place_order(self, option, quantity):
        """下单"""
        if quantity == 0:
            return False, None
        
        cost = option['price'] * abs(quantity) * (1 + self.config.transaction_cost)
        
        if quantity > 0:  # 买入
            if cost > self.cash:
                return False, None
            
            self.cash -= cost
            action_type = 'BUY'
        else:  # 卖出
            # 检查是否有持仓可卖
            option_id = option['id']
            if option_id in self.positions:
                current_qty = self.positions[option_id]['quantity']
                if current_qty + quantity < 0:  # 卖空超出持仓
                    return False, None
            action_type = 'SELL'
            self.cash += abs(cost)
        
        # 更新持仓
        option_id = option['id']
        if option_id in self.positions:
            self.positions[option_id]['quantity'] += quantity
            if self.positions[option_id]['quantity'] == 0:
                del self.positions[option_id]
        elif quantity > 0:
            self.positions[option_id] = {
                'option_data': option.copy(),
                'quantity': quantity,
                'entry_price': option['price'],
                'entry_time': self.current_step
            }
        
        trade_info = {
            'option_id': option_id,
            'quantity': quantity,
            'price': option['price'],
            'action': action_type,
            'cost': cost,
            'timestamp': self.current_step
        }
        
        return True, trade_info

    def _execute_delta_hedge(self, hedge_ratio):
        """执行Delta对冲"""
        target_delta = -self.portfolio_greeks['delta'] * hedge_ratio
        
        # 这里可以添加标的资产对冲逻辑
        # 简化版：记录对冲需求
        return {
            'type': 'DELTA_HEDGE',
            'target_delta': target_delta,
            'current_delta': self.portfolio_greeks['delta'],
            'timestamp': self.current_step
        }
    def _check_positions_for_closing(self):
        """检查是否需要平仓"""
        positions_to_close = []
        
        for option_id, position in list(self.positions.items()):
            option_data = position['option_data']
            current_price = option_data['price']
            entry_price = position['entry_price']
            
            # 止盈止损检查
            pnl_pct = (current_price - entry_price) / entry_price
            
            if pnl_pct <= -self.config.stop_loss_pct:
                positions_to_close.append((option_id, position, 'STOP_LOSS'))
            elif pnl_pct >= self.config.take_profit_pct:
                positions_to_close.append((option_id, position, 'TAKE_PROFIT'))
            elif option_data['days_to_expiry'] <= 1:  # 临近到期
                positions_to_close.append((option_id, position, 'EXPIRING'))
        
        # 执行平仓
        for option_id, position, reason in positions_to_close:
            self._close_position(option_id, position, reason)

    def _close_position(self, option_id, position, reason):
        """平仓"""
        current_price = position['option_data']['price']
        quantity = position['quantity']
        
        # 计算盈亏
        pnl = (current_price - position['entry_price']) * quantity
        self.cash += current_price * abs(quantity) * (1 - self.config.transaction_cost)
        
        # 记录
        self.closed_positions.append({
            'option_id': option_id,
            'quantity': quantity,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'pnl': pnl,
            'holding_period': self.current_step - position['entry_time'],
            'reason': reason,
            'timestamp': self.current_step
        })
        
        # 移除持仓
        del self.positions[option_id]
        
        logger.debug(f"平仓: {option_id}, 原因: {reason}, 盈亏: ${pnl:.2f}")

    def _update_positions(self):
        """更新持仓价值"""
        if not self.option_chain:
            return
        
        # 构建期权价格映射
        price_map = {opt['id']: opt['price'] for opt in self.option_chain}
        
        for option_id in list(self.positions.keys()):
            if option_id in price_map:
                self.positions[option_id]['option_data']['price'] = price_map[option_id]
                self.positions[option_id]['current_price'] = price_map[option_id]
            else:
                # 期权已过期或不存在
                logger.warning(f"期权 {option_id} 不存在于当前期权链中")
                del self.positions[option_id]

    def _update_portfolio_greeks(self):
        """更新组合希腊值"""
        total_delta = total_gamma = total_theta = total_vega = 0.0
        
        for position in self.positions.values():
            option_data = position['option_data']
            quantity = position['quantity']
            
            total_delta += option_data.get('delta', 0) * quantity
            total_gamma += option_data.get('gamma', 0) * quantity
            total_theta += option_data.get('theta', 0) * quantity
            total_vega += option_data.get('vega', 0) * quantity
        
        self.portfolio_greeks = {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega
        }

    def _calculate_reward(self, trades):
        """计算奖励 - 改进版"""
        current_value = self._calculate_portfolio_value()
        previous_value = self.value_history[-1] if self.value_history else self.config.initial_capital
        
        # 收益率
        return_pct = (current_value - previous_value) / previous_value if previous_value > 0 else 0
        
        # 奖励组成
        reward = 0.0
        
        # 1. 收益奖励（非线性）
        if return_pct > 0:
            reward += np.sign(return_pct) * (abs(return_pct) ** 0.5) * 100
        else:
            reward += return_pct * 150  # 损失惩罚更大
        
        # 2. 风险调整收益
        if len(self.value_history) >= 10:
            recent_returns = [
                (self.value_history[i] - self.value_history[i-1]) / self.value_history[i-1]
                for i in range(max(1, len(self.value_history)-10), len(self.value_history))
            ]
            if recent_returns:
                sharpe_ratio = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
                reward += sharpe_ratio * 50

        # 3. 回撤惩罚
        reward -= self.drawdown * 200
        
        # 4. 希腊值风险惩罚
        greek_penalty = 0.0
        if abs(self.portfolio_greeks['vega']) > self.config.max_portfolio_vega:
            greek_penalty += 0.1
        if self.portfolio_greeks['theta'] < self.config.max_portfolio_theta:
            greek_penalty += 0.1
        
        reward -= greek_penalty * 50
        
        # 5. 交易成本惩罚
        if trades:
            total_cost = sum(t.get('cost', 0) * self.config.transaction_cost for t in trades)
            cost_penalty = total_cost / current_value if current_value > 0 else 0
            reward -= cost_penalty * 100
        
        # 6. 分散奖励
        position_count = len(self.positions)
        if 0 < position_count <= 5:
            reward += 0.5  # 适度分散奖励
        
        return reward

    def _check_stop_loss(self):
        """检查止损"""
        return self.drawdown >= self.config.max_drawdown_limit
    
    def _check_done(self):
        """检查是否结束"""
        if self.current_step >= self.config.max_steps:
            return True
        
        if self.stop_loss_triggered:
            return True
        
        portfolio_value = self._calculate_portfolio_value()
        if portfolio_value <= self.config.initial_capital * 0.3:  # 破产线
            return True
        
        return False  # 所有条件都不满足时返回False

    def _calculate_portfolio_value(self):
        """计算组合价值"""
        total_value = self.cash
        
        for position in self.positions.values():
            total_value += position['quantity'] * position.get('current_price', position['entry_price'])
        
        return total_value

    def _get_state(self):
        """获取状态表示"""
        state = []
        
        # 1. 市场特征
        if self.current_market_data:
            for key in self.config.feature_columns[:15]:  # 前15个市场特征
                state.append(self.current_market_data.get(key, 0.0))
        else:
            state.extend([0.0] * 15)
        
        # 2. 期权链统计特征
        if self.option_chain:
            ivs = [opt.get('iv', 0.3) for opt in self.option_chain]
            deltas = [opt.get('delta', 0) for opt in self.option_chain]
            
            state.extend([
                np.mean(ivs) if ivs else 0.3,
                np.std(ivs) if ivs else 0.1,
                np.mean(deltas) if deltas else 0,
                len(self.option_chain)
            ])
        else:
            state.extend([0.3, 0.1, 0, 0])
        
        # 3. 组合状态
        portfolio_value = self._calculate_portfolio_value()
        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 1.0
        
        state.extend([
            cash_ratio,
            len(self.positions) / 10,  # 归一化
            portfolio_value / self.config.initial_capital - 1,
            self.current_step / self.config.max_steps,
            self.drawdown
        ])

        # 4. 希腊值暴露（归一化）
        state.extend([
            self.portfolio_greeks['delta'] / 10000,
            self.portfolio_greeks['gamma'] / 1000,
            self.portfolio_greeks['theta'] / -10000,
            self.portfolio_greeks['vega'] / 10000
        ])
        
        # 5. 历史收益特征
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
        
        # 确保状态维度正确
        state_array = np.array(state, dtype=np.float32)
        if len(state_array) < self.config.state_dim:
            padding = np.zeros(self.config.state_dim - len(state_array), dtype=np.float32)
            state_array = np.concatenate([state_array, padding])
        elif len(state_array) > self.config.state_dim:
            state_array = state_array[:self.config.state_dim]
        
        return state_array

    def _generate_mock_data(self):
        """生成模拟数据"""
        # 市场数据
        market_data = {}
        base_price = 5000 + np.random.randn() * 100
        
        for col in self.config.feature_columns[:15]:
            if col == 'underlying_price':
                market_data[col] = base_price
            elif col == 'volume':
                market_data[col] = np.random.uniform(1000, 10000)
            elif col == 'iv':
                market_data[col] = np.random.uniform(0.2, 0.5)
            elif col == 'implied_volatility':
                market_data[col] = np.random.uniform(0.2, 0.5)
            elif col == 'historical_volatility':
                market_data[col] = np.random.uniform(0.15, 0.4)
            else:
                market_data[col] = np.random.uniform(0, 1)
        
        # 期权链
        option_chain = []
        for i in range(20):  # 20个模拟期权
            option_type = 'CALL' if np.random.rand() > 0.5 else 'PUT'
            strike = base_price * np.random.uniform(0.8, 1.2)
            days_to_expiry = np.random.randint(1, 60)
            
            # 计算希腊值（简化版）
            moneyness = base_price / strike
            if option_type == 'CALL':
                delta = np.clip(np.random.normal(0.5, 0.2), 0, 1)
            else:
                delta = np.clip(np.random.normal(-0.5, 0.2), -1, 0)
            
            option_chain.append({
                'id': f'OPT_{i}_{option_type}_{strike:.0f}_{days_to_expiry}',
                'type': option_type,
                'strike': strike,
                'price': np.random.uniform(10, 200),
                'days_to_expiry': days_to_expiry,
                'iv': np.random.uniform(0.2, 0.5),
                'delta': delta,
                'gamma': np.random.uniform(0, 0.05),
                'theta': np.random.uniform(-20, 0),
                'vega': np.random.uniform(0, 100),
                'volume': np.random.randint(100, 1000),
                'bid_ask_spread': np.random.uniform(0.01, 0.1)
            })
        
        return market_data, option_chain

    def render(self, mode='human'):
        """渲染环境状态"""
        portfolio_value = self._calculate_portfolio_value()
        return_rate = (portfolio_value / self.config.initial_capital - 1) * 100
        
        print("\\n" + "="*60)
        print(f"步数: {self.current_step}/{self.config.max_steps}")
        print(f"组合价值: ${portfolio_value:,.2f} ({return_rate:+.2f}%)")
        print(f"现金: ${self.cash:,.2f}")
        print(f"持仓数量: {len(self.positions)}")
        print(f"最大回撤: {self.drawdown*100:.2f}%")
        print(f"组合希腊值: {self.portfolio_greeks}")
        print("="*60)
        
        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': len(self.positions),
            'drawdown': self.drawdown,
            'portfolio_greeks': self.portfolio_greeks
        }

class SACAgent:
    """软演员评论家 (SAC) 智能体 - 完善版"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 创建网络
        self.actor = ActorNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic = CriticNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 冻结目标网络参数
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # 自动调节温度参数
        if config.autotune_alpha:
            self.log_alpha = torch.tensor(np.log(config.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.target_entropy = config.target_entropy
        else:
            self.log_alpha = torch.tensor(np.log(config.alpha), device=self.device)
            self.alpha_optimizer = None

        # 经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(
            config.replay_buffer_size,
            config.state_dim,
            config.action_dim,
            self.device
        )
        
        # 训练状态
        self.total_steps = 0
        self.update_steps = 0
        self.episode_rewards = []
        self.training_losses = []
        
        logger.info(f"SAC智能体初始化完成，隐藏层维度: {config.hidden_dim}")

    @property
    def alpha(self):
        """温度参数"""
        return self.log_alpha.exp().detach()

    def select_action(self, state, deterministic=False, explore=True):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if deterministic or not explore:
            with torch.no_grad():
                action = self.actor.get_action(state_tensor, deterministic=True)
        else:
            # 添加探索噪声
            with torch.no_grad():
                action, _, _, _ = self.actor(state_tensor, deterministic=False, with_logprob=False)
        
        action_np = action.cpu().numpy()[0]
        
        # 添加少量随机探索
        if explore and np.random.rand() < 0.1:
            action_np += np.random.normal(0, 0.1, size=action_np.shape)
            action_np = np.clip(action_np, -1.0, 1.0)
        
        return action_np

    def update(self, batch):
        """更新网络参数"""
        if batch is None:
            return None
        
        # 解包批次数据
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        weights = batch.get('weights', torch.ones_like(rewards))
        
        with torch.no_grad():
            # 计算目标Q值
            next_actions, next_log_probs, _, _ = self.actor(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * q_next

        # 计算当前Q值
        q1, q2 = self.critic(states, actions)
        
        # 计算TD误差（用于优先级）
        td_errors = torch.abs(q1 - target_q).squeeze().detach().cpu().numpy()
        
        # 计算Critic损失（带重要性采样权重）
        critic_loss1 = (weights.unsqueeze(1) * (q1 - target_q).pow(2)).mean()
        critic_loss2 = (weights.unsqueeze(1) * (q2 - target_q).pow(2)).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新Actor
        new_actions, log_probs, _, _ = self.actor(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 更新温度参数（如果自动调节）
        if self.config.autotune_alpha and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)

        # 软更新目标网络
        self._soft_update(self.critic, self.critic_target, self.config.tau)
        
        # 记录损失
        loss_info = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.config.autotune_alpha else 0.0,
            'alpha': self.alpha.item(),
            'avg_q': q_new.mean().item(),
            'avg_log_prob': log_probs.mean().item()
        }
        
        self.training_losses.append(loss_info)
        self.update_steps += 1
        
        # 更新优先级
        if hasattr(self.replay_buffer, 'update_priorities'):
            self.replay_buffer.update_priorities(batch['indices'], td_errors)
        
        return loss_info

    def _soft_update(self, source, target, tau):
        """软更新目标网络"""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def train_step(self):
        """执行一步训练"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.config.batch_size)
        return self.update(batch)

    def save(self, path, save_buffer=False):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'total_steps': self.total_steps,
            'update_steps': self.update_steps,
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses,
            'config': self.config.to_dict()
        }
        
        if self.config.autotune_alpha and self.alpha_optimizer is not None:
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        
        if save_buffer:
            buffer_path = path.replace('.pth', '_buffer.pkl')
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.replay_buffer, f)
        
        logger.info(f"模型保存到: {path}")

    def load(self, path, load_buffer=False):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.total_steps = checkpoint['total_steps']
        self.update_steps = checkpoint['update_steps']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
        
        if self.config.autotune_alpha and 'alpha_optimizer_state_dict' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        if load_buffer:
            buffer_path = path.replace('.pth', '_buffer.pkl')
            if os.path.exists(buffer_path):
                with open(buffer_path, 'rb') as f:
                    self.replay_buffer = pickle.load(f)
        
        logger.info(f"从 {path} 加载模型，总步数: {self.total_steps}, 更新步数: {self.update_steps}")

class Trainer:
    """智能体训练器 - 完善版"""
    
    def __init__(self, config, env, agent):
        """初始化训练器"""
        self.config = config
        self.env = env
        self.agent = agent
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_portfolio_values = []
        self.episode_drawdowns = []
        self.best_reward = -np.inf
        self.best_model_path = None
        
        # 训练进度跟踪
        self.start_time = None
        self.current_episode = 0
        
        # 创建必要的目录
        os.makedirs("models", exist_ok=True)
        os.makedirs("config", exist_ok=True)
        os.makedirs("stats", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info(f"训练器初始化完成，最大episodes: {config.max_episodes}")

    def train(self):
        """完整训练过程"""
        self.start_time = time.time()
        
        logger.info("开始训练循环...")
        logger.info(f"配置参数: episodes={self.config.max_episodes}, "
                  f"steps={self.config.max_steps}, "
                  f"batch_size={self.config.batch_size}")
        
        for episode in range(self.config.max_episodes):
            self.current_episode = episode
            
            # 训练一个episode
            episode_reward, episode_length, info = self.train_episode(episode)
            
            # 记录统计
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_portfolio_values.append(info['portfolio_value'])
            self.episode_drawdowns.append(info.get('drawdown', 0.0))

            # 定期打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                avg_value = np.mean(self.episode_portfolio_values[-10:]) if len(self.episode_portfolio_values) >= 10 else np.mean(self.episode_portfolio_values)
                avg_drawdown = np.mean(self.episode_drawdowns[-10:]) if len(self.episode_drawdowns) >= 10 else np.mean(self.episode_drawdowns)
                
                # 计算训练时间
                elapsed_time = time.time() - self.start_time
                remaining_episodes = self.config.max_episodes - episode - 1
                estimated_total_time = elapsed_time / (episode + 1) * self.config.max_episodes
                estimated_remaining_time = estimated_total_time - elapsed_time
                
                logger.info(f"Episode {episode:4d}/{self.config.max_episodes} | 平均奖励: {avg_reward:.2f} | 平均价值: ${avg_value:,.2f} | 平均回撤: {avg_drawdown*100:.2f}%")
            
            # 定期评估
            if episode % self.config.eval_freq == 0 and episode > 0:
                self._perform_evaluation(episode)
            
            # 定期保存检查点
            if episode % self.config.save_freq == 0 and episode > 0:
                self._save_checkpoint(episode)
        
        # 训练完成后的处理
        return self._finalize_training()
    
    def train_episode(self, episode):
        """训练单个episode"""
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        for step in range(self.config.max_steps):
            # 选择动作：在预热阶段使用随机动作，否则使用策略
            if self.agent.total_steps < self.config.warmup_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(state, deterministic=False, explore=True)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            self.agent.total_steps += 1
            
            # 执行训练步骤
            if len(self.agent.replay_buffer) >= self.config.batch_size:
                for _ in range(self.config.gradient_steps):
                    self.agent.train_step()
            
            if done:
                break
        
        # 更新episode奖励记录
        self.agent.episode_rewards.append(episode_reward)
        
        # 更新组合价值和回撤记录
        portfolio_value = self.env._calculate_portfolio_value()
        self.episode_portfolio_values.append(portfolio_value)
        self.episode_drawdowns.append(self.env.drawdown)
        
        return episode_reward, step + 1, info
    
    def _perform_evaluation(self, episode):
        """执行评估"""
        eval_rewards, eval_stats = self.evaluate(num_episodes=3, render=False)
        avg_eval_reward = np.mean(eval_rewards)
        
        # 保存最佳模型
        if avg_eval_reward > self.best_reward:
            self.best_reward = avg_eval_reward
            self.best_model_path = f"models/best_model_episode_{episode}_reward{avg_eval_reward:.1f}.pth"
            self.agent.save(self.best_model_path, save_buffer=False)
            logger.info(f"保存最佳模型到: {self.best_model_path}")
    
    def _save_checkpoint(self, episode):
        """保存检查点"""
        model_path = f"models/checkpoint_episode_{episode}.pth"
        self.agent.save(model_path, save_buffer=False)
        self.save_stats(f"stats/checkpoint_stats_ep{episode}.pkl")
        logger.info(f"检查点保存到: {model_path}")
    
    def _finalize_training(self):
        """训练完成后的处理"""
        training_time = time.time() - self.start_time
        
        # 最终评估
        final_eval_rewards, final_eval_stats = self.evaluate(num_episodes=5, render=False)
        
        logger.info(f"训练完成，总时间: {training_time:.2f}秒，平均每episode: {training_time/self.config.max_episodes:.2f}秒")
        logger.info(f"最终评估平均奖励: {np.mean(final_eval_rewards):.2f} ± {np.std(final_eval_rewards):.2f}")
        
        # 保存最终模型
        final_model_path = "models/trained_model_final.pth"
        self.agent.save(final_model_path, save_buffer=False)
        logger.info(f"最终模型保存到: {final_model_path}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_portfolio_values': self.episode_portfolio_values,
            'episode_drawdowns': self.episode_drawdowns,
            'best_reward': self.best_reward,
            'best_model_path': self.best_model_path,
            'training_time': training_time,
            'final_eval_rewards': final_eval_rewards,
            'final_eval_stats': final_eval_stats
        }

    def evaluate(self, num_episodes=10, render=False):
        """评估智能体"""
        eval_rewards = []
        eval_stats = []
        
        # 保存当前模式
        original_mode = self.env.mode
        self.env.mode = 'eval'
        
        logger.info(f"开始评估，共 {num_episodes} 个episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            for step in range(self.config.max_steps):
                # 使用确定性策略
                action = self.agent.select_action(state, deterministic=True, explore=False)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                
                if render and step % 10 == 0:
                    self.env.render()
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_stats.append({
                'portfolio_value': info['portfolio_value'],
                'episode_length': step + 1,
                'drawdown': info.get('drawdown', 0.0),
                'positions': info.get('positions', 0),
                'portfolio_greeks': info.get('portfolio_greeks', {}),
                'cash': info.get('cash', 0.0),
                'step': step + 1
            })
            
            if episode == 0 or (episode + 1) % 2 == 0:
                logger.info(f"评估 Episode {episode}: 奖励={episode_reward:.2f}, "
                          f"组合价值=${info['portfolio_value']:,.2f}, "
                          f"持仓={info.get('positions', 0)}, "
                          f"现金=${info.get('cash', 0):,.2f}")
        
        # 恢复原始模式
        self.env.mode = original_mode
        
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_value = np.mean([s['portfolio_value'] for s in eval_stats])
        avg_drawdown = np.mean([s['drawdown'] for s in eval_stats])
        
        logger.info(f"评估完成: 平均奖励={avg_reward:.2f} ± {std_reward:.2f}, "
                  f"平均组合价值=${avg_value:,.2f}, "
                  f"平均回撤={avg_drawdown*100:.2f}%")
        
        return eval_rewards, eval_stats
    
    def save_stats(self, path="stats/training_stats.pkl"):
        """保存训练统计"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_portfolio_values': self.episode_portfolio_values,
            'episode_drawdowns': self.episode_drawdowns,
            'best_reward': self.best_reward,
            'best_model_path': self.best_model_path,
            'training_losses': self.agent.training_losses,
            'config': self.config.to_dict(),
            'total_training_steps': self.agent.total_steps,
            'current_episode': self.current_episode
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"训练统计保存到: {path}")

    def load_stats(self, path="stats/training_stats.pkl"):
        """加载训练统计"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                stats = pickle.load(f)
            
            self.episode_rewards = stats.get('episode_rewards', [])
            self.episode_lengths = stats.get('episode_lengths', [])
            self.episode_portfolio_values = stats.get('episode_portfolio_values', [])
            self.episode_drawdowns = stats.get('episode_drawdowns', [])
            self.best_reward = stats.get('best_reward', -np.inf)
            self.best_model_path = stats.get('best_model_path', None)
            
            # 更新agent的统计
            if 'training_losses' in stats:
                self.agent.training_losses = stats['training_losses']
            if 'total_training_steps' in stats:
                self.agent.total_steps = stats['total_training_steps']
            
            logger.info(f"从 {path} 加载训练统计，共 {len(self.episode_rewards)} 个episodes")
            return stats
        else:
            logger.warning(f"训练统计文件不存在: {path}")
            return None
    
    def plot_training_progress(self, save_path="plots/training_progress.png"):
        """绘制训练进度图"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # 创建图形
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(3, 3, figure=fig)
            
            # 1. 奖励曲线（左上）
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.episode_rewards, color='blue', alpha=0.7)
            ax1.set_title('Episode Rewards', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
            
            # 添加移动平均线
            if len(self.episode_rewards) >= 20:
                window = 20
                ma_rewards = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.episode_rewards)), ma_rewards, 
                        color='red', linewidth=2, label=f'MA ({window})')
                ax1.legend()
            
            # 2. 组合价值曲线（中上）
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.episode_portfolio_values, color='green', alpha=0.7)
            ax2.axhline(y=self.config.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
            ax2.set_title('Portfolio Value', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Value ($)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. 回撤曲线（右上）
            ax3 = fig.add_subplot(gs[0, 2])
            drawdown_pct = [d * 100 for d in self.episode_drawdowns]
            ax3.plot(drawdown_pct, color='orange', alpha=0.7)
            ax3.fill_between(range(len(drawdown_pct)), drawdown_pct, 0, color='orange', alpha=0.2)
            ax3.set_title('Max Drawdown (%)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
            
            # 4. 损失曲线（左下）
            ax4 = fig.add_subplot(gs[1, :])
            if hasattr(self.agent, 'training_losses') and self.agent.training_losses:
                losses = self.agent.training_losses
                # 取最近2000步
                recent_losses = losses[-2000:] if len(losses) > 2000 else losses
                
                critic_losses = [l.get('critic_loss', 0) for l in recent_losses]
                actor_losses = [l.get('actor_loss', 0) for l in recent_losses]
                alpha_losses = [l.get('alpha_loss', 0) for l in recent_losses]
                
                steps = range(len(recent_losses))
                ax4.plot(steps, critic_losses, label='Critic Loss', alpha=0.7)
                ax4.plot(steps, actor_losses, label='Actor Loss', alpha=0.7)
                ax4.plot(steps, alpha_losses, label='Alpha Loss', alpha=0.7)
                ax4.set_title('Training Losses (Last 2000 Updates)', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Update Step')
                ax4.set_ylabel('Loss')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No loss data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Training Losses', fontsize=12, fontweight='bold')
            
            # 5. Alpha值变化（中下）
            ax5 = fig.add_subplot(gs[2, 0])
            if hasattr(self.agent, 'training_losses') and self.agent.training_losses:
                losses = self.agent.training_losses
                alphas = [l.get('alpha', 0.2) for l in losses[-1000:]]
                ax5.plot(alphas, color='purple', alpha=0.7)
                ax5.set_title('Alpha (Temperature) Parameter', fontsize=12, fontweight='bold')
                ax5.set_xlabel('Update Step')
                ax5.set_ylabel('Alpha')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No alpha data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax5.transAxes, fontsize=12)
                ax5.set_title('Alpha Parameter', fontsize=12, fontweight='bold')
            
            # 6. Q值变化（中中）
            ax6 = fig.add_subplot(gs[2, 1])
            if hasattr(self.agent, 'training_losses') and self.agent.training_losses:
                losses = self.agent.training_losses
                avg_q = [l.get('avg_q', 0) for l in losses[-1000:]]
                ax6.plot(avg_q, color='brown', alpha=0.7)
                ax6.set_title('Average Q Value', fontsize=12, fontweight='bold')
                ax6.set_xlabel('Update Step')
                ax6.set_ylabel('Q Value')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No Q-value data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax6.transAxes, fontsize=12)
                ax6.set_title('Average Q Value', fontsize=12, fontweight='bold')
            
            # 7. 夏普比率估算（右下）
            ax7 = fig.add_subplot(gs[2, 2])
            if len(self.episode_portfolio_values) >= 20:
                returns = []
                for i in range(1, len(self.episode_portfolio_values)):
                    ret = (self.episode_portfolio_values[i] - self.episode_portfolio_values[i-1]) / self.episode_portfolio_values[i-1]
                    returns.append(ret)
                
                rolling_sharpe = []
                window = 20
                for i in range(window, len(returns)):
                    window_returns = returns[i-window:i]
                    if np.std(window_returns) > 0:
                        sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                        rolling_sharpe.append(sharpe)
                
                if rolling_sharpe:
                    ax7.plot(range(window, len(returns)), rolling_sharpe, color='teal', alpha=0.7)
                    ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax7.set_title('Rolling Sharpe Ratio (20 episodes)', fontsize=12, fontweight='bold')
                    ax7.set_xlabel('Episode')
                    ax7.set_ylabel('Sharpe Ratio')
                    ax7.grid(True, alpha=0.3)
                else:
                    ax7.text(0.5, 0.5, 'Insufficient data for Sharpe ratio', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax7.transAxes, fontsize=12)
                    ax7.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
            else:
                ax7.text(0.5, 0.5, 'Insufficient episodes for analysis', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax7.transAxes, fontsize=12)
                ax7.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
            
            plt.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"训练进度图保存到: {save_path}")
            return True
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制图表。请运行: pip install matplotlib")
            return False
        except Exception as e:
            logger.error(f"绘制训练进度图时出错: {str(e)}", exc_info=True)
            return False


def main():
    """主函数"""
    print("=" * 70)
    print("期权交易策略智能体系统 - 完善版")
    print("=" * 70)
    
    try:
        # 创建必要目录
        os.makedirs("models", exist_ok=True)
        os.makedirs("config", exist_ok=True)
        os.makedirs("stats", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # 创建配置
        config = AgentConfig()
        
        # 创建环境
        env = OptionTradingEnv(config, mode='train')
        
        # 创建智能体
        agent = SACAgent(config)
        
        # 创建训练器
        trainer = Trainer(config, env, agent)
        
        # 训练
        print("\\n开始训练智能体...")
        stats = trainer.train()
        
        # 保存最终模型
        final_model_path = "models/option_trading_agent_final.pth"
        agent.save(final_model_path, save_buffer=False)
        
        # 保存配置和统计
        config.save("config/agent_config.json")
        trainer.save_stats("stats/training_stats.pkl")
        
        # 绘制训练进度图
        trainer.plot_training_progress("plots/training_progress.png")
        
        # 评估最终模型
        print("\\n评估最终模型...")
        eval_rewards, eval_stats = trainer.evaluate(num_episodes=5, render=False)
        
        # 总结
        print("\\n" + "="*70)
        print("训练总结:")
        print(f"  总训练episodes: {len(stats['episode_rewards'])}")
        print(f"  最终平均评估奖励: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  最佳模型: {trainer.best_model_path}")
        print(f"  最佳评估奖励: {trainer.best_reward:.2f}")
        print(f"  最终模型保存到: {final_model_path}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\\n训练被用户中断")
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"主程序运行出错: {str(e)}", exc_info=True)
        print(f"程序运行出错: {str(e)}")
        raise

def quick_test():
    """快速测试"""
    print("快速测试期权交易智能体...")
    
    # 创建必要目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 简化配置
    config = AgentConfig()
    config.max_episodes = 50
    config.max_steps = 100
    config.warmup_steps = 500
    config.eval_freq = 10
    config.save_freq = 25
    
    # 创建环境
    env = OptionTradingEnv(config, mode='train')
    
    # 创建智能体
    agent = SACAgent(config)
    
    # 简单训练
    print("训练中...")
    trainer = Trainer(config, env, agent)
    
    for episode in range(config.max_episodes):
        episode_reward, episode_length, info = trainer.train_episode(episode)
        
        if episode % 5 == 0:
            print(f"Episode {episode:3d}: "
                  f"Reward={episode_reward:8.2f}, "
                  f"Value=${info['portfolio_value']:,.2f}, "
                  f"Steps={episode_length:3d}, "
                  f"Drawdown={info['drawdown']*100:5.2f}%")
    
    print("快速训练完成！")
    
    # 测试智能体
    print("\\n测试智能体...")
    state = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = agent.select_action(state, deterministic=True, explore=False)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            env.render()
        
        if done:
            break
    
    print(f"测试完成！总奖励: {total_reward:.2f}")
    print(f"最终组合价值: ${info['portfolio_value']:,.2f}")
    
    # 保存测试模型
    test_model_path = "models/test_model.pth"
    agent.save(test_model_path, save_buffer=False)
    print(f"测试模型保存到: {test_model_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="期权交易策略智能体")
    parser.add_argument("--mode", type=str, default="test", 
                       choices=["full", "test", "train", "eval", "deploy", "analyze"], 
                       help="运行模式")
    parser.add_argument("--model_path", type=str, default=None,
                       help="模型路径")
    parser.add_argument("--config_path", type=str, default=None,
                       help="配置路径")
    parser.add_argument("--episodes", type=int, default=None,
                       help="训练episodes数")
    
    args = parser.parse_args()
    
    # 创建必要目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    if args.mode == "full":
        main()
    elif args.mode == "test":
        quick_test()
    elif args.mode == "train":
        # 训练模式
        config = AgentConfig()
        if args.config_path:
            config.load(args.config_path)
        
        if args.episodes:
            config.max_episodes = args.episodes
        
        env = OptionTradingEnv(config, mode='train')
        agent = SACAgent(config)
        
        if args.model_path and os.path.exists(args.model_path):
            agent.load(args.model_path)
            print(f"从 {args.model_path} 加载模型继续训练")
        
        trainer = Trainer(config, env, agent)
        stats = trainer.train()
        
        # 保存最终模型
        final_path = args.model_path.replace(".pth", "_continued.pth") if args.model_path else "models/trained_model.pth"
        agent.save(final_path, save_buffer=False)
        print(f"训练完成，模型保存到: {final_path}")
    elif args.mode == "eval":
        # 评估模式
        config = AgentConfig()
        if args.config_path:
            config.load(args.config_path)
        
        env = OptionTradingEnv(config, mode='eval')
        agent = SACAgent(config)
        
        if args.model_path and os.path.exists(args.model_path):
            agent.load(args.model_path)
        else:
            raise ValueError("评估模式需要指定模型路径")
        
        trainer = Trainer(config, env, agent)
        eval_rewards, eval_stats = trainer.evaluate(num_episodes=10, render=True)
        
        print(f"\\n评估结果:")
        print(f"  平均奖励: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  平均组合价值: ${np.mean([s['portfolio_value'] for s in eval_stats]):,.2f}")
        print(f"  平均回撤: {np.mean([s['drawdown'] for s in eval_stats])*100:.2f}%")
    elif args.mode == "deploy":
        # 部署模式
        print("部署模式 - 期权交易智能体")
        
        try:
            # 加载配置
            config = AgentConfig()
            if args.config_path:
                config.load(args.config_path)
            
            # 加载模型
            if not args.model_path:
                raise ValueError("部署模式需要指定模型路径")
            
            # 创建智能体
            agent = SACAgent(config)
            agent.load(args.model_path)
            
            # 创建环境
            env = OptionTradingEnv(config, mode='deploy')
            
            # 部署循环
            print(f"部署模式启动，使用模型: {args.model_path}")
            print("按 Ctrl+C 停止部署")
            
            # 初始化
            state = env.reset()
            episode_count = 0
            
            # 模拟实时数据处理
            while True:
                # 选择动作（确定性）
                action = agent.select_action(state, deterministic=True, explore=False)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 显示信息
                if env.current_step % 5 == 0:
                    print(f"\\n步骤 {env.current_step}:")
                    print(f"  组合价值: ${info['portfolio_value']:,.2f}")
                    print(f"  持仓: {info['positions']}个")
                    print(f"  回撤: {info['drawdown']:.2%}")
                    print(f"  希腊值: delta={info['portfolio_greeks']['delta']:.2f}, "
                          f"gamma={info['portfolio_greeks']['gamma']:.4f}, "
                          f"theta={info['portfolio_greeks']['theta']:.2f}, "
                          f"vega={info['portfolio_greeks']['vega']:.2f}")
                
                # 更新状态
                state = next_state
                
                # 检查是否需要重置
                if done:
                    episode_count += 1
                    print(f"\\n部署周期 {episode_count} 结束，重置环境...")
                    state = env.reset()
                
                # 模拟实时数据间隔
                time.sleep(1)  # 实际部署中可以替换为实时数据更新
            
        except KeyboardInterrupt:
            print("\\n部署模式被用户中断")
        except Exception as e:
            logger.error(f"部署模式运行出错: {str(e)}", exc_info=True)
            raise
    elif args.mode == "analyze":
        # 分析模式
        print("分析模式 - 分析训练结果")
        
        if not args.model_path:
            raise ValueError("分析模式需要指定模型路径")
        
        # 加载模型
        config = AgentConfig()
        agent = SACAgent(config)
        agent.load(args.model_path)
        
        print(f"模型分析: {args.model_path}")
        print(f"总训练步数: {agent.total_steps}")
        print(f"总更新步数: {agent.update_steps}")
        print(f"训练episodes数: {len(agent.episode_rewards)}")
        
        if agent.episode_rewards:
            print(f"平均episode奖励: {np.mean(agent.episode_rewards):.2f}")
            print(f"最佳episode奖励: {np.max(agent.episode_rewards):.2f}")
            print(f"最差episode奖励: {np.min(agent.episode_rewards):.2f}")
        
        # 加载训练统计
        stats_path = args.model_path.replace(".pth", "_stats.pkl")
        if os.path.exists(stats_path):
            trainer = Trainer(config, None, agent)
            stats = trainer.load_stats(stats_path)
            print(f"训练统计已加载")