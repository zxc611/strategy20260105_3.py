"""
期权自动交易策略主程序 - Options Automated Trading Strategy
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from config import Config
from data_fetcher import MarketDataFetcher
from risk_manager import RiskManager
from signal_generator import SignalGenerator
from position_manager import PositionManager, Position
from order_executor import OrderExecutor
from utils import setup_logger, calculate_days_to_expiry, is_market_open, format_currency, format_percent


class OptionsStrategy:
    """期权自动交易策略"""
    
    def __init__(self, config: Config = None):
        """
        初始化策略
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.logger = setup_logger(self.config.LOG_FILE, self.config.LOG_LEVEL)
        
        # 初始化各个模块
        self.data_fetcher = MarketDataFetcher()
        self.risk_manager = RiskManager(self.config)
        self.signal_generator = SignalGenerator(self.config, self.data_fetcher)
        self.position_manager = PositionManager(self.config.INITIAL_CAPITAL)
        self.order_executor = OrderExecutor(self.config.MAX_SLIPPAGE)
        
        self.logger.info("期权自动交易策略初始化完成")
        self.logger.info(f"初始资金: {format_currency(self.config.INITIAL_CAPITAL)}")
        
    def run_strategy(self, symbols: List[str]):
        """
        运行策略
        
        Args:
            symbols: 标的代码列表
        """
        self.logger.info("="*60)
        self.logger.info("开始运行期权自动交易策略")
        self.logger.info("="*60)
        
        # 检查市场是否开盘
        if not is_market_open():
            self.logger.warning("市场未开盘，策略暂停")
            return
        
        for symbol in symbols:
            self.logger.info(f"\n处理标的: {symbol}")
            
            try:
                # 生成交易信号
                signals = self.signal_generator.generate_signals(symbol)
                self._log_signals(signals)
                
                # 执行交易决策
                self._execute_trading_decision(symbol, signals)
                
                # 管理现有持仓
                self._manage_existing_positions(symbol)
                
            except Exception as e:
                self.logger.error(f"处理标的 {symbol} 时出错: {str(e)}")
                continue
        
        # 显示绩效报告
        self._display_performance_report()
        
    def _log_signals(self, signals: Dict):
        """
        记录信号
        
        Args:
            signals: 信号字典
        """
        self.logger.info(f"  趋势信号: {signals['trend_signal']}")
        self.logger.info(f"  动量信号: {signals['momentum_signal']}")
        self.logger.info(f"  波动率信号: {signals['volatility_signal']}")
        self.logger.info(f"  综合信号: {signals['combined_signal']}")
        self.logger.info(f"  信号强度: {signals['signal_strength']:.2f}")
        self.logger.info(f"  推荐策略: {signals['recommended_strategy']}")
        
    def _execute_trading_decision(self, symbol: str, signals: Dict):
        """
        执行交易决策
        
        Args:
            symbol: 标的代码
            signals: 信号字典
        """
        # 获取期权链
        option_chain = self.data_fetcher.get_option_chain(symbol)
        
        # 筛选符合条件的期权
        suitable_options = self._filter_options(option_chain, signals)
        
        if suitable_options.empty:
            self.logger.info("  未找到符合条件的期权")
            return
        
        # 根据推荐策略选择期权
        strategy_type = signals['recommended_strategy']
        selected_option = self._select_option(suitable_options, strategy_type, signals)
        
        if selected_option is None:
            self.logger.info("  未能选择合适的期权")
            return
        
        # 检查风险管理
        if not self._check_risk_constraints(selected_option):
            self.logger.warning("  风险检查未通过，放弃交易")
            return
        
        # 创建并执行订单
        self._place_order(selected_option, strategy_type, signals)
        
    def _filter_options(self, option_chain: pd.DataFrame, signals: Dict) -> pd.DataFrame:
        """
        筛选期权
        
        Args:
            option_chain: 期权链
            signals: 信号字典
            
        Returns:
            筛选后的期权
        """
        filtered = option_chain.copy()
        
        # 根据信号筛选期权类型
        if signals['combined_signal'] == 'BUY':
            filtered = filtered[filtered['type'] == 'CALL']
        elif signals['combined_signal'] == 'SELL':
            filtered = filtered[filtered['type'] == 'PUT']
        
        # 筛选持仓量
        filtered = filtered[filtered['open_interest'] >= self.config.MIN_OPEN_INTEREST]
        
        # 筛选Delta范围
        filtered = filtered[
            (abs(filtered['delta']) >= self.config.PREFERRED_DELTA_RANGE[0]) &
            (abs(filtered['delta']) <= self.config.PREFERRED_DELTA_RANGE[1])
        ]
        
        # 筛选到期天数
        filtered = filtered.copy()
        filtered['days_to_expiry'] = filtered['expiry'].apply(calculate_days_to_expiry)
        filtered = filtered[
            (filtered['days_to_expiry'] >= self.config.MIN_DAYS_TO_EXPIRY) &
            (filtered['days_to_expiry'] <= self.config.MAX_DAYS_TO_EXPIRY)
        ]
        
        return filtered
        
    def _select_option(self, options: pd.DataFrame, strategy_type: str, 
                      signals: Dict) -> Optional[pd.Series]:
        """
        选择期权
        
        Args:
            options: 可选期权
            strategy_type: 策略类型
            signals: 信号字典
            
        Returns:
            选中的期权
        """
        if options.empty:
            return None
        
        # 根据策略类型选择
        if strategy_type in ['LONG_CALL', 'LONG_PUT']:
            # 选择Delta适中、流动性好的期权
            options = options.sort_values('open_interest', ascending=False)
            return options.iloc[0]
        elif strategy_type in ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']:
            # 选择价差策略的期权
            options = options.sort_values('delta', ascending=False)
            return options.iloc[0]
        else:
            # 其他策略选择流动性最好的
            options = options.sort_values('volume', ascending=False)
            return options.iloc[0]
        
    def _check_risk_constraints(self, option: pd.Series) -> bool:
        """
        检查风险约束
        
        Args:
            option: 期权数据
            
        Returns:
            是否通过检查
        """
        # 计算所需资金
        entry_price = (option['bid'] + option['ask']) / 2
        position_value = entry_price * 100  # 假设每手100股
        
        # 检查单个持仓限制
        total_capital = self.position_manager.get_total_value()
        if not self.risk_manager.check_position_limit(position_value, total_capital):
            self.logger.warning("  超过单个持仓限制")
            return False
        
        # 检查总持仓限制
        total_position_value = self.position_manager.get_position_value() + position_value
        if not self.risk_manager.check_total_position_limit(total_position_value, total_capital):
            self.logger.warning("  超过总持仓限制")
            return False
        
        # 检查日亏损限制
        if not self.risk_manager.check_daily_loss_limit(
            self.risk_manager.daily_pnl, total_capital
        ):
            self.logger.warning("  触发日亏损限制")
            return False
        
        return True
        
    def _place_order(self, option: pd.Series, strategy_type: str, signals: Dict):
        """
        下单
        
        Args:
            option: 期权数据
            strategy_type: 策略类型
            signals: 信号字典
        """
        # 计算入场价格和数量
        entry_price = (option['bid'] + option['ask']) / 2
        total_capital = self.position_manager.get_total_value()
        
        # 获取市场波动率
        indicators = self.data_fetcher.get_market_indicators(option['symbol'].split('_')[0])
        volatility = indicators.get('volatility', 0.2)
        
        quantity = self.risk_manager.calculate_position_size(
            entry_price, total_capital, volatility
        )
        
        # 创建订单
        order = self.order_executor.create_order(
            symbol=option['symbol'],
            order_type='BUY',
            quantity=quantity,
            price=entry_price,
            strategy=strategy_type
        )
        
        # 执行订单
        market_price = option['last']
        if self.order_executor.execute_order(order, market_price):
            # 创建持仓
            position = Position(
                symbol=option['symbol'],
                option_type=option['type'],
                quantity=quantity,
                entry_price=order.filled_price,
                strike=option['strike'],
                expiry=option['expiry']
            )
            
            if self.position_manager.add_position(position):
                self.logger.info(f"  成功开仓: {quantity} 手 {option['symbol']} @ {order.filled_price}")
            else:
                self.logger.error("  开仓失败: 资金不足")
                
    def _manage_existing_positions(self, symbol: str):
        """
        管理现有持仓
        
        Args:
            symbol: 标的代码
        """
        # 获取相关持仓
        positions_to_close = []
        
        for pos_symbol, position in self.position_manager.positions.items():
            if symbol not in pos_symbol:
                continue
            
            # 获取当前价格
            option_chain = self.data_fetcher.get_option_chain(symbol)
            current_option = option_chain[option_chain['symbol'] == pos_symbol]
            
            if current_option.empty:
                continue
            
            current_price = current_option.iloc[0]['last']
            position.update_price(current_price)
            
            # 检查止损
            if self.risk_manager.check_stop_loss(
                position.entry_price, current_price, 'LONG'
            ):
                self.logger.info(f"  触发止损: {pos_symbol}")
                positions_to_close.append((pos_symbol, current_price, '止损'))
                continue
            
            # 检查止盈
            if self.risk_manager.check_take_profit(
                position.entry_price, current_price, 'LONG'
            ):
                self.logger.info(f"  触发止盈: {pos_symbol}")
                positions_to_close.append((pos_symbol, current_price, '止盈'))
                continue
            
            # 检查到期时间
            days_to_expiry = calculate_days_to_expiry(position.expiry)
            if days_to_expiry <= 3:
                self.logger.info(f"  临近到期: {pos_symbol}")
                positions_to_close.append((pos_symbol, current_price, '临近到期'))
        
        # 平仓
        for pos_symbol, exit_price, reason in positions_to_close:
            closed = self.position_manager.close_position(pos_symbol, exit_price)
            if closed:
                self.logger.info(f"  平仓成功 ({reason}): {pos_symbol} @ {exit_price}, "
                               f"盈亏: {format_currency(closed['pnl'])} "
                               f"({format_percent(closed['pnl_percent'])})")
                
                # 更新日盈亏
                self.risk_manager.update_daily_pnl(closed['pnl'])
                
    def _display_performance_report(self):
        """显示绩效报告"""
        self.logger.info("\n" + "="*60)
        self.logger.info("绩效报告")
        self.logger.info("="*60)
        
        metrics = self.position_manager.get_performance_metrics()
        
        self.logger.info(f"总资产: {format_currency(metrics['total_value'])}")
        self.logger.info(f"现金: {format_currency(metrics['cash'])}")
        self.logger.info(f"持仓价值: {format_currency(metrics['position_value'])}")
        self.logger.info(f"总盈亏: {format_currency(metrics['total_pnl'])} "
                        f"({format_percent(metrics['return_percent'])})")
        self.logger.info(f"开仓数: {metrics['open_positions']}")
        self.logger.info(f"平仓数: {metrics['closed_positions']}")
        self.logger.info(f"胜率: {format_percent(metrics['win_rate'])}")
        self.logger.info(f"平均每笔盈亏: {format_currency(metrics['avg_pnl_per_trade'])}")
        
        # 显示持仓明细
        positions_summary = self.position_manager.get_positions_summary()
        if not positions_summary.empty:
            self.logger.info("\n当前持仓:")
            for _, pos in positions_summary.iterrows():
                self.logger.info(f"  {pos['symbol']}: {pos['quantity']}手, "
                               f"入场价: {pos['entry_price']:.2f}, "
                               f"当前价: {pos['current_price']:.2f}, "
                               f"盈亏: {format_currency(pos['pnl'])} "
                               f"({format_percent(pos['pnl_percent'])})")
        
        # 订单统计
        order_stats = self.order_executor.get_order_statistics()
        self.logger.info(f"\n订单统计:")
        self.logger.info(f"  总订单数: {order_stats['total_orders']}")
        self.logger.info(f"  成交订单: {order_stats['filled_orders']}")
        self.logger.info(f"  拒绝订单: {order_stats['rejected_orders']}")
        self.logger.info(f"  成交率: {format_percent(order_stats['fill_rate'])}")
        
    def get_positions_dataframe(self) -> pd.DataFrame:
        """获取持仓DataFrame"""
        return self.position_manager.get_positions_summary()
    
    def get_closed_positions_dataframe(self) -> pd.DataFrame:
        """获取已平仓DataFrame"""
        return self.position_manager.get_closed_positions_summary()


def main():
    """主函数"""
    # 创建策略实例
    strategy = OptionsStrategy()
    
    # 运行策略
    symbols = ['510050', '510300', '159919']  # 示例标的代码
    strategy.run_strategy(symbols)
    

if __name__ == '__main__':
    main()
