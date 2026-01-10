"""
示例使用 - Example Usage
演示如何使用期权自动交易策略系统
"""

from strategy import OptionsStrategy
from config import Config


def example_basic_usage():
    """基础使用示例"""
    print("="*60)
    print("示例 1: 基础使用")
    print("="*60)
    
    # 创建策略实例
    strategy = OptionsStrategy()
    
    # 运行策略
    symbols = ['510050', '510300']  # 50ETF, 沪深300ETF
    strategy.run_strategy(symbols)
    
    print("\n策略运行完成!")


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "="*60)
    print("示例 2: 自定义配置")
    print("="*60)
    
    # 创建自定义配置
    config = Config()
    config.INITIAL_CAPITAL = 500000  # 50万初始资金
    config.MAX_POSITION_SIZE = 0.15  # 单个持仓最大15%
    config.STOP_LOSS_PERCENT = 0.10  # 止损10%
    config.TAKE_PROFIT_PERCENT = 0.25  # 止盈25%
    
    # 创建策略实例
    strategy = OptionsStrategy(config)
    
    # 运行策略
    symbols = ['159919']  # 创业板ETF
    strategy.run_strategy(symbols)
    
    print("\n自定义配置策略运行完成!")


def example_check_positions():
    """查看持仓示例"""
    print("\n" + "="*60)
    print("示例 3: 查看持仓和绩效")
    print("="*60)
    
    # 创建策略实例
    strategy = OptionsStrategy()
    
    # 运行策略
    symbols = ['510050']
    strategy.run_strategy(symbols)
    
    # 获取持仓信息
    print("\n当前持仓详情:")
    positions_df = strategy.get_positions_dataframe()
    if not positions_df.empty:
        print(positions_df.to_string())
    else:
        print("  暂无持仓")
    
    # 获取已平仓信息
    print("\n已平仓详情:")
    closed_positions_df = strategy.get_closed_positions_dataframe()
    if not closed_positions_df.empty:
        print(closed_positions_df.to_string())
    else:
        print("  暂无平仓记录")
    
    # 获取绩效指标
    metrics = strategy.position_manager.get_performance_metrics()
    print("\n绩效指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def example_multiple_runs():
    """多次运行示例"""
    print("\n" + "="*60)
    print("示例 4: 模拟多个交易日运行")
    print("="*60)
    
    # 创建策略实例
    strategy = OptionsStrategy()
    
    # 模拟5个交易日
    symbols = ['510050', '510300']
    for day in range(1, 6):
        print(f"\n第{day}个交易日:")
        print("-" * 40)
        
        strategy.run_strategy(symbols)
        
        # 重置日盈亏
        strategy.risk_manager.reset_daily_pnl()
    
    print("\n多日交易模拟完成!")


def example_signal_analysis():
    """信号分析示例"""
    print("\n" + "="*60)
    print("示例 5: 单独分析交易信号")
    print("="*60)
    
    # 创建策略实例
    strategy = OptionsStrategy()
    
    # 分析多个标的的信号
    symbols = ['510050', '510300', '159919']
    
    for symbol in symbols:
        print(f"\n{symbol} 的交易信号:")
        print("-" * 40)
        
        # 生成信号
        signals = strategy.signal_generator.generate_signals(symbol)
        
        # 显示信号详情
        print(f"  趋势信号: {signals['trend_signal']}")
        print(f"  动量信号: {signals['momentum_signal']}")
        print(f"  波动率信号: {signals['volatility_signal']}")
        print(f"  综合信号: {signals['combined_signal']}")
        print(f"  信号强度: {signals['signal_strength']:.2%}")
        print(f"  推荐策略: {signals['recommended_strategy']}")


def example_risk_management():
    """风险管理示例"""
    print("\n" + "="*60)
    print("示例 6: 风险管理功能演示")
    print("="*60)
    
    config = Config()
    strategy = OptionsStrategy(config)
    
    # 模拟持仓
    print("\n测试风险管理规则:")
    print("-" * 40)
    
    total_capital = strategy.position_manager.get_total_value()
    print(f"总资金: ¥{total_capital:,.2f}")
    
    # 测试持仓限制
    test_position_value = total_capital * 0.25
    can_open = strategy.risk_manager.check_position_limit(test_position_value, total_capital)
    print(f"测试开仓 (25%资金): {'允许' if can_open else '拒绝'}")
    
    # 测试止损
    entry_price = 100
    current_price = 80
    should_stop = strategy.risk_manager.check_stop_loss(entry_price, current_price, 'LONG')
    print(f"测试止损 (入场100, 当前80): {'触发' if should_stop else '未触发'}")
    
    # 测试止盈
    current_price = 135
    should_take = strategy.risk_manager.check_take_profit(entry_price, current_price, 'LONG')
    print(f"测试止盈 (入场100, 当前135): {'触发' if should_take else '未触发'}")


def main():
    """运行所有示例"""
    try:
        # 示例1: 基础使用
        example_basic_usage()
        
        # 示例2: 自定义配置
        example_custom_config()
        
        # 示例3: 查看持仓
        example_check_positions()
        
        # 示例4: 多次运行
        # example_multiple_runs()  # 可选：耗时较长
        
        # 示例5: 信号分析
        example_signal_analysis()
        
        # 示例6: 风险管理
        example_risk_management()
        
        print("\n" + "="*60)
        print("所有示例运行完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
