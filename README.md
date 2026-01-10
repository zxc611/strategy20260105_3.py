# 期权自动交易策略 (Options Automated Trading Strategy)

一个功能完整、模块化的期权自动交易策略系统，适用于中国期权市场。

## 功能特点

### 核心功能
- **多策略支持**: 支持多种期权策略（看涨、看跌、价差、跨式等）
- **自动信号生成**: 基于技术指标自动生成交易信号
- **智能风险管理**: 完善的仓位管理和风险控制机制
- **实时持仓监控**: 自动止损止盈、到期管理
- **模块化设计**: 各功能模块独立，易于扩展和维护

### 技术特性
- 趋势分析 (移动平均线)
- 动量指标 (RSI)
- 波动率监控
- 多维度信号综合
- 凯利公式仓位管理
- 滑点控制
- 交易日志记录

## 项目结构

```
strategy20260105_3.py/
├── strategy.py           # 主策略文件
├── config.py            # 配置文件
├── data_fetcher.py      # 数据获取模块
├── signal_generator.py  # 信号生成模块
├── risk_manager.py      # 风险管理模块
├── position_manager.py  # 仓位管理模块
├── order_executor.py    # 订单执行模块
├── utils.py             # 工具函数
├── example.py           # 使用示例
└── requirements.txt     # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基础使用

```python
from strategy import OptionsStrategy

# 创建策略实例
strategy = OptionsStrategy()

# 运行策略
symbols = ['510050', '510300']  # 50ETF, 沪深300ETF
strategy.run_strategy(symbols)
```

### 3. 运行示例

```bash
python example.py
```

## 详细说明

### 配置参数

在 `config.py` 中可以调整以下参数：

#### 交易参数
- `INITIAL_CAPITAL`: 初始资金 (默认: 1,000,000)
- `MAX_POSITION_SIZE`: 单个持仓最大占比 (默认: 0.2)
- `MAX_TOTAL_POSITION`: 总持仓最大占比 (默认: 0.8)

#### 风险管理
- `STOP_LOSS_PERCENT`: 止损比例 (默认: 0.15)
- `TAKE_PROFIT_PERCENT`: 止盈比例 (默认: 0.30)
- `MAX_DAILY_LOSS`: 单日最大亏损比例 (默认: 0.05)

#### 期权参数
- `MIN_DAYS_TO_EXPIRY`: 最小到期天数 (默认: 7)
- `MAX_DAYS_TO_EXPIRY`: 最大到期天数 (默认: 90)
- `PREFERRED_DELTA_RANGE`: 优选Delta范围 (默认: 0.3-0.7)
- `MIN_OPEN_INTEREST`: 最小持仓量 (默认: 100)

### 支持的期权策略

1. **LONG_CALL** - 买入看涨期权
   - 适用: 看涨趋势 + 低波动率
   
2. **LONG_PUT** - 买入看跌期权
   - 适用: 看跌趋势 + 低波动率
   
3. **BULL_CALL_SPREAD** - 牛市看涨价差
   - 适用: 看涨趋势 + 高波动率
   
4. **BEAR_PUT_SPREAD** - 熊市看跌价差
   - 适用: 看跌趋势 + 高波动率
   
5. **STRADDLE** - 跨式策略
   - 适用: 中性趋势 + 高波动率
   
6. **IRON_CONDOR** - 铁秃鹰策略
   - 适用: 中性趋势 + 正常波动率

### 风险管理机制

1. **仓位控制**
   - 单个持仓限制
   - 总持仓限制
   - 基于波动率的动态仓位调整

2. **止损止盈**
   - 自动止损: 默认15%
   - 自动止盈: 默认30%
   - 可自定义阈值

3. **风险监控**
   - 单日最大亏损限制
   - 持仓量要求
   - Delta范围控制
   - 到期时间管理

### 使用示例

#### 示例1: 自定义配置

```python
from strategy import OptionsStrategy
from config import Config

# 创建自定义配置
config = Config()
config.INITIAL_CAPITAL = 500000
config.STOP_LOSS_PERCENT = 0.10
config.TAKE_PROFIT_PERCENT = 0.25

# 使用自定义配置创建策略
strategy = OptionsStrategy(config)
strategy.run_strategy(['510050'])
```

#### 示例2: 查看持仓和绩效

```python
# 运行策略后查看持仓
positions_df = strategy.get_positions_dataframe()
print(positions_df)

# 查看绩效指标
metrics = strategy.position_manager.get_performance_metrics()
print(f"总盈亏: {metrics['total_pnl']}")
print(f"收益率: {metrics['return_percent']}%")
print(f"胜率: {metrics['win_rate']}%")
```

#### 示例3: 信号分析

```python
# 仅分析信号，不执行交易
signals = strategy.signal_generator.generate_signals('510050')
print(f"综合信号: {signals['combined_signal']}")
print(f"推荐策略: {signals['recommended_strategy']}")
```

## 模块说明

### strategy.py - 主策略
整合所有模块，执行完整的交易流程：
1. 生成交易信号
2. 筛选符合条件的期权
3. 风险检查
4. 执行交易
5. 管理现有持仓
6. 生成绩效报告

### data_fetcher.py - 数据获取
- 获取期权链数据
- 获取标的价格
- 获取历史数据
- 计算技术指标 (MA, RSI, 波动率)

### signal_generator.py - 信号生成
- 趋势分析
- 动量分析
- 波动率分析
- 综合信号生成
- 策略推荐

### risk_manager.py - 风险管理
- 持仓限制检查
- 止损止盈判断
- 仓位大小计算
- 期权筛选标准验证
- 日盈亏跟踪

### position_manager.py - 仓位管理
- 持仓增加/关闭
- 持仓价格更新
- 盈亏计算
- 绩效统计
- 持仓报表生成

### order_executor.py - 订单执行
- 订单创建
- 订单执行
- 滑点控制
- 订单状态管理
- 订单统计

### utils.py - 工具函数
- 日志设置
- 到期天数计算
- 市场开盘检查
- 格式化函数
- 绩效指标计算

## 注意事项

1. **数据源**: 当前使用模拟数据，实际使用需接入真实行情数据源
2. **交易接口**: 需要对接实际的期权交易接口
3. **回测**: 建议先进行充分的历史数据回测
4. **风险**: 期权交易存在高风险，请谨慎使用
5. **合规**: 确保符合当地监管要求

## 扩展开发

### 添加新的交易策略

在 `signal_generator.py` 的 `_recommend_option_strategy` 方法中添加新策略逻辑。

### 接入真实数据

修改 `data_fetcher.py` 中的方法，接入实际的行情数据 API。

### 自定义风险规则

在 `risk_manager.py` 中添加新的风险检查方法。

## 技术支持

- 问题反馈: 请提交 Issue
- 功能建议: 欢迎 Pull Request

## 免责声明

本软件仅用于学习和研究目的。实际交易存在风险，使用本软件进行交易所产生的任何损失由使用者自行承担。

## 许可证

MIT License

---

**作者**: zxc611  
**更新时间**: 2026-01-10
