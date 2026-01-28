# ======================
# 新增导入项（修复缺失依赖问题）
# ======================
from typing import List, Dict, Tuple, Optional, Any  # 导入类型注解
from datetime import time, datetime, timedelta, date  # 处理日期时间
import hashlib  # 生成唯一标识符
import traceback  # 错误追踪
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import logging  # 日志记录
from cachetools import TTLCache  # 数据缓存
from pytz import timezone  # 时区处理
import holidays  # 节假日处理
import time as sys_time  # 用于睡眠函数
from operator import itemgetter  # 用于排序
from typing import Literal

from pythongo.base import BaseParams, BaseState, Field
from pythongo.classdef import KLineData, OrderData, TickData, TradeData
from pythongo.core import KLineStyleType
from pythongo.ui import BaseStrategy
from pythongo.utils import KLineGenerator

# 初始化日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================
# 类型定义（修复类型缺失问题）
# ======================
class Params(BaseParams):
    """参数映射模型"""
    exchange: str = Field(default="", title="交易所代码")
    instrument_id: str = Field(default="", title="合约代码")
    order_price: int | float = Field(default=0, title="报单价格")
    order_volume: int = Field(default=1, title="报单手数")
    order_direction: Literal["buy", "sell"] = Field(default="buy", title="报单方向")

class OptionContract:
    """期权合约类型声明"""
    def __init__(self, 
                 symbol: str,  # 合约代码
                 contract_type: str,  # 合约类型（CALL/PUT）
                 strike: float,  # 行权价
                 volume: int,  # 成交量
                 delta: float,  # Delta值
                 gamma: float,  # Gamma值
                 expired: bool,  # 是否到期
                 last_price: float, # 最后成交价
                 expiration: datetime):  # 到期日
        # 初始化合约属性
        self.symbol = symbol
        self.contract_type = contract_type
        self.strike = strike
        self.volume = volume
        self.delta = delta
        self.gamma = gamma
        self.expired = expired
        self.last_price = last_price
        self.expiration = expiration

class AccountInfo:
    """账户信息类型声明"""
    def __init__(self, cash: float, margin_ratio: float, positions: Dict[str, int]):  # 现金、保证金比例、持仓
        # 初始化账户信息
        self.cash = cash
        self.margin_ratio = margin_ratio
        self.positions = positions  # 持仓字典（合约代码：数量）

# 订单状态枚举
class OrderStatus:
    FILLED = 'filled'  # 已成交
    REJECTED = 'rejected'  # 已拒绝
    PENDING = 'pending'  # 待处理

# ======================
# 修正GlobalConfig类（修复类型不匹配问题）
# ======================
class GlobalConfig:
    SYMBOLS: List[str] = ["IF2403", "IC2403", "IH2403"]  # 交易标的
    OTM_THRESHOLD: float = 0.02  # 虚值期权阈值
    VOLUME_MULTIPLIERS: Tuple[float, float] = (5, 0.5)  # 成交量乘数
    MAX_POSITION: int = 10  # 最大持仓限制
    MAX_RISK_PER_TRADE: float = 0.02  # 单笔交易最大风险
    MARGIN_RATE_BASE: float = 0.15  # 基础保证金率
    SLIPPAGE: float = 0.0002  # 滑点
    MAX_DELTA: float = 0.25  # 最大delta限制
    EXPIRE_DAYS_WARNING: int = 3  # 到期日前警告天数
    STOP_LOSS_RATIO: float = 0.5  # 止损比例（50%）
    OPTION_CONTRACT_SIZE: int = 100  # 期权合约乘数（新增）

    # 交易所代码映射（合约前缀 -> 交易所代码）
    SYMBOL_EXCHANGE_MAP = {
        "IF": "CFFEX",  # 中金所股指期货
        "IC": "CFFEX",  # 中金所股指期货
        "IH": "CFFEX",  # 中金所股指期货
        "IO": "CFFEX",  # 中金所股指期权
        "MO": "CFFEX",  # 中金所股指期权
        "HO": "CFFEX",  # 中金所股指期权
        "CU": "SHFE",   # 上期所铜期权
        "AU": "SHFE",   # 上期所黄金期权
        "RU": "SHFE",   # 上期所橡胶期权
        "M": "DCE",     # 大商所豆粕期权
        "C": "DCE",     # 大商所玉米期权
        "I": "DCE",     # 大商所铁矿石期权
        "SR": "CZCE",   # 郑商所白糖期权
        "CF": "CZCE",   # 郑商所棉花期权
        "RM": "CZCE",   # 郑商所菜籽粕期权
        "SC": "INE",    # 能源中心原油期权
        "LU": "INE",    # 能源中心低硫燃料油期权
        "PG": "DCE",    # 大商所液化石油气期权
        "MA": "CZCE",   # 郑商所甲醇期权
        "TA": "CZCE",   # 郑商所PTA期权
        "FG": "CZCE",   # 郑商所玻璃期权
        "AP": "CZCE",   # 郑商所苹果期权
        "ZN": "SHFE",   # 上期所锌期权
    }

# ======================
# 完整的交易时间判断解决方案
# ======================
class TradingTimeUtils:
    """交易时间判断工具类（完整解决方案）"""
    
    # 交易所完整交易时间配置（日盘+夜盘）
    EXCHANGE_TRADING_HOURS = {
        "CFFEX": {  # 中国金融期货交易所（中金所）
            "day": {
                "morning_start": time(9, 30),    # 上午开盘时间
                "morning_end": time(11, 30),     # 上午收盘时间
                "afternoon_start": time(13, 0),  # 下午开盘时间
                "afternoon_end": time(15, 0),    # 下午收盘时间
            },
            "night": []  # 中金所股指期权没有夜盘
        },
        "SHFE": {  # 上海期货交易所（上期所）
            "day": {
                "morning_start": time(9, 0),     # 上午第一节开盘时间
                "morning_end": time(10, 15),     # 上午第一节收盘时间
                "morning_start2": time(10, 30),  # 上午第二节开盘时间
                "morning_end2": time(11, 30),    # 上午第二节收盘时间
                "afternoon_start": time(13, 30), # 下午开盘时间
                "afternoon_end": time(15, 0),    # 下午收盘时间
            },
            "night": [
                (time(21, 0), time(1, 0))  # 有色金属、贵金属等夜盘：21:00-次日01:00
            ]
        },
        "DCE": {   # 大连商品交易所（大商所）
            "day": {
                "morning_start": time(9, 0),     # 上午第一节开盘
                "morning_end": time(10, 15),     # 上午第一节收盘
                "morning_start2": time(10, 30),  # 上午第二节开盘
                "morning_end2": time(11, 30),    # 上午第二节收盘
                "afternoon_start": time(13, 30), # 下午开盘
                "afternoon_end": time(15, 0),    # 下午收盘
            },
            "night": [
                (time(21, 0), time(23, 30))  # 农产品、化工、黑色系等夜盘：21:00-23:30
            ]
        },
        "CZCE": {  # 郑州商品交易所（郑商所）
            "day": {
                "morning_start": time(9, 0),     # 上午第一节开盘
                "morning_end": time(10, 15),     # 上午第一节收盘
                "morning_start2": time(10, 30),  # 上午第二节开盘
                "morning_end2": time(11, 30),    # 上午第二节收盘
                "afternoon_start": time(13, 30), # 下午开盘
                "afternoon_end": time(15, 0),    # 下午收盘
            },
            "night": [
                (time(21, 0), time(23, 30))  # 白糖、棉花、菜粕等夜盘：21:00-23:30
            ]
        },
        "INE": {   # 上海国际能源交易中心（上海能源所）
            "day": {
                "morning_start": time(9, 0),     # 上午开盘
                "morning_end": time(11, 30),     # 上午收盘
                "afternoon_start": time(13, 30), # 下午开盘
                "afternoon_end": time(15, 0),    # 下午收盘
            },
            "night": [
                (time(21, 0), time(2, 30))  # 原油、低硫燃料油等夜盘：21:00-次日02:30
            ]
        },
        "GX": {    # 广州期货交易所（广商所）
            "day": {
                "morning_start": time(9, 15),    # 上午开盘
                "morning_end": time(11, 30),     # 上午收盘
                "afternoon_start": time(13, 30),  # 下午开盘
                "afternoon_end": time(15, 30),   # 下午收盘（较晚）
            },
            "night": []  # 目前无夜盘
        }
    }

    @classmethod
    def get_exchange_from_symbol(cls, symbol: str) -> str:
        """从合约代码获取交易所代码"""
        # 遍历交易所映射表
        for prefix, exchange_code in GlobalConfig.SYMBOL_EXCHANGE_MAP.items():
            if symbol.startswith(prefix):
                return exchange_code
        # 默认返回中金所
        logging.warning(f"无法识别合约{symbol}的交易所，默认使用CFFEX")
        return "CFFEX"

    @classmethod
    def is_trading_time(cls, symbol: str) -> bool:
        """
        判断当前时间是否在交易时间内（北京时间），包括日盘和夜盘
        
        参数:
            symbol: 合约代码
        
        返回:
            bool: True表示在交易时间内，False表示非交易时间
        """
        # 获取合约对应的交易所代码
        exchange = cls.get_exchange_from_symbol(symbol)
        
        # 获取交易所配置
        config = cls.EXCHANGE_TRADING_HOURS.get(exchange)
        if not config:
            logging.error(f"未知交易所: {exchange}")
            return False
            
        day_config = config["day"]
        night_sessions = config["night"]
        
        # 转换为北京时间（中国标准时间）
        shanghai_tz = timezone('Asia/Shanghai')
        now_shanghai = datetime.now(shanghai_tz)
        
        # 获取当前日期和前一天的日期（用于处理跨天夜盘）
        today = now_shanghai.date()
        yesterday = today - timedelta(days=1)
        
        # 检查是否为周末（周六和周日）
        if today.weekday() >= 5:  # 周六=5, 周日=6
            # 特别处理周五夜盘延伸到周六凌晨的情况
            if today.weekday() == 5:  # 周六
                # 检查是否有夜盘跨越到周六凌晨
                pass  # 具体判断会在夜盘处理中完成
            else:
                # 周日全天休市
                return False
        
        # 获取节假日信息（使用holidays库）
        cn_holidays = holidays.China()
        
        # 检查是否为法定节假日（当天没有交易）
        if today in cn_holidays:
            # 特别处理节假日前的夜盘延伸到节假日的部分
            pass  # 具体逻辑在夜盘处理中实现
        else:
            # 正常交易日，不进行特殊处理
            pass
        
        # 1. 检查日盘交易时间 ---------------------------------------------------
        if day_config:
            # 准备日盘的所有交易时段
            day_sessions = []
            
            # 上午第一节（所有交易所都有）
            day_sessions.append((
                datetime.combine(today, day_config["morning_start"]),
                datetime.combinate(today, day_config["morning_end"])
            ))
            
            # 上午第二节（部分交易所有）
            if "morning_start2" in day_config and "morning_end2" in day_config:
                day_sessions.append((
                    datetime.combine(today, day_config["morning_start2"]),
                    datetime.combine(today, day_config["morning_end2"])
                ))
            
            # 下午交易时段（所有交易所都有）
            if "afternoon_start" in day_config and "afternoon_end" in day_config:
                day_sessions.append((
                    datetime.combine(today, day_config["afternoon_start"]),
                    datetime.combine(today, day_config["afternoon_end"])
                ))
            
            # 检查是否在任何一个日盘时段内
            for start_dt, end_dt in day_sessions:
                # 添加时区信息
                start_dt = shanghai_tz.localize(start_dt)
                end_dt = shanghai_tz.localize(end_dt)
                
                # 检查当前时间是否在此时间段内
                if start_dt <= now_shanghai <= end_dt:
                    return True
        
        # 2. 检查夜盘交易时间 ---------------------------------------------------
        for session in night_sessions:
            start_time, end_time = session
            
            if end_time > start_time:
                # 不跨天的情况（结束时间晚于开始时间）
                start_dt = shanghai_tz.localize(datetime.combine(today, start_time))
                end_dt = shanghai_tz.localize(datetime.combine(today, end_time))
                
                # 检查当前时间是否在此时段内
                if start_dt <= now_shanghai <= end_dt:
                    return True
                
            else:
                # 跨天的情况（结束时间早于开始时间）
                # 第一部分：今天晚上的时段（从开始时间到午夜）
                start_dt1 = shanghai_tz.localize(datetime.combine(today, start_time))
                end_dt1 = shanghai_tz.localize(datetime.combine(today, time(23, 59, 59)))
                
                # 第二部分：明天凌晨的时段（从午夜到结束时间）
                start_dt2 = shanghai_tz.localize(datetime.combine(today + timedelta(days=1), time(0, 0)))
                end_dt2 = shanghai_tz.localize(datetime.combine(today + timedelta(days=1), end_time))
                
                # 检查当前时间是否在第一部分
                in_first_part = start_dt1 <= now_shanghai <= end_dt1
                # 检查当前时间是否在第二部分
                in_second_part = start_dt2 <= now_shanghai <= end_dt2
                
                if in_first_part or in_second_part:
                    # 检查节假日逻辑
                    if today in cn_holidays or (in_second_part and (today + timedelta(days=1)) in cn_holidays):
                        return False
                    
                    # 检查周末逻辑
                    today_weekday = today.weekday()
                    tomorrow_weekday = (today + timedelta(days=1)).weekday()
                    
                    # 周五夜盘延伸到周六凌晨的情况（周五->周六）
                    if in_second_part and today_weekday == 4 and tomorrow_weekday == 5:
                        return True
                    
                    # 其他情况直接返回
                    return True
                    
        # 所有交易时段都不匹配，返回非交易时间
        return False

# ======================
# 增强DataManager类（修复数据获取问题）
# ======================
class DataManager:
    def __init__(self):
        self.cache = TTLCache(maxsize=100, ttl=60)  # 数据缓存（60秒有效期）
        
    def get_price(self, symbol: str, count: int = 2, frequency: str = "1m") -> pd.DataFrame:
        """获取K线数据（已适配无限易API）"""
        # 检查合约代码有效性
        if not symbol:
            raise ValueError("合约代码不能为空")
            
        try:
            # 创建缓存键
            cache_key = f"{symbol}_{frequency}_{count}"
            # 检查缓存中是否已有数据
            if cache_key not in self.cache:
                # 模拟从无限易API获取数据
                logging.info(f"从无限易API请求数据: {symbol}")
                data = self._mock_infinite_api(symbol, frequency, count)
                
                # 验证数据完整性
                if data.empty or len(data) < count:
                    raise ValueError(f"获取{symbol}数据失败，返回{len(data)}条记录")
                
                # 存入缓存
                self.cache[cache_key] = data
            # 返回缓存数据
            return self.cache[cache_key]
        except Exception as e:
            # 处理异常情况
            logging.error(f"数据获取异常：{str(e)}")
            return pd.DataFrame()
    
    def _mock_infinite_api(self, symbol: str, frequency: str, count: int) -> pd.DataFrame:
        """模拟无限易API返回的K线数据"""
        # 在实际应用中应替换为真实的API调用
        
        # 生成时间序列
        now = datetime.now()
        if frequency == "1d":
            # 日线数据
            dates = [date.today() - timedelta(days=i) for i in range(count)]
            data = {
                'date': dates,
                'open': np.random.uniform(3000, 3200, count),
                'high': np.random.uniform(3200, 3300, count),
                'low': np.random.uniform(3000, 3100, count),
                'close': np.random.uniform(3050, 3250, count),
                'volume': np.random.randint(100, 1000, count)
            }
        else:
            # 分钟线数据
            minutes = int(frequency[:-1])
            dates = [now - timedelta(minutes=i) for i in range(count * minutes, 0, -minutes)]
            data = {
                'date': dates,
                'open': np.random.uniform(3000, 3200, count),
                'high': np.random.uniform(3200, 3300, count),
                'low': np.random.uniform(3000, 3100, count),
                'close': np.random.uniform(3050, 3250, count),
                'volume': np.random.randint(100, 1000, count)
            }
        # 创建DataFrame并设置日期索引
        return pd.DataFrame(data).set_index('date')

    def get_option_chain(self, underlying: str) -> List[OptionContract]:
        """获取指定标的的期权链（无限易适配）"""
        # 记录日志
        logging.info(f"从无限易API获取期权链: {underlying}")
        
        # 获取标的当前价格
        current_price = self.get_price(underlying, count=1).close.iloc[-1]
        
        # 创建空合约列表
        contracts = []
        # 生成行权价序列（基于当前价格）
        strikes = np.arange(round(current_price - 300, -1), round(current_price + 300, -1), 50)
        
        # 设置到期日（模拟当月和下月合约）
        current_month = datetime.now().replace(day=1) + timedelta(days=32)
        current_month = current_month.replace(day=15)  # 当月合约到期日设为下月15日
        next_month = (current_month + timedelta(days=32)).replace(day=15)
        
        # 为每个行权价生成看涨和看跌期权
        for strike in strikes:
            # 看涨期权
            call_price = max(current_price - strike, 0) + strike * 0.1
            # 当月合约
            contracts.append(OptionContract(
                symbol=f"{underlying}C{strike}01",  # 合约代码 (01表示当月)
                contract_type="CALL",  # 合约类型
                strike=strike,  # 行权价
                volume=np.random.randint(100, 500),  # 随机成交量
                delta=np.random.uniform(0.2, 0.8),  # 随机Delta值
                gamma=np.random.uniform(0, 0.1),  # 随机Gamma值
                expired=False,  # 未到期
                last_price=call_price,  # 最后成交价
                expiration=current_month  # 到期日
            ))
            # 看跌期权
            put_price = max(strike - current_price, 0) + strike * 0.1
            contracts.append(OptionContract(
                symbol=f"{underlying}P{strike}01",  # 合约代码 (01表示当月)
                contract_type="PUT",  # 合约类型
                strike=strike,  # 行权价
                volume=np.random.randint(100, 500),  # 随机成交量
                delta=np.random.uniform(-0.8, -0.2),  # 随机Delta值
                gamma=np.random.uniform(0, 0.1),  # 随机Gamma值
                expired=False,  # 未到期
                last_price=put_price,  # 最后成交价
                expiration=current_month  # 到期日
            ))
            
            # 下月合约（成交量通常会小一些）
            contracts.append(OptionContract(
                symbol=f"{underlying}C{strike}02",  # 合约代码 (02表示下月)
                contract_type="CALL",  # 合约类型
                strike=strike,  # 行权价
                volume=np.random.randint(50, 300),  # 随机成交量（通常更小）
                delta=np.random.uniform(0.2, 0.8),  # 随机Delta值
                gamma=np.random.uniform(0, 0.1),  # 随机Gamma值
                expired=False,  # 未到期
                last_price=call_price,  # 最后成交价
                expiration=next_month  # 到期日
            ))
            # 看跌期权下月合约
            contracts.append(OptionContract(
                symbol=f"{underlying}P{strike}02",  # 合约代码 (02表示下月)
                contract_type="PUT",  # 合约类型
                strike=strike,  # 行权价
                volume=np.random.randint(50, 300),  # 随机成交量（通常更小）
                delta=np.random.uniform(-0.8, -0.2),  # 随机Delta值
                gamma=np.random.uniform(0, 0.1),  # 随机Gamma值
                expired=False,  # 未到期
                last_price=put_price,  # 最后成交价
                expiration=next_month  # 到期日
            ))
        # 返回合约列表
        return contracts

# ======================
# 增强SignalGenerator（修复逻辑漏洞）
# ======================
class SignalGenerator:
    def __init__(self, data_manager: DataManager):
        # 初始化数据管理器
        self.data_manager = data_manager
    
    def check_trend(self, symbol: str) -> bool:
        """检查当前趋势方向"""
        # 获取5分钟K线数据
        data = self.data_manager.get_price(symbol, count=5, frequency="5m")
        
        # 验证数据有效性
        if data.empty or len(data) < 3:
            return False
            
        # 计算移动平均线
        data['ma3'] = data.close.rolling(3).mean()
        data['ma5'] = data.close.rolling(5).mean()
        
        # 判断趋势方向（金叉为上涨趋势）
        if data['ma3'].iloc[-1] > data['ma5'].iloc[-1] and data['ma3'].iloc[-2] <= data['ma5'].iloc[-2]:
            return True
        return False

    def detect_volume_spike(self, symbol: str) -> bool:
        """检测成交量异常波动（修复逻辑）"""
        # 获取1分钟K线成交量数据
        vol_data = self.data_manager.get_price(symbol, count=10, frequency="1m")['volume']
        
        # 验证数据有效性
        if vol_data.empty or len(vol_data) < 6:
            return False
            
        # 计算5期平均成交量
        avg_vol = vol_data.iloc[-6:-1].mean()
        # 获取当前成交量
        current_vol = vol_data.iloc[-1]
        
        # 检查成交量是否显著放大
        return current_vol > avg_vol * GlobalConfig.VOLUME_MULTIPLIERS[0]

# ======================
# 增强TradeExecutor（修复订单问题）
# ======================
class TradeExecutor:
    def __init__(self, data_manager: DataManager):
        # 初始化数据管理器
        self.data_manager = data_manager
        # API连接状态
        self.api_connected = False
        # 连接API
        self.connect_to_infinite_api()
    
    def connect_to_infinite_api(self) -> bool:
        """连接无限易交易API"""
        try:
            # 实际环境中应调用无限易的登录接口
            logging.info("成功连接无限易API")
            self.api_connected = True
            return True
        except Exception as e:
            # 处理连接失败
            logging.error(f"连接无限易API失败: {str(e)}")
            return False
    
    def send_order(self, 
                   symbol: str,  # 合约代码
                   direction: str,  # 买卖方向
                   lots: int,  # 交易手数
                   price_type: str = 'LIMIT',  # 价格类型
                   price: float = None) -> bool:  # 指定价格
        """发送交易订单（无限易适配）"""
        # 检查API连接状态
        if not self.api_connected:
            logging.error("API未连接，无法发送订单")
            return False
                
        # 验证交易数量有效性
        if lots <= 0:
            logging.error("无效的交易数量")
            return False
            
        try:
            # 获取最优报价（如未提供）
            if price is None:
                price = self._get_best_price(symbol, direction)
            
            # 考虑滑点影响
            slippage = GlobalConfig.SLIPPAGE * price
            # 卖出价减去滑点，买入价加上滑点
            if direction == 'SELL':
                price -= slippage
            else:
                price += slippage
                
            # 记录订单信息
            logging.info(f"发送订单: {direction} {lots}手 {symbol} @ {price:.2f}")
            
            # 生成唯一订单ID（模拟）
            order_id = hashlib.md5(f"{datetime.now()}{symbol}".encode()).hexdigest()
            
            # 模拟订单执行（95%成功率）
            if np.random.rand() > 0.05:
                logging.info(f"订单 {order_id} 成交成功")
                return True
            else:
                # 处理失败情况
                logging.warning(f"订单 {order_id} 被拒绝")
                return False
                
        except Exception as e:
            # 处理订单异常
            logging.error(f"订单处理异常: {str(e)}")
            return False

    def _get_best_price(self, symbol: str, direction: str) -> float:
        """获取最优报价（无限易适配）"""
        try:
            # 提取标的代码
            underlying = symbol[:2]  # 简化处理（仅适用于两位代码）
            # 获取标的价格
            market_data = self.data_manager.get_price(underlying, count=1)
            last_price = market_data.close.iloc[-1]  # 最新收盘价
            
            # 模拟报价策略
            if direction == 'BUY':
                return last_price * 1.001  # 买入价略高于市价
            else:
                return last_price * 0.999  # 卖出价略低于市价
        except:
            # 获取失败时使用最后成交价
            return self.data_manager.get_price(symbol, count=1).close.iloc[-1]

# ======================
# 增强风险管理（修复波动率计算）
# ======================
class RiskManager:
    def __init__(self, data_manager: DataManager):
        # 初始化数据管理器
        self.data_manager = data_manager
    
    def get_historical_volatility(self, symbol: str, period: int = 20) -> float:
        """计算历史波动率（改进算法）"""
        try:
            # 获取指定周期的历史数据
            data = self.data_manager.get_price(symbol, count=period+1, frequency="1d")
            
            # 检查数据充足性
            if len(data) < 5:
                logging.warning(f"数据不足，无法计算{symbol}的波动率")
                return 0.2
                
            # 计算对数收益率
            returns = np.log(data.close / data.close.shift(1))
            # 计算年化波动率
            volatility = returns.std() * np.sqrt(252)
            
            # 记录波动率
            logging.info(f"{symbol}历史波动率: {volatility:.2%}")
            return volatility
        except Exception as e:
            # 处理计算异常
            logging.error(f"波动率计算失败：{str(e)}")
            return 0.2
    
    def position_sizing(self, 
                        account: AccountInfo,  # 账户信息
                        entry_price: float,  # 入场价格
                        stop_loss: float) -> int:  # 止损价格
        """基于风险的仓位管理"""
        # 计算每手风险值
        risk_per_lot = abs(entry_price - stop_loss) * GlobalConfig.OPTION_CONTRACT_SIZE
        # 避免除零错误
        if risk_per_lot == 0:
            return 0
            
        # 计算单笔交易最大可承担风险
        max_risk = account.cash * GlobalConfig.MAX_RISK_PER_TRADE
        
        # 计算最大可交易手数
        max_lots = int(max_risk / risk_per_lot)
        
        # 考虑最大持仓限制
        return min(max_lots, GlobalConfig.MAX_POSITION)

# ======================
# 策略核心类（增加到期日处理和止损逻辑）
# ======================
class OptionTradingStrategy(BaseStrategy):
    def __init__(self):
        # 初始化各个模块
        super().__init__()
        self.params_map = Params()

        self.data_manager = DataManager()
        self.signal_generator = SignalGenerator(self.data_manager)
        self.trade_executor = TradeExecutor(self.data_manager)
        self.risk_manager = RiskManager(self.data_manager)
        
        # 初始化账户信息（初始资金100万）
        self.account = AccountInfo(
            cash=1000000,
            margin_ratio=GlobalConfig.MARGIN_RATE_BASE,
            positions={}  # 初始无持仓
        )
        
        # 记录标的趋势状态
        self.symbol_trend = {}
        
        # 记录主合约月份
        self.main_contracts = {}
        
        # 初始化交易对象
        self.selected_contract = None
        self.selected_direction = "BUY"
        
        # 记录持仓合约详细信息（新增）
        self.position_contracts = {}
    
    def run(self):
        """主策略逻辑"""
        try:
            # 启动日志
            logging.info("===== 策略开始运行 =====")
            
            # 主循环
            while True:
                # 1. 检查交易时间
                if not self._check_trading_time():
                    # 非交易时间等待60秒
                    sys_time.sleep(60)
                    continue
                    
                # 2. 检查期权到期日并自动处理（修改）
                self._handle_near_expiry()
                
                # 3. 检查止损并自动处理（新增）
                self._check_stop_loss()
                
                # 4. 策略核心逻辑 - 处理每个标的
                symbol_scores = self._calculate_symbol_scores()
                
                if symbol_scores:
                    # 选择排名第一的标的
                    top_symbol, score = symbol_scores[0]
                    logging.info(f"选择交易标的：{top_symbol}，得分：{score}")
                    
                    # 处理该标的的期权交易
                    self._process_underlying(top_symbol)
                
                # 5. 轮询间隔15秒
                sys_time.sleep(15)
                
        except KeyboardInterrupt:
            # 手动终止策略
            logging.info("策略被手动终止")
        except Exception as e:
            # 全局异常处理
            error_msg = f"策略崩溃：{str(e)}\n{traceback.format_exc()}"
            logging.critical(error_msg)
            self._send_alert(error_msg)
    
    def _calculate_symbol_scores(self) -> List[Tuple[str, int]]:
        """计算各标的得分并排序"""
        symbol_scores = []
        
        for symbol in GlobalConfig.SYMBOLS:
            try:
                # 获取期权链
                options = self.data_manager.get_option_chain(symbol)
                
                # 过滤可用期权（未到期且delta在合理范围）
                valid_options = [opt for opt in options if not opt.expired and abs(opt.delta) < GlobalConfig.MAX_DELTA]
                
                # 获取标的当前价
                underlying_data = self.data_manager.get_price(symbol, count=1)
                if underlying_data.empty:
                    continue
                current_price = underlying_data.close.iloc[-1]
                
                # 获取前日收市价（使用1天K线）
                prev_day_data = self.data_manager.get_price(symbol, count=2, frequency="1d")
                if prev_day_data.empty or len(prev_day_data) < 2:
                    continue
                prev_close = prev_day_data.close.iloc[-2]
                
                # 检查趋势方向并缓存
                is_uptrend = self.signal_generator.check_trend(symbol)
                self.symbol_trend[symbol] = is_uptrend
                
                # 根据趋势确定主合约月份（当月和下月）
                current_month = min(opt.expiration for opt in valid_options)
                next_month = current_month + timedelta(days=31)
                
                # 识别主合约月份
                current_month_options = [opt for opt in valid_options 
                                        if (opt.expiration - current_month).days <= 15]
                next_month_options = [opt for opt in valid_options 
                                      if (opt.expiration - next_month).days <= 15]
                
                # 统计当月虚值期权数量（上涨趋势看涨期权，下跌趋势看跌期权）
                current_month_count = 0
                next_month_count = 0
                
                # 根据趋势确定虚值期权类型
                if is_uptrend:  # 上涨趋势
                    # 筛选当月和下月的虚值看涨期权
                    current_month_count = len([opt for opt in current_month_options 
                                              if opt.contract_type == "CALL" and opt.strike > prev_close])
                    next_month_count = len([opt for opt in next_month_options 
                                           if opt.contract_type == "CALL" and opt.strike > prev_close])
                else:  # 下跌趋势
                    # 筛选当月和下月的虚值看跌期权
                    current_month_count = len([opt for opt in current_month_options 
                                              if opt.contract_type == "PUT" and opt.strike < prev_close])
                    next_month_count = len([opt for opt in next_month_options 
                                           if opt.contract_type == "PUT" and opt.strike < prev_close])
                
                # 计算总得分（当月+下月虚值期权数量）
                total_count = current_month_count + next_month_count
                symbol_scores.append((symbol, total_count))
                
                logging.debug(f"{symbol}趋势: {'上涨' if is_uptrend else '下跌'}, "
                             f"当月虚值: {current_month_count}, 次月虚值: {next_month_count}, 总分: {total_count}")
            
            except Exception as e:
                logging.error(f"计算{symbol}得分时出错: {str(e)}")
                continue
        
        # 按得分降序排序
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        return symbol_scores
    
    def _process_underlying(self, symbol: str):
        """处理单个标的的期权交易"""
        # 获取期权链
        options = self.data_manager.get_option_chain(symbol)
        
        # 过滤可用期权（未到期且delta在合理范围）
        valid_options = [opt for opt in options if not opt.expired and abs(opt.delta) < GlobalConfig.MAX_DELTA]
        
        # 获取标的当前价
        underlying_data = self.data_manager.get_price(symbol, count=1)
        if underlying_data.empty:
            return
        current_price = underlying_data.close.iloc[-1]
        
        # 获取前日收市价
        prev_day_data = self.data_manager.get_price(symbol, count=2, frequency="1d")
        if prev_day_data.empty or len(prev_day_data) < 2:
            return
        prev_close = prev_day_data.close.iloc[-2]
        
        # 检查趋势方向（使用缓存结果）
        is_uptrend = self.symbol_trend.get(symbol, True)
        
        # 确定主合约月份（当月到期）
        main_expiration = min(opt.expiration for opt in valid_options)
        main_options = [opt for opt in valid_options 
                        if (opt.expiration - main_expiration).days <= 15]
        
        # 选择成交量最大的当月期权合约
        if is_uptrend:  # 上涨趋势，选择虚值看涨期权
            # 虚值看涨期权（行权价大于前日收市价）
            call_options = [opt for opt in main_options 
                            if opt.contract_type == "CALL" and opt.strike > prev_close]
            if call_options:
                # 选择成交量最大的合约
                target_option = max(call_options, key=lambda opt: opt.volume)
                self._trade_option(target_option, "BUY")
        else:  # 下跌趋势，选择虚值看跌期权
            # 虚值看跌期权（行权价小于前日收市价）
            put_options = [opt for opt in main_options 
                           if opt.contract_type == "PUT" and opt.strike < prev_close]
            if put_options:
                # 选择成交量最大的合约
                target_option = max(put_options, key=lambda opt: opt.volume)
                self._trade_option(target_option, "BUY")
    
    def _trade_option(self, option: OptionContract, direction: str):
        """交易特定的期权合约"""
        # 计算止损价（50%的止损比例）
        stop_loss = option.last_price * (1 - GlobalConfig.STOP_LOSS_RATIO)
        
        # 计算仓位大小（基于风险管理）
        position_size = self.risk_manager.position_sizing(
            self.account,  # 账户信息
            option.last_price,  # 入场价格
            stop_loss  # 止损价格
        )
        
        # 验证仓位大小
        if position_size > 0:
            # 记录选择的合约
            self.selected_contract = option
            self.selected_direction = direction
            
            # 发送订单
            if self.trade_executor.send_order(
                symbol=option.symbol,
                direction=direction,
                lots=position_size
            ):
                # 更新账户信息
                # 计算交易成本（考虑期权合约乘数）
                cost = position_size * option.last_price * GlobalConfig.OPTION_CONTRACT_SIZE
                # 扣除现金
                self.account.cash -= cost
                # 更新持仓
                if option.symbol in self.account.positions:
                    # 已有持仓增加
                    self.account.positions[option.symbol] += position_size
                else:
                    # 新增持仓记录
                    self.account.positions[option.symbol] = position_size
                
                # 记录持仓合约详细信息（新增）
                self.position_contracts[option.symbol] = option
                
                logging.info(f"交易执行：{direction} {position_size}手 {option.symbol} @ {option.last_price:.2f}，止损价：{stop_loss:.2f}")
    
    def _check_trading_time(self) -> bool:
        """检查当前是否在交易时间内"""
        # 遍历所有交易标的，任一标的可交易即返回True
        for symbol in GlobalConfig.SYMBOLS:
            if TradingTimeUtils.is_trading_time(symbol):
                return True
        return False

    def _handle_near_expiry(self):
        """自动处理即将到期的期权持仓（新增）"""
        to_close = []
        today = datetime.now()
        
        # 检查所有持仓合约
        for symbol, contract in list(self.position_contracts.items()):
            # 计算剩余天数
            days_to_expiry = (contract.expiration - today).days
            # 如果剩余天数小于等于警告阈值
            if 0 <= days_to_expiry <= GlobalConfig.EXPIRE_DAYS_WARNING:
                to_close.append(symbol)
        
        # 平仓处理
        for symbol in to_close:
            if symbol in self.account.positions and self.account.positions[symbol] > 0:
                # 获取持仓数量
                position_size = self.account.positions[symbol]
                # 平仓操作
                if self.trade_executor.send_order(
                    symbol=symbol,
                    direction="SELL",
                    lots=position_size
                ):
                    # 更新账户
                    self.account.cash += position_size * self.position_contracts[symbol].last_price * GlobalConfig.OPTION_CONTRACT_SIZE
                    # 清除持仓记录
                    self.account.positions[symbol] = 0
                    del self.position_contracts[symbol]
                    
                    alert_msg = f"到期平仓: {symbol} {position_size}手"
                    logging.warning(alert_msg)
                    self._send_alert(alert_msg)
    
    def _check_stop_loss(self):
        """检查止损并自动处理（新增）"""
        to_close = []
        
        # 检查所有持仓合约
        for symbol, contract in list(self.position_contracts.items()):
            # 获取当前价格
            current_price_data = self.data_manager.get_price(symbol, count=1)
            if not current_price_data.empty:
                current_price = current_price_data.close.iloc[-1]
                # 计算止损触发价
                stop_loss_price = contract.last_price * (1 - GlobalConfig.STOP_LOSS_RATIO)
                
                # 如果当前价格低于止损价
                if current_price <= stop_loss_price:
                    to_close.append(symbol)
        
        # 平仓处理
        for symbol in to_close:
            if symbol in self.account.positions and self.account.positions[symbol] > 0:
                # 获取持仓数量
                position_size = self.account.positions[symbol]
                # 平仓操作
                if self.trade_executor.send_order(
                    symbol=symbol,
                    direction="SELL",
                    lots=position_size
                ):
                    # 更新账户（简化处理）
                    self.account.cash += position_size * self.position_contracts[symbol].last_price * 0.6 * GlobalConfig.OPTION_CONTRACT_SIZE
                    # 清除持仓记录
                    self.account.positions[symbol] = 0
                    del self.position_contracts[symbol]
                    
                    alert_msg = f"止损触发: {symbol} {position_size}手"
                    logging.warning(alert_msg)
                    self._send_alert(alert_msg)

    def _send_alert(self, message: str):
        """发送警报（无限易适配）"""
        # 记录警报
        logging.warning(f"[警报] {message}")
        # 实际环境中应调用无限易的警报接口

# ======================
# 主执行逻辑增强
# ======================
if __name__ == "__main__":
    try:
        # 初始化策略
        strategy = OptionTradingStrategy()
        # 启动策略
        strategy.run()
    except Exception as e:
        # 主程序异常处理
        error_msg = f"主程序崩溃：{str(e)}\n{traceback.format_exc()}"
        logging.critical(error_msg)

