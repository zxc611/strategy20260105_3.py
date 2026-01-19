"""
期权宽度交易信号生成器 - Strategy20260105_3
说明：顶部内容因意外粘贴导致语法错误，已恢复为标准导入区。
"""
from datetime import datetime, timedelta, time as dtime
import os
import json
import traceback
import types
import re
import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple

from dataclasses import dataclass

# 兼容性导入：优先顶层导入，失败后回退到已确认存在的定义
try:
    from pythongo import infini
except Exception:
    infini = None  # 调试环境兜底

# 实盘日盘时段定义
# 中金所（股指/国债）
CFFEX_DAY_SESSIONS = [
    (dtime(9, 30, 0), dtime(11, 30, 0)),
    (dtime(13, 0, 0), dtime(15, 0, 0)),
]
# 商品交易所（上期所/大商所/郑商所/广期所）
COMMODITY_DAY_SESSIONS = [
    (dtime(9, 0, 0), dtime(10, 15, 0)),
    (dtime(10, 30, 0), dtime(11, 30, 0)),
    (dtime(13, 30, 0), dtime(15, 0, 0)),
]

# 夜盘（大商所/郑商所/广期所）
NIGHT_START_DCE_CZCE_GFEX = dtime(21, 0, 0)
NIGHT_END_DCE_CZCE_GFEX = dtime(23, 0, 0)

# 夜盘（上期所，跨午夜至次日2:00）
NIGHT_START_SHFE = dtime(21, 0, 0)
NIGHT_END_SHFE = dtime(2, 0, 0)

try:
    from pythongo.base import BaseStrategy  # type: ignore
except Exception:
    class BaseStrategy:
        """最小兜底，避免导入失败导致模块无法加载"""
        pass

# 尝试导入 KLineData（行情/合约数据结构）
try:
    from pythongo import KLineData  # type: ignore
except Exception:
    try:
        from pythongo.utils import KLineData  # type: ignore
    except Exception:
        try:
            from pythongo.classdef import KLineData  # type: ignore
        except Exception:
            class KLineData:
                pass

# 兼容 TickData 与 KLineGenerator（用于 Tick 合成 K 线）
try:
    from pythongo import TickData, KLineGenerator  # type: ignore
except Exception:
    try:
        from pythongo.utils import TickData, KLineGenerator  # type: ignore
    except Exception:
        try:
            from pythongo.classdef import TickData, KLineGenerator  # type: ignore
        except Exception:
            class TickData:
                pass

            class KLineGenerator:
                def __init__(self, callback=None, exchange: str = "", instrument_id: str = "", style: str = "M1", *args, **kwargs) -> None:
                    self.callback = callback
                    self.exchange = exchange
                    self.instrument_id = instrument_id
                    self.style = style

                def tick_to_kline(self, tick: Any) -> None:
                    price = getattr(tick, "last_price", None) or getattr(tick, "last", None) or getattr(tick, "price", None)
                    if price is None:
                        return
                    bar = types.SimpleNamespace(
                        exchange=self.exchange or getattr(tick, "exchange", ""),
                        instrument_id=self.instrument_id or getattr(tick, "instrument_id", ""),
                        style=self.style,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=getattr(tick, "volume", getattr(tick, "Volume", 0))
                    )
                    if callable(self.callback):
                        try:
                            self.callback(bar)
                        except Exception:
                            pass

try:
    from pythongo.core import MarketCenter  # type: ignore
    from pythongo.utils import Scheduler  # type: ignore
except Exception:
    MarketCenter = None  # type: ignore
    Scheduler = None  # type: ignore

# 修补底层 MarketCenter.get_instrument_data 参数不兼容问题（若可用）
if MarketCenter and hasattr(MarketCenter, "get_instrument_data"):
    try:
        def _patched_get_instrument_data(self, exchange=None, instrument_id=None, *args, **kwargs):
            try:
                return self.get_instrument_data_original(exchange, instrument_id)
            except Exception:
                return None
        MarketCenter.get_instrument_data_original = MarketCenter.get_instrument_data  # type: ignore
        MarketCenter.get_instrument_data = _patched_get_instrument_data  # type: ignore
    except Exception:
        pass


"""参数模型定义，兼容 BaseParams 不可用时的降级"""
try:
    from pythongo.base import BaseParams, Field  # type: ignore

    class Params(BaseParams):
        max_kline: int = Field(int, default=200, title="K线缓存长度")
        kline_style: str = Field(str, default="M1", title="K线周期")
        subscribe_options: bool = Field(bool, default=True, title="是否订阅期权行情")
        debug_output: bool = Field(bool, default=True, title="是否输出调试信息")
        diagnostic_output: bool = Field(bool, default=True, title="诊断/测试输出开关（交易/回测自动关闭）")
        api_key: str = Field(str, default="", title="通用API密钥（可选，映射到环境变量API_KEY）")
        infini_api_key: str = Field(str, default="", title="Infini行情密钥（可选，映射到环境变量INFINI_API_KEY）")
        access_key: str = Field(str, default="", title="访问密钥 AccessKey（平台提供）")
        access_secret: str = Field(str, default="", title="访问密钥 AccessSecret（平台提供）")
        run_profile: str = Field(str, default="full", title="运行预设(full|lite)")
        enable_scheduler: bool = Field(bool, default=True, title="是否启用定时任务（回测可关闭）")
        use_tick_kline_generator: bool = Field(bool, default=True, title="是否启用Tick合成K线")
        backtest_tick_mode: bool = Field(bool, default=False, title="回测模式：仅用Tick驱动K线（跳过历史K线拉取）")
        exchange: str = Field(str, default="CFFEX", title="默认交易所，用于查询合约")
        future_product: str = Field(str, default="IF", title="期货品种，用于查询合约")
        option_product: str = Field(str, default="IO", title="期权品种，用于查询合约")
        auto_load_history: bool = Field(bool, default=True, title="启动后自动加载历史K线")
        load_history_options: bool = Field(bool, default=True, title="加载历史K线时是否包含期权")
        load_all_products: bool = Field(bool, default=True, title="是否加载全部品种（忽略产品过滤）")
        exchanges: str = Field(str, default="CFFEX,SHFE,DCE,CZCE", title="交易所列表（逗号分隔）")
        future_products: str = Field(str, default="IF,IH,IC,CU,AL,ZN,RB,AU,AG,M,Y,A,J,JM,I,CF,SR,MA,TA", title="期货品种列表（逗号分隔）")
        option_products: str = Field(str, default="IO,HO,MO", title="期权品种列表（逗号分隔）")
        include_future_products_for_options: bool = Field(bool, default=True, title="将期货品种一并尝试作为期权品种加载（覆盖商品期权）")
        subscription_batch_size: int = Field(int, default=10, title="订阅批次大小")
        subscription_interval: int = Field(int, default=1, title="订阅批次间隔(秒)")
        subscription_backoff_factor: float = Field(float, default=1.0, title="订阅批次退避因子")
        subscribe_only_current_next_options: bool = Field(bool, default=True, title="仅订阅指定月/指定下月期权（旧字段名）")
        subscribe_only_current_next_futures: bool = Field(bool, default=True, title="仅订阅指定月/指定下月期货（旧字段名，仅限CFFEX IF/IH/IC）")
        enable_doc_examples: bool = Field(bool, default=False, title="启用说明文档示例")
        pause_unsubscribe_all: bool = Field(bool, default=True, title="暂停时退订所有行情")
        pause_force_stop_scheduler: bool = Field(bool, default=True, title="暂停时强制停止调度器（resume时重启）")
        pause_on_stop: bool = Field(bool, default=False, title="平台 on_stop 回调是否按暂停处理")
        history_minutes: int = Field(int, default=1440, title="历史K线拉取回看分钟数")
        log_file_path: str = Field(str, default="strategy_startup.log", title="本地日志文件路径")
        test_mode: bool = Field(bool, default=False, title="测试模式：忽略开盘时间门控")
        auto_start_after_init: bool = Field(bool, default=False, title="初始化后自动触发 on_start，避免卡在初始化状态")
        # 指定月/指定下月新增参数，兼容旧字段
        subscribe_only_specified_month_options: bool = Field(bool, default=True, title="仅订阅指定月/指定下月期权")
        subscribe_only_specified_month_futures: bool = Field(bool, default=True, title="仅订阅指定月/指定下月期货")
        specified_month: str = Field(str, default="", title="指定月合约代码")
        next_specified_month: str = Field(str, default="", title="指定下月合约代码")
        month_mapping: Dict[str, Any] = Field(dict, default_factory=dict, title="品种指定月/指定下月映射")
        # 开仓/风控参数
        option_buy_lots_min: int = Field(int, default=1, title="期权买入开仓最小手数")
        option_buy_lots_max: int = Field(int, default=100, title="期权买入开仓最大手数")
        option_contract_multiplier: float = Field(float, default=10000, title="期权合约乘数（价格*乘数*手数）")
        position_limit_valid_hours_max: int = Field(int, default=720, title="开仓资金限额可设置的最大有效小时数")
        position_limit_default_valid_hours: int = Field(int, default=24, title="开仓资金限额默认有效小时数")
        position_limit_max_ratio: float = Field(float, default=0.2, title="开仓资金限额占总资金比例上限")
        position_limit_min_amount: float = Field(float, default=1000, title="开仓资金限额最小金额")
        option_order_price_type: str = Field(str, default="2", title="期权开仓委托价类型")
        option_order_time_condition: str = Field(str, default="3", title="期权开仓时间条件")
        option_order_volume_condition: str = Field(str, default="1", title="期权开仓成交量条件")
        option_order_contingent_condition: str = Field(str, default="1", title="期权开仓触发条件")
        option_order_force_close_reason: str = Field(str, default="0", title="期权开仓强平原因")
        option_order_hedge_flag: str = Field(str, default="1", title="期权开仓投机/套保标记")
        option_order_min_volume: int = Field(int, default=1, title="期权开仓最小成交量")
        option_order_business_unit: str = Field(str, default="1", title="期权开仓业务单元")
        option_order_is_auto_suspend: int = Field(int, default=0, title="期权开仓是否自动挂起")
        option_order_user_force_close: int = Field(int, default=0, title="期权开仓是否用户强平")
        option_order_is_swap: int = Field(int, default=0, title="期权开仓是否互换单")
        # 平仓参数
        close_take_profit_ratio: float = Field(float, default=1.5, title="止盈倍数（开仓价*倍数）")
        close_overnight_check_time: str = Field(str, default="14:58", title="隔夜仓检查时间(HH:MM)")
        close_daycut_time: str = Field(str, default="15:58", title="日内平仓时间(HH:MM)")
        close_max_hold_days: int = Field(int, default=3, title="最大持仓天数(>=则平仓)")
        close_overnight_loss_threshold: float = Field(float, default=-0.5, title="隔夜亏损平仓阈值(收益率)")
        close_overnight_profit_threshold: float = Field(float, default=4.0, title="隔夜盈利平仓阈值(收益率)")
        close_max_chase_attempts: int = Field(int, default=5, title="追单最大次数")
        close_chase_interval_seconds: int = Field(int, default=2, title="追单间隔秒数")
        close_chase_task_timeout_seconds: int = Field(int, default=30, title="追单任务超时秒数")
        close_delayed_timeout_seconds: int = Field(int, default=30, title="延迟平仓超时秒数")
        close_delayed_max_retries: int = Field(int, default=3, title="延迟平仓最大重试次数")
        close_order_price_type: str = Field(str, default="2", title="平仓委托价类型")
except Exception:
    @dataclass
    class Params:
        max_kline: int = 200
        kline_style: str = "M1"
        subscribe_options: bool = True
        debug_output: bool = True
        diagnostic_output: bool = True
        api_key: str = ""
        infini_api_key: str = ""
        access_key: str = ""
        access_secret: str = ""
        run_profile: str = "lite"
        enable_scheduler: bool = True
        use_tick_kline_generator: bool = True
        backtest_tick_mode: bool = False
        exchange: str = "CFFEX"
        future_product: str = "IF"
        option_product: str = "IO"
        auto_load_history: bool = True
        load_history_options: bool = True
        load_all_products: bool = True
        exchanges: str = "CFFEX,SHFE,DCE,CZCE"
        future_products: str = "IF,IH,IC,CU,AL,ZN,RB,AU,AG,M,Y,A,J,JM,I,CF,SR,MA,TA"
        option_products: str = "IO,HO,MO"
        include_future_products_for_options: bool = True
        subscription_batch_size: int = 10
        subscription_interval: int = 1
        subscription_backoff_factor: float = 1.0
        subscribe_only_current_next_options: bool = True
        subscribe_only_current_next_futures: bool = True
        enable_doc_examples: bool = False
        pause_unsubscribe_all: bool = True
        pause_force_stop_scheduler: bool = True
        pause_on_stop: bool = True
        history_minutes: int = 1440
        log_file_path: str = "strategy_startup.log"
        test_mode: bool = False
        auto_start_after_init: bool = True
        # 指定月/指定下月新增参数，兼容旧字段
        subscribe_only_specified_month_options: bool = True
        subscribe_only_specified_month_futures: bool = True
        specified_month: str = ""
        next_specified_month: str = ""
        month_mapping: Dict[str, Any] = None
        # 开仓/风控参数
        option_buy_lots_min: int = 1
        option_buy_lots_max: int = 100
        option_contract_multiplier: float = 10000
        position_limit_valid_hours_max: int = 720
        position_limit_default_valid_hours: int = 24
        position_limit_max_ratio: float = 0.2
        position_limit_min_amount: float = 1000
        option_order_price_type: str = "2"
        option_order_time_condition: str = "3"
        option_order_volume_condition: str = "1"
        option_order_contingent_condition: str = "1"
        option_order_force_close_reason: str = "0"
        option_order_hedge_flag: str = "1"
        option_order_min_volume: int = 1
        option_order_business_unit: str = "1"
        option_order_is_auto_suspend: int = 0
        option_order_user_force_close: int = 0
        option_order_is_swap: int = 0
        # 平仓参数
        close_take_profit_ratio: float = 1.5
        close_overnight_check_time: str = "14:58"
        close_daycut_time: str = "15:58"
        close_max_hold_days: int = 3
        close_overnight_loss_threshold: float = -0.5
        close_overnight_profit_threshold: float = 4.0
        close_max_chase_attempts: int = 5
        close_chase_interval_seconds: int = 2
        close_chase_task_timeout_seconds: int = 30
        close_delayed_timeout_seconds: int = 30
        close_delayed_max_retries: int = 3
        close_order_price_type: str = "2"


class ApiKeyLoader:
    """API密钥加载器"""

    @staticmethod
    def load_api_key(params: Params) -> Dict[str, str]:
        """加载API Key到环境变量"""
        loaded_keys: Dict[str, str] = {}

        # 1) 参数优先
        key_infini = (getattr(params, "infini_api_key", "") or "").strip()
        key_generic = (getattr(params, "api_key", "") or "").strip()
        key_access = (getattr(params, "access_key", "") or "").strip()
        secret_access = (getattr(params, "access_secret", "") or "").strip()

        # 2) 项目 .env
        try:
            base_dir = os.path.dirname(__file__)
            env_path = os.path.join(base_dir, "..", "..", ".env")
            if os.path.isfile(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith("#") or "=" not in s:
                            continue
                        k, v = s.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"").strip("'")
                        if k == "INFINI_API_KEY" and v and not key_infini:
                            key_infini = v
                        if k == "API_KEY" and v and not key_generic:
                            key_generic = v
                        if k in ("INFINI_ACCESS_KEY", "ACCESS_KEY", "AccessKey") and v and not key_access:
                            key_access = v
                        if k in ("INFINI_ACCESS_SECRET", "ACCESS_SECRET", "AccessSecret") and v and not secret_access:
                            secret_access = v
        except Exception:
            pass

        # 3) 环境变量兜底
        if not key_infini:
            key_infini = (os.getenv("INFINI_API_KEY") or "").strip()
        if not key_generic:
            key_generic = (os.getenv("API_KEY") or "").strip()
        if not key_access:
            key_access = (os.getenv("INFINI_ACCESS_KEY") or os.getenv("ACCESS_KEY") or os.getenv("AccessKey") or "").strip()
        if not secret_access:
            secret_access = (os.getenv("INFINI_ACCESS_SECRET") or os.getenv("ACCESS_SECRET") or os.getenv("AccessSecret") or "").strip()

        # 注入环境变量
        if key_infini:
            os.environ["INFINI_API_KEY"] = key_infini
            loaded_keys["infini_api_key"] = key_infini

        if key_generic:
            os.environ["API_KEY"] = key_generic
            loaded_keys["api_key"] = key_generic

        if key_access:
            os.environ["INFINI_ACCESS_KEY"] = key_access
            os.environ.setdefault("ACCESS_KEY", key_access)
            loaded_keys["access_key"] = key_access

        if secret_access:
            os.environ["INFINI_ACCESS_SECRET"] = secret_access
            os.environ.setdefault("ACCESS_SECRET", secret_access)
            loaded_keys["access_secret"] = secret_access

        return loaded_keys


@dataclass
class PositionLimitConfig:
    """仓位限额配置"""
    limit_amount: float = 0.0
    effective_until: Optional[datetime] = None
    created_at: Optional[datetime] = None
    account_id: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class PositionType:
    """仓位类型枚举"""
    INTRADAY = "intraday"    # 日内仓（当天开仓）
    OVERNIGHT = "overnight"  # 隔夜仓（非当天开仓）


class OrderStatus:
    """订单状态枚举"""
    PENDING = "pending"      # 等待中
    PARTIAL = "partial"      # 部分成交
    COMPLETED = "completed"  # 完全成交
    CANCELLED = "cancelled"  # 已撤销
    CHASING = "chasing"      # 追单中


@dataclass
class PositionRecord:
    """持仓记录数据类"""
    position_id: str                    # 仓位ID
    instrument_id: str                  # 合约代码
    open_price: float                   # 开仓价
    volume: int                         # 持仓数量
    direction: str                      # 方向: "0"-买, "1"-卖
    open_time: datetime                 # 开仓时间
    open_date: datetime.date            # 开仓日期
    position_type: str                  # 仓位类型（开仓时立即确定）
    stop_profit_price: float            # 止盈价(开仓价150%)
    days_held: int = 0                  # 持仓天数
    chase_count: int = 0                # 追单次数


@dataclass  
class ChaseOrderTask:
    """追单任务数据类"""
    position_id: str                    # 关联的仓位ID
    instrument_id: str                  # 合约代码
    original_order_id: str              # 原始订单ID
    remaining_volume: int               # 剩余数量
    chase_count: int = 0                # 已追单次数
    created_at: datetime = None         # 创建时间
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class OptionBuyOpenExecutor:
    """期权买方开仓执行器"""

    def __init__(self, strategy_instance: Any = None):
        self.strategy = strategy_instance
        self.params = getattr(strategy_instance, "params", None)
        self.limit_configs: Dict[str, PositionLimitConfig] = {}
        self.config_file = "option_buy_limits.json"

        self._request_id_base = int(time.time() * 1000)
        self._request_id_counter = 0
        self._lock = threading.RLock()

        self._load_configs()

    def set_position_limit(
        self,
        account_id: str,
        limit_amount: float,
        valid_hours: Optional[int] = None,
        force_set: bool = False
    ) -> Tuple[bool, str]:
        """设置开仓资金限额"""
        try:
            if not account_id or not isinstance(account_id, str):
                return False, "账户ID无效"

            if not isinstance(limit_amount, (int, float)) or limit_amount <= 0:
                return False, "限额金额无效"

            if valid_hours is None:
                try:
                    valid_hours = int(getattr(self.params, "position_limit_default_valid_hours", 24) or 24)
                except Exception:
                    valid_hours = 24

            max_hours = 720
            try:
                max_hours = int(getattr(self.params, "position_limit_valid_hours_max", 720) or 720)
            except Exception:
                pass
            if not 1 <= valid_hours <= max_hours:
                return False, "有效小时数超出范围"

            total_capital = self._get_account_total_capital(account_id)
            if total_capital is None:
                return False, "无法获取账户总资金"

            if not force_set:
                ratio = 0.2
                try:
                    ratio = float(getattr(self.params, "position_limit_max_ratio", 0.2) or 0.2)
                except Exception:
                    pass
                max_allowed = total_capital * ratio
                if limit_amount > max_allowed:
                    return False, f"超额限制：限额{limit_amount:.2f}元＞{max_allowed:.2f}元（{ratio*100:.0f}%）"

            min_amt = 1000
            try:
                min_amt = float(getattr(self.params, "position_limit_min_amount", 1000) or 1000)
            except Exception:
                pass
            if limit_amount < min_amt:
                return False, f"限额至少{min_amt:.0f}元"

            effective_until = datetime.now() + timedelta(hours=valid_hours)

            config = PositionLimitConfig(
                limit_amount=float(limit_amount),
                effective_until=effective_until,
                account_id=account_id
            )

            with self._lock:
                self.limit_configs[account_id] = config
                self._save_configs()

            return True, f"买方开仓限额设置成功：{limit_amount:.2f}元"

        except Exception:
            return False, "设置失败"

    def execute_option_buy_open(
        self,
        account_id: str,
        option_data: Dict[str, Any],
        lots: int = 1
    ) -> Tuple[bool, str, Optional[Dict]]:
        """执行期权买入开仓"""
        try:
            if not account_id or not isinstance(account_id, str):
                return False, "账户ID无效", None

            lots_min = 1
            lots_max = 100
            try:
                if self.strategy and hasattr(self.strategy, "params"):
                    params_obj = self.strategy.params
                    lots_min = int(
                        getattr(params_obj, "option_buy_lots_min", getattr(params_obj, "lots_min", 1)) or 1
                    )
                    lots_max = int(
                        getattr(params_obj, "option_buy_lots_max", getattr(params_obj, "lots_max", 100)) or 100
                    )
            except Exception:
                pass
            if not isinstance(lots, int) or not lots_min <= lots <= lots_max:
                return False, f"手数无效：{lots}", None

            if not self._check_position_limit(account_id):
                return False, "无有效资金限额", None

            try:
                if self.strategy and hasattr(self.strategy, "_is_instrument_allowed"):
                    inst_id = option_data.get("instrument_id", "")
                    exch = option_data.get("exchange", "")
                    if not self.strategy._is_instrument_allowed(inst_id, exch):
                        return False, "合约不在期权链或非指定月", None
            except Exception:
                pass

            validation_result = self._validate_option_data(option_data)
            if not validation_result[0]:
                return False, validation_result[1], None

            option_price = validation_result[2]

            contract_multiplier = 10000
            try:
                if self.strategy and hasattr(self.strategy, "params"):
                    params_obj = self.strategy.params
                    contract_multiplier = float(
                        getattr(params_obj, "option_contract_multiplier", getattr(params_obj, "contract_multiplier", 10000))
                        or 10000
                    )
            except Exception:
                pass
            required_amount = option_price * contract_multiplier * lots

            available, avail_msg = self._check_available_amount(account_id, required_amount)
            if not available:
                return False, avail_msg, None

            broker_id = self._get_broker_id()
            user_id = self._get_user_id()

            if not broker_id:
                return False, "缺少经纪商代码", None

            if not user_id:
                return False, "缺少用户代码", None

            order_request = self._build_ctp_order(
                broker_id=broker_id,
                user_id=user_id,
                account_id=account_id,
                option_data=option_data,
                price=option_price,
                volume=lots
            )

            if not self.strategy:
                return False, "策略实例未提供", None

            if not hasattr(self.strategy, "send_order"):
                return False, "策略缺少send_order方法", None

            order_result = self.strategy.send_order(order_request)

            if not isinstance(order_result, dict):
                return False, "委托返回格式错误", None

            if not order_result.get("success", False):
                error_msg = order_result.get("message", "委托失败")
                return False, error_msg, None

            self._update_limit_after_order(account_id, required_amount)

            order_response = {
                "order_id": order_result.get("order_id", ""),
                "order_sys_id": order_result.get("order_sys_id", ""),
                "order_time": datetime.now(),
                "order_status": order_result.get("status", "submitted"),
                "account_id": account_id,
                "instrument_id": option_data.get("instrument_id", ""),
                "exchange": option_data.get("exchange", "CFFEX"),
                "option_type": option_data.get("option_type", "C"),
                "direction": "BUY",
                "offset_flag": "OPEN",
                "price": option_price,
                "volume": lots,
                "premium_total": required_amount,
                "contract_multiplier": contract_multiplier,
                "front_id": order_result.get("front_id", ""),
                "session_id": order_result.get("session_id", ""),
            }

            # 记录交易明细（买入开仓）用于日结
            try:
                if self.strategy and hasattr(self.strategy, "record_trade_event"):
                    self.strategy.record_trade_event(
                        side="BUY",
                        offset="OPEN",
                        instrument_id=option_data.get("instrument_id", ""),
                        exchange=option_data.get("exchange", "CFFEX"),
                        price=option_price,
                        volume=lots,
                        account_id=account_id,
                        extra={
                            "option_type": option_data.get("option_type", ""),
                            "premium_total": required_amount,
                        },
                    )
            except Exception:
                pass

            return True, "买入开仓委托成功", order_response

        except Exception:
            return False, "开仓执行失败", None

    def _validate_option_data(self, option_data: Dict) -> Tuple[bool, str, float]:
        """验证期权数据"""
        try:
            if "instrument_id" not in option_data:
                return False, "缺少合约代码", 0.0

            if "current_price" not in option_data:
                return False, "缺少当前价格", 0.0

            try:
                price = float(option_data["current_price"])
                if price <= 0:
                    return False, "价格必须大于0", 0.0
            except (ValueError, TypeError):
                return False, "价格格式错误", 0.0

            option_type = option_data.get("option_type", "").upper()
            if option_type not in ["C", "P"]:
                instrument_id = option_data["instrument_id"].upper()
                if "C" in instrument_id:
                    option_type = "C"
                elif "P" in instrument_id:
                    option_type = "P"
                else:
                    return False, "无法确定期权类型", 0.0

            option_data["option_type"] = option_type

            if "exchange" not in option_data:
                option_data["exchange"] = "CFFEX"

            return True, "验证通过", price

        except Exception:
            return False, "数据验证失败", 0.0

    def _check_position_limit(self, account_id: str) -> bool:
        """检查资金限额"""
        with self._lock:
            if account_id not in self.limit_configs:
                return False

            config = self.limit_configs[account_id]

            now = datetime.now()
            if config.effective_until and now > config.effective_until:
                del self.limit_configs[account_id]
                self._save_configs()
                return False

            if config.limit_amount <= 0:
                del self.limit_configs[account_id]
                self._save_configs()
                return False

            return True

    def _check_available_amount(self, account_id: str, required_amount: float) -> Tuple[bool, str]:
        """检查可用资金"""
        try:
            with self._lock:
                config = self.limit_configs.get(account_id)
                if not config:
                    return False, "无有效资金限额"

                if config.limit_amount < required_amount:
                    return False, "买方限额不足"

                available_capital = self._get_account_available_capital(account_id)
                if available_capital is None:
                    return False, "无法获取账户可用资金"

                if available_capital < required_amount:
                    return False, "账户资金不足"

                return True, "资金充足"

        except Exception:
            return False, "资金检查失败"

    def _build_ctp_order(
        self,
        broker_id: str,
        user_id: str,
        account_id: str,
        option_data: Dict[str, Any],
        price: float,
        volume: int
    ) -> Dict[str, Any]:
        """构建CTP委托请求"""
        exchange = option_data.get("exchange", "CFFEX")
        instrument_id = option_data.get("instrument_id", "")
        option_type = option_data.get("option_type", "C")

        params_obj = getattr(self.strategy, "params", None)
        order_price_type = "2"
        order_time_condition = "3"
        order_volume_condition = "1"
        order_contingent_condition = "1"
        order_force_close_reason = "0"
        order_hedge_flag = "1"
        order_min_volume = 1
        order_business_unit = "1"
        order_is_auto_suspend = 0
        order_user_force_close = 0
        order_is_swap = 0

        try:
            if params_obj:
                order_price_type = str(getattr(params_obj, "option_order_price_type", order_price_type) or order_price_type)
                order_time_condition = str(
                    getattr(params_obj, "option_order_time_condition", order_time_condition) or order_time_condition
                )
                order_volume_condition = str(
                    getattr(params_obj, "option_order_volume_condition", order_volume_condition) or order_volume_condition
                )
                order_contingent_condition = str(
                    getattr(params_obj, "option_order_contingent_condition", order_contingent_condition)
                    or order_contingent_condition
                )
                order_force_close_reason = str(
                    getattr(params_obj, "option_order_force_close_reason", order_force_close_reason)
                    or order_force_close_reason
                )
                order_hedge_flag = str(getattr(params_obj, "option_order_hedge_flag", order_hedge_flag) or order_hedge_flag)
                order_min_volume = int(getattr(params_obj, "option_order_min_volume", order_min_volume) or order_min_volume)
                order_business_unit = str(
                    getattr(params_obj, "option_order_business_unit", order_business_unit) or order_business_unit
                )
                order_is_auto_suspend = int(
                    getattr(params_obj, "option_order_is_auto_suspend", order_is_auto_suspend) or order_is_auto_suspend
                )
                order_user_force_close = int(
                    getattr(params_obj, "option_order_user_force_close", order_user_force_close) or order_user_force_close
                )
                order_is_swap = int(getattr(params_obj, "option_order_is_swap", order_is_swap) or order_is_swap)
        except Exception:
            pass

        request_id = self._generate_unique_request_id()

        order_request = {
            "BrokerID": broker_id,
            "InvestorID": account_id,
            "UserID": user_id,
            "ExchangeID": exchange,
            "InstrumentID": instrument_id,
            "Direction": "0",
            "OffsetFlag": "0",
            "OrderPriceType": order_price_type,
            "LimitPrice": float(price),
            "VolumeTotalOriginal": int(volume),
            "OptionsType": "1" if option_type == "C" else "2",
            "TimeCondition": order_time_condition,
            "VolumeCondition": order_volume_condition,
            "ContingentCondition": order_contingent_condition,
            "ForceCloseReason": order_force_close_reason,
            "HedgeFlag": order_hedge_flag,
            "MinVolume": order_min_volume,
            "IsAutoSuspend": order_is_auto_suspend,
            "UserForceClose": order_user_force_close,
            "IsSwapOrder": order_is_swap,
            "BusinessUnit": order_business_unit,
            "RequestID": request_id,
            "OrderRef": self._generate_order_ref(),
        }

        strike_price = option_data.get("strike_price")
        if strike_price is not None:
            order_request["StrikePrice"] = float(strike_price)

        return order_request

    def _generate_unique_request_id(self) -> int:
        """生成唯一请求ID"""
        self._request_id_counter += 1
        return self._request_id_base + self._request_id_counter

    def _generate_order_ref(self) -> str:
        """生成订单引用"""
        return f"BUY_{datetime.now().strftime('%H%M%S')}"

    def _get_account_available_capital(self, account_id: str) -> Optional[float]:
        """获取可用资金"""
        if not self.strategy:
            return None

        try:
            if hasattr(self.strategy, "get_account"):
                result = self.strategy.get_account(account_id)
            elif hasattr(self.strategy, "query_account"):
                result = self.strategy.query_account(account_id)
            else:
                return None

            if result and isinstance(result, dict):
                if "Available" in result:
                    return float(result["Available"])

                for field in ("available", "可用资金"):
                    if field in result:
                        value = result[field]
                        if isinstance(value, (int, float)):
                            return float(value)
                        if isinstance(value, str):
                            try:
                                return float(value)
                            except ValueError:
                                continue
        except Exception:
            pass

        return None

    def _get_account_total_capital(self, account_id: str) -> Optional[float]:
        """获取总资金"""
        if not self.strategy:
            return None

        try:
            if hasattr(self.strategy, "get_account"):
                result = self.strategy.get_account(account_id)
            elif hasattr(self.strategy, "query_account"):
                result = self.strategy.query_account(account_id)
            else:
                return None

            if result and isinstance(result, dict):
                if "Balance" in result:
                    return float(result["Balance"])

                for field in ("balance", "总资产", "TotalAsset"):
                    if field in result:
                        value = result[field]
                        if isinstance(value, (int, float)):
                            return float(value)
                        if isinstance(value, str):
                            try:
                                return float(value)
                            except ValueError:
                                continue
        except Exception:
            pass

        return None

    def _get_broker_id(self) -> str:
        """获取经纪商代码"""
        if self.strategy and hasattr(self.strategy, "broker_id"):
            broker_id = self.strategy.broker_id
            if broker_id:
                return str(broker_id)

        try:
            config_file = "broker_config.json"
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)
                return config.get("broker_id", "")
        except Exception:
            pass

        return ""

    def _get_user_id(self) -> str:
        """获取用户代码"""
        if self.strategy and hasattr(self.strategy, "user_id"):
            user_id = self.strategy.user_id
            if user_id:
                return str(user_id)

        try:
            config_file = "user_config.json"
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)
                return config.get("user_id", "")
        except Exception:
            pass

        return ""

    def _update_limit_after_order(self, account_id: str, used_amount: float):
        """更新限额"""
        with self._lock:
            if account_id in self.limit_configs:
                config = self.limit_configs[account_id]
                config.limit_amount -= used_amount

                if config.limit_amount <= 0:
                    del self.limit_configs[account_id]

                self._save_configs()

    def _load_configs(self):
        """加载配置"""
        try:
            if not os.path.exists(self.config_file):
                return

            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            with self._lock:
                for account_id, config_data in data.items():
                    try:
                        if not isinstance(config_data, dict):
                            continue

                        if "effective_until" in config_data and isinstance(config_data["effective_until"], str):
                            config_data["effective_until"] = datetime.strptime(
                                config_data["effective_until"], "%Y-%m-%d %H:%M:%S"
                            )

                        if "created_at" in config_data and isinstance(config_data["created_at"], str):
                            config_data["created_at"] = datetime.strptime(
                                config_data["created_at"], "%Y-%m-%d %H:%M:%S"
                            )

                        config = PositionLimitConfig(**config_data)

                        if config.effective_until and datetime.now() > config.effective_until:
                            continue

                        self.limit_configs[account_id] = config

                    except Exception:
                        continue

        except Exception:
            with self._lock:
                self.limit_configs = {}

    def _save_configs(self):
        """保存配置"""
        try:
            save_data = {}
            for account_id, config in self.limit_configs.items():
                if not self._is_limit_valid(config):
                    continue

                save_data[account_id] = {
                    "limit_amount": float(config.limit_amount),
                    "account_id": config.account_id,
                    "effective_until": config.effective_until.strftime("%Y-%m-%d %H:%M:%S")
                    if config.effective_until
                    else None,
                    "created_at": config.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    if config.created_at
                    else None,
                }

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

        except Exception:
            pass

    def _is_limit_valid(self, config: PositionLimitConfig) -> bool:
        """检查限额有效性"""
        if config.limit_amount <= 0:
            return False

        if config.effective_until and datetime.now() > config.effective_until:
            return False

        return True


class PositionManager:
    """
    平仓管理器 - 整合自20260113.py
    功能：
    1. 开仓成功立即按开仓价的150%发出止盈指令
    2. 15:58分前平去所有日内仓
    3. 隔夜仓在14:58时亏损50%或盈利400%对价平仓
    4. 开仓超过3天，在15:58分全部对价平仓
    5. 对价平仓 + 追单引擎确保成交
    6. 完全自动执行
    7. 完整仓位信息查询
    """
    
    def __init__(self, strategy_instance: Any = None):
        """初始化"""
        self.strategy = strategy_instance
        params_obj = getattr(self.strategy, "params", None)

        def _get_param(name: str, default: Any, caster):
            try:
                val = getattr(params_obj, name, default)
                if val is None:
                    return default
                return caster(val)
            except Exception:
                return default

        def _parse_time(val: Any, default_str: str):
            try:
                if isinstance(val, dtime):
                    return val
                if isinstance(val, str) and val:
                    return datetime.strptime(val, "%H:%M").time()
            except Exception:
                pass
            try:
                return datetime.strptime(default_str, "%H:%M").time()
            except Exception:
                return dtime(0, 0, 0)
        
        # 数据存储
        self.position_records: Dict[str, PositionRecord] = {}  # 持仓记录
        self.active_orders: Dict[str, Dict] = {}               # 活跃订单追踪
        self.chase_tasks: Dict[str, ChaseOrderTask] = {}       # 追单任务队列
        self.latest_ticks: Dict[str, Any] = {}                 # 最新行情缓存
        
        # 线程锁（保证线程安全，使用可重入锁避免死锁）
        self._lock = threading.RLock()
        
        # 定时器ID
        self.timer_id: Optional[str] = None
        self.chase_timer_id: Optional[str] = None
        
        # 时间配置
        self.TIME_CHECK_OVERNIGHT = _parse_time(_get_param("close_overnight_check_time", "14:58", str), "14:58")
        self.TIME_CLOSE_POSITIONS = _parse_time(_get_param("close_daycut_time", "15:58", str), "15:58")
        
        # 参数配置
        self.stop_profit_ratio: float = _get_param("close_take_profit_ratio", 1.5, float)
        self.overnight_loss_threshold: float = _get_param("close_overnight_loss_threshold", -0.5, float)
        self.overnight_profit_threshold: float = _get_param("close_overnight_profit_threshold", 4.0, float)
        self.close_max_hold_days: int = _get_param("close_max_hold_days", 3, int)
        self.max_chase_attempts: int = _get_param("close_max_chase_attempts", 5, int)
        self.chase_interval_seconds: int = _get_param("close_chase_interval_seconds", 2, int)
        self.chase_task_timeout_seconds: int = _get_param("close_chase_task_timeout_seconds", 30, int)
        self.delayed_close_timeout_seconds: int = _get_param("close_delayed_timeout_seconds", 30, int)
        self.delayed_close_max_retries: int = _get_param("close_delayed_max_retries", 3, int)
        self.close_order_price_type: str = str(_get_param("close_order_price_type", "2", str))
        
        if self.strategy:
            self.strategy.output("平仓管理器初始化完成")
    
    def start(self) -> None:
        """启动平仓管理器"""
        if not self.strategy:
            return
            
        # 启动定时器（使用 Scheduler），回测可关闭
        try:
            if getattr(self.strategy.params, "enable_scheduler", True) and hasattr(self.strategy, 'scheduler') and self.strategy.scheduler:
                # 每秒检查定时器
                self.strategy.scheduler.add_job(
                    func=self._on_second_timer,
                    trigger="interval",
                    id="position_manager_second_timer",
                    seconds=1,
                    replace_existing=True,
                )
                # 追单定时器
                self.strategy.scheduler.add_job(
                    func=self._process_chase_tasks,
                    trigger="interval",
                    id="position_manager_chase_timer",
                    seconds=self.chase_interval_seconds,
                    replace_existing=True,
                )
            else:
                if self.strategy:
                    self.strategy._debug("平仓管理器定时器已跳过（enable_scheduler=False，回测模式）")
        except Exception as e:
            if self.strategy:
                self.strategy.output(f"平仓管理器定时器启动失败: {e}", force=True)
        
        # 加载已有持仓
        self._load_existing_positions()
        
        if self.strategy:
            self.strategy.output("平仓管理器启动完成")
    
    def stop(self) -> None:
        """停止平仓管理器"""
        # 取消所有定时器（使用 Scheduler）
        try:
            if hasattr(self.strategy, 'scheduler') and self.strategy.scheduler:
                self.strategy.scheduler.remove_job("position_manager_second_timer")
                self.strategy.scheduler.remove_job("position_manager_chase_timer")
        except Exception as e:
            if self.strategy:
                self.strategy.output(f"平仓管理器定时器停止失败: {e}", force=True)
            
        # 取消所有活跃订单
        self._cancel_all_pending_orders()
        
        # 清理所有字典，避免内存泄漏
        self.position_records.clear()
        self.active_orders.clear()
        self.chase_tasks.clear()
        self.latest_ticks.clear()
        if hasattr(self, 'delayed_close_tasks'):
            self.delayed_close_tasks.clear()
        
        if self.strategy:
            self.strategy.output("平仓管理器停止，所有资源已清理")
    
    def handle_new_position(self, trade_data: Any) -> None:
        """处理新开仓 - 规则1（修正：开仓时立即确定仓位类型）"""
        position_id = self._generate_position_id(trade_data)
        current_date = datetime.now().date()
        
        # 核心修正：根据开仓日期立即确定仓位类型
        open_date = getattr(trade_data, 'trade_date', current_date)
        
        if isinstance(open_date, str):
            # 如果是字符串格式，转换为日期对象
            open_date = datetime.strptime(open_date, "%Y%m%d").date()
        
        # 判断仓位类型
        if open_date == current_date:
            position_type = PositionType.INTRADAY  # 当天开仓 = 日内仓
        else:
            position_type = PositionType.OVERNIGHT  # 非当天开仓 = 隔夜仓
        
        # 创建持仓记录
        record = PositionRecord(
            position_id=position_id,
            instrument_id=trade_data.instrument_id,
            open_price=trade_data.price,
            volume=trade_data.volume,
            direction=trade_data.direction,
            open_time=datetime.now(),
            open_date=open_date,
            position_type=position_type,  # 开仓时已确定
            stop_profit_price=trade_data.price * self.stop_profit_ratio  # 参数化止盈倍数
        )
        
        self.position_records[position_id] = record
        
        # 立即设置止盈单（对价平仓）
        self._place_stop_profit_order(record)
        
        if self.strategy:
            self.strategy.output(f"[规则1] 新开仓 | ID:{position_id} | "
                       f"类型:{position_type} | "
                       f"开仓日:{open_date} | 价格:{trade_data.price:.2f} | "
                       f"止盈价:{record.stop_profit_price:.2f}")
    
    def handle_tick(self, tick: Any) -> None:
        """行情Tick回调 - 核心数据源"""
        with self._lock:
            # 1. 缓存最新行情
            self.latest_ticks[tick.instrument_id] = tick
            
            # 2. 实时止盈检查
            self._check_stop_profit_realtime(tick)
            
            # 3. 检查是否需要立即追单
            self._check_immediate_chase(tick.instrument_id)
    
    def on_start(self, *args: Any, **kwargs: Any) -> None:
        """策略启动：由平台事件驱动，在此处执行启动逻辑"""
        try:
            super().on_start(*args, **kwargs)
            self.output("[调试] on_start() 方法开始执行")

            # 交易/回测场景静默测试/诊断输出
            self._enforce_diagnostic_silence()
            self.output("=== 开始诊断数据通路 ===")

            try:
                infini_result = infini.get_instruments_by_product(exchange="CFFEX", product_id="IF")
                count = len(infini_result) if infini_result else 0
                self.output(f"[诊断] infini SDK 直接调用: 获取到 {count} 条IF合约", force=True)
            except Exception as exc:
                self.output(f"[诊断] infini SDK 直接调用失败: {exc}", force=True)

            try:
                mc_result = self.market_center.get_instrument_list(exchange="CFFEX")
                count = len(mc_result) if mc_result else 0
                self.output(f"[诊断] MarketCenter 调用: 获取到 {count} 条CFFEX合约", force=True)
            except AttributeError as exc:
                self.output(f"[诊断] MarketCenter 方法不存在: {exc}. 正在检查可用方法...", force=True)
                try:
                    attrs = [m for m in dir(self.market_center) if not m.startswith('_')]
                    self.output(f"[诊断] MarketCenter 对象属性: {attrs}", force=True)
                except Exception as inner_exc:
                    self.output(f"[诊断] 无法列出 MarketCenter 属性: {inner_exc}", force=True)
            except Exception as exc:
                self.output(f"[诊断] MarketCenter 调用异常: {exc}", force=True)

            try:
                # [Fix] 使用底层 MarketCenter 获取合约数据，移除不存在的 self.get_instrument 调用
                inst = None
                if getattr(self, "market_center", None) and hasattr(self.market_center, "get_instrument_data"):
                    inst = self.market_center.get_instrument_data(exchange="SHFE", instrument_id="ag2406")
                result = "成功" if inst else "返回空"
                self.output(f"[诊断] MarketCenter.get_instrument_data调用: {result}", force=True)
            except Exception as exc:
                self.output(f"[诊断] MarketCenter.get_instrument_data调用异常: {exc}", force=True)

            self.output("=== 诊断结束 ===")
            self.my_state = "starting"
            self.my_started = True
            self.my_is_running = True
            self.my_is_paused = False
            self.my_destroyed = False
            self.my_trading = True

            self.output("策略启动逻辑开始...")

            try:
                if bool(getattr(self.params, "enable_output_mode_ui", True)) and not (getattr(self, "_ui_running", False) or getattr(self, "_ui_creating", False)):
                    self._start_output_mode_ui()
            except Exception as exc:
                try:
                    self.output(f"输出模式界面启动失败: {exc}", force=True)
                except Exception:
                    pass

            try:
                if hasattr(self, 'position_manager') and self.position_manager:
                    self.position_manager.start()
                    self.output("平仓管理器已启动", force=True)
            except Exception as exc:
                try:
                    self.output(f"平仓管理器启动失败: {exc}", force=True)
                except Exception:
                    pass

            self.output("=== on_start - 状态快照 ===")
            self._log_status_snapshot("on_start")

            self.output("[调试] on_start() - 即将在后台线程中调用 start() 方法")
            import threading

            def _start_in_thread():
                try:
                    self.start()
                    self.my_state = "running"
                    self.my_trading = True
                    self.output("[调试] start() 方法在后台线程中执行完成")
                except Exception as exc:
                    self.output(f"start() 方法在后台线程中执行失败: {exc}\n{traceback.format_exc()}")
                    self.my_state = "error"
                    self.my_is_running = False
                    self.my_trading = False

            threading.Thread(target=_start_in_thread, daemon=True).start()
            self.output("[调试] on_start() - 已在后台线程中启动 start() 方法，on_start 即将返回")

            self.output("=== 2026-01-09 23:50 修改版本已加载 ===")
            self.output("=== on_start 执行完成，策略应已启动 ===")

            self._load_api_key()
        except Exception as e:
            self.output(f"加载 API Key 失败: {e}")

    def _execute_1558_closing(self) -> None:
        """执行15:58平仓 - 规则3&6"""
        if self.strategy:
            self.strategy.output(f"[规则3&6] 开始执行15:58平仓逻辑")

        current_date = datetime.now().date()
        positions_to_close: list[tuple[str, str]] = []

        with self._lock:
            for position_id, record in self.position_records.items():
                if record.volume <= 0:
                    continue

                close_reason = None
                if record.position_type == PositionType.INTRADAY:
                    close_reason = "15:58平日内仓"
                elif record.position_type == PositionType.OVERNIGHT:
                    days_held = (current_date - record.open_date).days
                    if self.close_max_hold_days and days_held >= self.close_max_hold_days:
                        close_reason = f"持仓超{days_held}天平仓"

                if close_reason:
                    positions_to_close.append((position_id, close_reason))

        for position_id, reason in positions_to_close:
            self._close_position_at_opposite_price(position_id, reason)
    
    def _check_overnight_positions_1458(self) -> None:
        """检查隔夜仓盈亏 - 规则4&5"""
        with self._lock:
            if self.strategy:
                self.strategy.output(f"[规则4&5] 开始检查隔夜仓盈亏")
            
            for position_id, record in self.position_records.items():
                if (record.volume <= 0 or 
                    record.position_type != PositionType.OVERNIGHT):
                    continue
                
                # 获取最新价格
                tick = self.latest_ticks.get(record.instrument_id)
                if not tick or record.open_price <= 0:
                    continue
                
                # 获取最新价格（兼容不同的 tick 字段）
                current_price = getattr(tick, "last", None) or getattr(tick, "last_price", None) or getattr(tick, "price", None)
                if current_price is None:
                    continue
                
                profit_rate = (current_price - record.open_price) / record.open_price
                
                # 规则4：亏损阈值平仓
                if profit_rate <= self.overnight_loss_threshold:
                    self._close_position_at_opposite_price(
                        position_id, 
                        f"隔夜仓亏损{profit_rate:.1%}平仓"
                    )
                    
                # 规则5：盈利阈值平仓  
                elif profit_rate >= self.overnight_profit_threshold:
                    self._close_position_at_opposite_price(
                        position_id,
                        f"隔夜仓盈利{profit_rate:.1%}平仓"
                    )
    
    def _close_position_at_opposite_price(self, position_id: str, reason: str) -> None:
        """对价平仓核心方法"""
        try:
            # 先在锁内读取共享数据，释放后再下单，缩短持锁时间
            with self._lock:
                record = self.position_records.get(position_id)
                if not record or record.volume <= 0:
                    return

                tick = self.latest_ticks.get(record.instrument_id)
                if not tick:
                    if self.strategy:
                        self.strategy.output(f"警告：无法获取{record.instrument_id}行情，平仓推迟")
                    self._schedule_delayed_close(position_id, reason)
                    return

                if record.direction == "0":
                    close_direction = "1"
                    opposite_price = getattr(tick, "bid", None) or getattr(tick, "BidPrice1", None)
                    price_type = "BID1"
                else:
                    close_direction = "0"
                    opposite_price = getattr(tick, "ask", None) or getattr(tick, "AskPrice1", None)
                    price_type = "ASK1"

                if opposite_price is None:
                    if self.strategy:
                        self.strategy.output(f"警告：无法获取{record.instrument_id}对手价，平仓推迟")
                    self._schedule_delayed_close(position_id, reason)
                    return

                instrument_id = record.instrument_id
                volume = record.volume

            # 发单在锁外执行，避免阻塞其他线程
            if self.strategy:
                order_id = self.strategy.place_order(
                    exchange=self.strategy.params.exchange if hasattr(self.strategy.params, 'exchange') else "SHFE",
                    instrument_id=instrument_id,
                    direction=close_direction,
                    offset_flag="1",  # 平仓
                    price=float(opposite_price),
                    volume=volume,
                    order_price_type=self.close_order_price_type  # 参数化平仓委托价类型
                )
            else:
                order_id = None

            if order_id:
                with self._lock:
                    self.active_orders[order_id] = {
                        "position_id": position_id,
                        "instrument_id": instrument_id,
                        "original_volume": volume,
                        "traded_volume": 0,
                        "status": OrderStatus.PENDING,
                        "chase_count": 0,
                        "created_at": datetime.now(),
                        "close_reason": reason
                    }

                if self.strategy:
                    self.strategy.output(
                        f"[对价平仓] {reason} | {instrument_id} | 方向:{close_direction} | "
                        f"价格{price_type}:{float(opposite_price):.2f} | 数量:{volume} | 订单ID:{order_id}"
                    )
            else:
                if self.strategy:
                    self.strategy.output(f"错误：平仓委托失败 {instrument_id}")
        except Exception as e:
            if self.strategy:
                self.strategy.output(f"异常：对价平仓失败 {position_id} - {e}", force=True)
    
    def _process_chase_tasks(self) -> None:
        """处理追单任务"""
        try:
            # 两段式：锁内准备任务，锁外下单，锁内更新状态
            tasks_to_remove: list[str] = []
            tasks_to_place: list[dict] = []
            current_time = datetime.now()

            with self._lock:
                if not self.chase_tasks:
                    return

                for position_id, task in list(self.chase_tasks.items()):
                    if task.chase_count >= self.max_chase_attempts:
                        if self.strategy:
                            self.strategy.output(f"警告：{position_id} 达到最大追单次数，停止追单")
                        tasks_to_remove.append(position_id)
                        continue

                    if (current_time - task.created_at).total_seconds() > self.chase_task_timeout_seconds:
                        if self.strategy:
                            self.strategy.output(f"警告：{position_id} 追单任务超时")
                        tasks_to_remove.append(position_id)
                        continue

                    tick = self.latest_ticks.get(task.instrument_id)
                    if not tick:
                        continue

                    record = self.position_records.get(position_id)
                    if not record or record.volume <= 0:
                        tasks_to_remove.append(position_id)
                        continue

                    if record.direction == "0":
                        close_direction = "1"
                        opposite_price = getattr(tick, "bid", None) or getattr(tick, "BidPrice1", None)
                    else:
                        close_direction = "0"
                        opposite_price = getattr(tick, "ask", None) or getattr(tick, "AskPrice1", None)

                    if opposite_price is None:
                        tasks_to_remove.append(position_id)
                        continue

                    tasks_to_place.append({
                        "position_id": position_id,
                        "instrument_id": task.instrument_id,
                        "remaining_volume": task.remaining_volume,
                        "close_direction": close_direction,
                        "opposite_price": float(opposite_price),
                    })

            order_results: list[tuple[str, Optional[str], dict]] = []
            for item in tasks_to_place:
                chase_order_id = None
                if self.strategy:
                    chase_order_id = self.strategy.place_order(
                        exchange=self.strategy.params.exchange if hasattr(self.strategy.params, 'exchange') else "SHFE",
                        instrument_id=item["instrument_id"],
                        direction=item["close_direction"],
                        offset_flag="1",
                        price=item["opposite_price"],
                        volume=item["remaining_volume"],
                        order_price_type=self.close_order_price_type,
                    )
                order_results.append((item["position_id"], chase_order_id, item))

            with self._lock:
                for position_id, order_id, item in order_results:
                    if order_id:
                        task = self.chase_tasks.get(position_id)
                        if task:
                            task.chase_count += 1
                        self.active_orders[order_id] = {
                            "position_id": position_id,
                            "instrument_id": item["instrument_id"],
                            "original_volume": item["remaining_volume"],
                            "traded_volume": 0,
                            "status": OrderStatus.CHASING,
                            "chase_count": task.chase_count if task else 1,
                            "created_at": datetime.now(),
                            "close_reason": f"追单#{task.chase_count if task else 1}"
                        }
                        if self.strategy:
                            self.strategy.output(
                                f"[追单#{task.chase_count if task else 1}] {item['instrument_id']} | "
                                f"价格:{item['opposite_price']:.2f} | 数量:{item['remaining_volume']}"
                            )
                        tasks_to_remove.append(position_id)
                    else:
                        tasks_to_remove.append(position_id)

                for pid in tasks_to_remove:
                    self.chase_tasks.pop(pid, None)
        except Exception as e:
            if self.strategy:
                self.strategy.output(f"异常：追单任务处理失败 - {e}", force=True)
    
    def _on_second_timer(self) -> None:
        """每秒定时器（修正：移除所有14:30相关逻辑）"""
        try:
            now = datetime.now()
            current_time = now.time()

            with self._lock:
                # 处理延迟平仓任务
                if hasattr(self, 'delayed_close_tasks'):
                    tasks_to_remove = []
                    for position_id, task in list(self.delayed_close_tasks.items()):
                        # 检查是否超过重试次数或超时时间
                        retry_count = task.get('retry_count', 0)
                        created_at = task.get('created_at', now)

                        # 超过超时或重试限制，放弃延迟平仓
                        if ((now - created_at).total_seconds() > self.delayed_close_timeout_seconds or
                            retry_count >= self.delayed_close_max_retries):
                            tasks_to_remove.append(position_id)
                            if self.strategy:
                                self.strategy.output(f"[延迟平仓超时] {position_id} 已放弃，原因：超时或重试次数过多")
                            continue

                        # 尝试执行延迟平仓
                        record = self.position_records.get(position_id)
                        if not record:
                            tasks_to_remove.append(position_id)
                            continue

                        tick = self.latest_ticks.get(record.instrument_id)
                        if tick:
                            self._close_position_at_opposite_price(position_id, task['reason'])
                            task['retry_count'] += 1
                            record = self.position_records.get(position_id)
                            if record and record.volume <= 0:
                                tasks_to_remove.append(position_id)

                    # 清理已处理的延迟任务
                    for position_id in tasks_to_remove:
                        self.delayed_close_tasks.pop(position_id, None)

                # 规则4&5：14:58检查隔夜仓（14:58:00-14:58:59）
                time_check_overnight_end = dtime(self.TIME_CHECK_OVERNIGHT.hour, self.TIME_CHECK_OVERNIGHT.minute, 59)
                if self.TIME_CHECK_OVERNIGHT <= current_time <= time_check_overnight_end:
                    self._check_overnight_positions_1458()

                # 规则3&6：15:58执行平仓（15:58:00-15:58:59）
                time_close_positions_end = dtime(self.TIME_CLOSE_POSITIONS.hour, self.TIME_CLOSE_POSITIONS.minute, 59)
                if self.TIME_CLOSE_POSITIONS <= current_time <= time_close_positions_end:
                    self._execute_1558_closing()
        except Exception as e:
            if self.strategy:
                self.strategy.output(f"异常：定时器执行失败 - {e}", force=True)
    
    def _check_stop_profit_realtime(self, tick: Any) -> None:
        """实时止盈检查"""
        with self._lock:
            # 获取最新价格（兼容不同的 tick 字段）
            current_price = getattr(tick, "last", None) or getattr(tick, "last_price", None) or getattr(tick, "price", None)
            if current_price is None:
                return
                
            for position_id, record in self.position_records.items():
                if (record.volume > 0 and 
                    record.instrument_id == tick.instrument_id and
                    current_price >= record.stop_profit_price):
                    
                    self._close_position_at_opposite_price(
                        position_id, 
                        f"止盈触发({current_price:.2f}>={record.stop_profit_price:.2f})"
                    )
    
    def _check_immediate_chase(self, instrument_id: str) -> None:
        """立即追单检查（在on_tick中调用）"""
        with self._lock:
            # 检查是否有该合约的追单任务且距离上次追单已超过最小间隔
            for task in list(self.chase_tasks.values()):
                if task.instrument_id == instrument_id:
                    # 可以在这里实现更积极的追单逻辑
                    pass
    
    def _update_position_after_complete(self, position_id: str, traded_volume: int) -> None:
        """更新持仓记录"""
        with self._lock:
            if position_id in self.position_records:
                self.position_records[position_id].volume -= traded_volume
                if self.position_records[position_id].volume <= 0:
                    if self.strategy:
                        self.strategy.output(f"仓位{position_id} 已完全平仓")
                    # 可选：清除零仓位记录
                    # self.position_records.pop(position_id, None)
    
    def _generate_position_id(self, trade_data: Any) -> str:
        """生成唯一仓位ID"""
        timestamp = int(datetime.now().timestamp() * 1000)
        # 兼容不同的 trade_id 字段
        trade_id = getattr(trade_data, 'trade_id', None) or getattr(trade_data, 'TradeID', None) or str(timestamp)
        return f"{trade_data.instrument_id}_{trade_id}_{timestamp}"
    
    def _load_existing_positions(self) -> None:
        """加载已有持仓"""
        with self._lock:
            try:
                # 这里调用PythonGO API获取实际持仓
                # all_positions = self.strategy.get_all_position()
                # 实际实现时需要根据API返回的数据结构解析
                
                # 模拟示例：加载一个示例持仓
                current_date = datetime.now().date()
                example_position = PositionRecord(
                    position_id="example_existing_position",
                    instrument_id="ag2406",
                    open_price=5000.0,
                    volume=2,
                    direction="0",
                    open_time=datetime.now() - timedelta(days=2),
                    open_date=current_date - timedelta(days=2),
                    position_type=PositionType.OVERNIGHT,  # 非当天开仓
                    stop_profit_price=7500.0,
                    days_held=2
                )
                self.position_records[example_position.position_id] = example_position
                
                if self.strategy:
                    self.strategy.output(f"持仓加载完成，找到{len(self.position_records)}个持仓记录")
            except Exception as e:
                if self.strategy:
                    self.strategy.output(f"加载持仓失败: {e}")
    
    def _cancel_all_pending_orders(self) -> None:
        """取消所有待处理订单"""
        with self._lock:
            for order_id in list(self.active_orders.keys()):
                try:
                    if self.strategy:
                        self.strategy.cancel_order(order_id)
                        self.strategy.output(f"已取消订单 {order_id}")
                except:
                    pass
    
    def _schedule_delayed_close(self, position_id: str, reason: str) -> None:
        """安排延迟平仓"""
        with self._lock:
            # 将延迟平仓任务加入队列，等待下次定时器检查
            # 使用字典存储延迟任务，避免重复添加
            if not hasattr(self, 'delayed_close_tasks'):
                self.delayed_close_tasks = {}
            
            if position_id not in self.delayed_close_tasks:
                self.delayed_close_tasks[position_id] = {
                    "reason": reason,
                    "created_at": datetime.now(),
                    "retry_count": 0
                }
                if self.strategy:
                    self.strategy.output(f"[延迟平仓] {position_id} 已加入延迟队列，原因：{reason}")
    
    def _place_stop_profit_order(self, record: PositionRecord) -> None:
        """放置止盈单（对价方式）"""
        # 止盈触发后使用对价平仓，逻辑已在_check_stop_profit_realtime中实现
        # 这里可以记录止盈单设置，用于监控
        if self.strategy:
            self.strategy.output(f"[止盈设置] {record.instrument_id} | "
                       f"止盈价:{record.stop_profit_price:.2f}")
    
    def get_position_info(self) -> List[Dict]:
        """获取仓位信息 - 规则7"""
        with self._lock:
            result = []
            current_date = datetime.now().date()
            
            for position_id, record in self.position_records.items():
                if record.volume > 0:
                    days_held = (current_date - record.open_date).days
                    
                    result.append({
                        "仓位ID": position_id,
                        "合约": record.instrument_id,
                        "开仓价": f"{record.open_price:.2f}",
                        "持仓量": record.volume,
                        "方向": "多头" if record.direction == "0" else "空头",
                        "性质": record.position_type,
                        "开仓日期": record.open_date.strftime("%Y-%m-%d"),
                        "持仓天数": days_held,
                        "开仓超过3天": days_held >= 3,
                        "止盈价": f"{record.stop_profit_price:.2f}",
                        "追单次数": record.chase_count
                    })
            
            return result
    
    def get_manager_status(self) -> Dict:
        """获取平仓管理器状态"""
        with self._lock:
            # 统计各类型仓位数量
            intraday_count = sum(1 for r in self.position_records.values() 
                               if r.volume > 0 and r.position_type == PositionType.INTRADAY)
            overnight_count = sum(1 for r in self.position_records.values() 
                                if r.volume > 0 and r.position_type == PositionType.OVERNIGHT)
            
            return {
                "活跃仓位数": sum(1 for r in self.position_records.values() if r.volume > 0),
                "日内仓数量": intraday_count,
                "隔夜仓数量": overnight_count,
                "活跃订单数": len(self.active_orders),
                "追单任务数": len(self.chase_tasks),
                "最新行情数": len(self.latest_ticks),
                "当前时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "参数": {
                    "最大追单次数": self.max_chase_attempts,
                    "追单间隔秒数": self.chase_interval_seconds
                }
            }


class Strategy20260105_3(BaseStrategy):
    @property
    def started(self) -> bool:  # type: ignore[override]
        return getattr(self, "my_started", False)

    @started.setter
    def started(self, value: bool) -> None:  # type: ignore[override]
        self.my_started = bool(value)

    @property
    def is_running(self) -> bool:  # type: ignore[override]
        return getattr(self, "my_is_running", False)

    @is_running.setter
    def is_running(self, value: bool) -> None:  # type: ignore[override]
        self.my_is_running = bool(value)

    @property
    def running(self) -> bool:  # type: ignore[override]
        return getattr(self, "my_is_running", False)

    @running.setter
    def running(self, value: bool) -> None:  # type: ignore[override]
        self.my_is_running = bool(value)

    @property
    def is_paused(self) -> bool:  # type: ignore[override]
        return getattr(self, "my_is_paused", False)

    @is_paused.setter
    def is_paused(self, value: bool) -> None:  # type: ignore[override]
        self.my_is_paused = bool(value)

    @property
    def paused(self) -> bool:  # type: ignore[override]
        return getattr(self, "my_is_paused", False)

    @paused.setter
    def paused(self, value: bool) -> None:  # type: ignore[override]
        self.my_is_paused = bool(value)

    @property
    def trading(self) -> bool:  # type: ignore[override]
        return getattr(self, "my_trading", False)

    @trading.setter
    def trading(self, value: bool) -> None:  # type: ignore[override]
        self.my_trading = bool(value)

    @property
    def destroyed(self) -> bool:  # type: ignore[override]
        return getattr(self, "my_destroyed", False)

    @destroyed.setter
    def destroyed(self, value: bool) -> None:  # type: ignore[override]
        self.my_destroyed = bool(value)

    @property
    def state(self) -> str:  # type: ignore[override]
        return getattr(self, "my_state", "")

    @state.setter
    def state(self, value: str) -> None:  # type: ignore[override]
        self.my_state = str(value)

    def _get_trading_sessions(self, ref: Optional[datetime] = None, exchange: Optional[str] = None) -> List[Tuple[datetime, datetime]]:
        """按实盘定义返回日/夜盘时段，可按交易所选择（含上期所跨午夜夜盘）"""
        base_now = ref or datetime.now()
        today = base_now.date()
        yesterday = today - timedelta(days=1)
        exch = (exchange or "").upper()

        sessions: List[Tuple[datetime, datetime]] = []

        # 日盘（按交易所选择，未指定则并集）
        if exch == "CFFEX":
            day_defs = CFFEX_DAY_SESSIONS
        elif exch in ("SHFE", "DCE", "CZCE", "GFEX"):
            day_defs = COMMODITY_DAY_SESSIONS
        else:
            day_defs = CFFEX_DAY_SESSIONS + COMMODITY_DAY_SESSIONS
        for start_t, end_t in day_defs:
            sessions.append((datetime.combine(today, start_t), datetime.combine(today, end_t)))

        # 夜盘（商品交易所）
        include_dce_czce_gfex = exch in ("DCE", "CZCE", "GFEX", "")
        include_shfe = exch in ("SHFE", "")

        if include_dce_czce_gfex:
            sessions.append((datetime.combine(today, NIGHT_START_DCE_CZCE_GFEX), datetime.combine(today, NIGHT_END_DCE_CZCE_GFEX)))
            sessions.append((datetime.combine(yesterday, NIGHT_START_DCE_CZCE_GFEX), datetime.combine(yesterday, NIGHT_END_DCE_CZCE_GFEX)))

        if include_shfe:
            # 当日夜盘 21:00 至次日 02:00，需跨午夜
            sessions.append((datetime.combine(today, NIGHT_START_SHFE), datetime.combine(today + timedelta(days=1), NIGHT_END_SHFE)))
            # 覆盖当日凌晨 00:00-02:00 的前一日夜盘尾段
            sessions.append((datetime.combine(yesterday, NIGHT_START_SHFE), datetime.combine(today, NIGHT_END_SHFE)))

        return sessions

    def is_market_open(self) -> bool:
        """判断当前是否为开盘时间（含日盘和各交易所夜盘）；测试模式下始终返回True"""
        if bool(getattr(self.params, "test_mode", False)):
            return True
        now_dt = datetime.now()
        default_exch = getattr(self.params, "exchange", None)
        for start, end in self._get_trading_sessions(now_dt, default_exch):
            if start <= now_dt <= end:
                return True
        return False

    """
    期权宽度交易信号生成器
    
    严格按照用户要求，不改变核心策略逻辑
    1. 同步移动：期货上涨时看涨虚值期权上涨或者期货下跌时看跌虚值期权上涨
    2. 期货的期权宽度：期货对应的指定月和指定下月虚值期权同步移动的不同行权价品种数量之和
    3. 上涨定义为收盘价大于前一个收盘价
    4. 交易信号判定原则：
        A. 最优原则：期货对应的指定月和指定下月虚值期权全部同步移动为最优原则，取其期权宽度值最大者
        B. 次优原则：期货对应的指定月和指定下月虚值期权全部同步移动其期权宽度值较大优于期货对应的指定月和指定下月虚值期权部分同步移动但其期权宽度值较小者
        C. 期货对应的指定月和指定下月虚值期权都不是全部同步移动，则取期权宽度值最大者
    """
    def __init__(self) -> None:
        """初始化策略"""
        super().__init__()
        # 策略参数
        self.params = Params()

        # 提前注入密钥，确保 MarketCenter/SDK 初始化时凭证已就绪
        try:
            ApiKeyLoader.load_api_key(self.params)
        except Exception:
            pass
        # 标记是否已应用预设，避免重复
        self._profile_applied = False

        # 期货-期权前缀映射（股指）
        self.future_to_option_map = {
            "IF": "IO",
            "IH": "HO",
            "IM": "MO",
        }
        
        # 日志文件路径：需要在其他属性之前设置，避免 output 方法调用失败
        self.log_file_path = getattr(self.params, "log_file_path", "strategy_startup.log") or "strategy_startup.log"
        
        # 基础组件
        self.market_center = MarketCenter()
        # 兼容：部分版本的 MarketCenter.get_instrument_data 签名不同，增加宽容包装
        try:
            original_gid = getattr(self.market_center, "get_instrument_data", None)
            if original_gid:
                def _gid_wrapper(mc_self, exchange=None, instrument_id=None, *args, **kwargs):
                    try:
                        return original_gid(exchange, instrument_id)
                    except TypeError:
                        # 如果底层被无参调用，则尽量用已有参数或直接回退
                        try:
                            return original_gid()
                        except Exception:
                            return None
                self.market_center.get_instrument_data = types.MethodType(_gid_wrapper, self.market_center)
        except Exception:
            pass
        
        # 数据存储
        self.future_instruments = []
        self.option_instruments = {}
        self.future_symbol_to_exchange = {}
        self.kline_data = {}
        self.option_width_results = {}
        # 信号去重/节流
        self.signal_last_emit = {}
        self.signal_cooldown_sec = float(getattr(self.params, "signal_cooldown_sec", 1.0) or 0.0)
        setattr(self.params, "signal_cooldown_sec", self.signal_cooldown_sec)
        # TOP3 输出去重/节流
        self.top3_last_signature = None
        self.top3_last_emit_time = None
        # 交易模式最近一次有效TOP3表格缓存（用于日终汇总）
        self.last_trade_table_lines: List[str] = []
        self.last_trade_table_timestamp: Optional[datetime] = None
        self._insufficient_option_kline_logged = set()
        self._non_option_return_logged = set()
        self._no_option_group_logged = set()

        # 诊断：暂停期间被丢弃的回调计数
        self.paused_drop_counts = {"tick": 0, "kline": 0}
        
        # 期权买方开仓执行器
        self.option_buy_executor = OptionBuyOpenExecutor(strategy_instance=self)

        # 平仓管理器
        self.position_manager = PositionManager(strategy_instance=self)

        # 自动/手动交易标志
        self.auto_trading_enabled = bool(getattr(self.params, "auto_trading_enabled", True))

        # 手动交易次数限制（上午/下午各1次），按日重置
        self.manual_trade_attempts = {"morning": 0, "afternoon": 0}
        self.manual_trade_date = datetime.now().date()

        # 参数表修改频率限制（每月仅允许设置 N 次）
        self.param_edit_month: Optional[str] = None
        self.param_edit_count: int = 0
        self._param_override_cache: Dict[str, Any] = {}

        # 当日交易明细记录（开仓/平仓等），用于日结输出
        self.daily_trade_events: List[Dict[str, Any]] = []
        self._daily_trade_date = datetime.now().date()

        # 标志位
        self.data_loaded = False
        self._auto_starting = False
        self._instruments_ready = False
        self._month_mapping_last_refresh_date = None
        self._option_fetch_failures = set()

        # 兼容BestVersion迁移核心成员变量
        self.calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "last_calculation_time": None,
            "average_calculation_time": 0.0
        }
        self.data_lock = threading.RLock()
        self.option_type_cache = {}
        self._option_type_failed_logged = set()
        self.out_of_money_cache = {}
        self.cache_max_size = 10000
        self.subscribed_instruments = set()  # 记录已订阅的合约，避免重复订阅
        self.history_loaded = False
        self.my_is_running = False
        self.my_is_paused = False
        self.my_trading = False
        self.my_state = "stopped"
        self.my_started = False
        self.kline_insufficient_logged = set()
        self.option_kline_insufficient_logged = set()
        self.zero_price_logged = set()  # 记录已提示过的零价格合约
        self.history_retry_done = False
        self.history_retry_count = 0
        self.my_destroyed = False
        self._ui_running = False
        self._diag_backup = None
        self.log_file_path = getattr(self.params, "log_file_path", "strategy_startup.log") or "strategy_startup.log"
        
        # 定时任务
        self.scheduler = Scheduler("PythonGO")
        try:
            self.calculation_interval = int(getattr(self.params, "calculation_interval", 180) or 180)
        except Exception:
            self.calculation_interval = 180
        # 不在 __init__ 中启动调度器，避免初始化时启动定时任务
        # try:
        #     self.scheduler.start()
        # except Exception:
        #     pass
        
        # 订阅相关
        self.subscription_queue = []  # 订阅队列
        # 来自参数的订阅节流配置
        self.subscription_batch_size = getattr(self.params, "subscription_batch_size", 10)
        self.subscription_interval = getattr(self.params, "subscription_interval", 1)
        self.subscription_backoff_factor = getattr(self.params, "subscription_backoff_factor", 1.0)
        self.subscription_job_ids = set()  # 跟踪订阅批次任务ID，便于清理
        # 文档示例订阅跟踪
        self._doc_demo_instruments = set()

        # 尝试加载 API Key，确保下位SDK 能够访问行情
        self._load_api_key()

        # 交易/回测环境下强制关闭测试/诊断输出
        self._enforce_diagnostic_silence()

        # 若启动即开启调试或强制调试，自动加载参数表（含月映射）
        try:
            if bool(getattr(self.params, "force_debug_on_start", False)) or bool(getattr(self.params, "debug_output", False)):
                setattr(self.params, "debug_output", True)
                self._apply_param_overrides_for_debug()
        except Exception:
            pass

    def _load_api_key(self) -> None:
        """加载/注入 API Key：优先 params，其次环境变量，最后写回环境与SDK。"""
        # 尝试从本地 secrets 文件读取（当 param_table.json 无法持久化时使用）
        try:
            if not hasattr(self, "_load_local_secrets_called"):
                self._load_local_secrets_called = True
        except Exception:
            pass

        try:
            secrets = {}
            try:
                # 本地凭证文件路径：与脚本同目录下的 local_secrets.json
                base_dir = os.path.dirname(os.path.abspath(__file__))
                local_path = os.path.join(base_dir, "local_secrets.json")
                if os.path.exists(local_path):
                    with open(local_path, "r", encoding="utf-8") as _f:
                        try:
                            secrets = json.load(_f) if _f else {}
                        except Exception:
                            secrets = {}
            except Exception:
                secrets = {}
        except Exception:
            secrets = {}
        try:
            # 1) 从参数获取
            key_infini = (getattr(self.params, "infini_api_key", "") or "").strip()
            key_generic = (getattr(self.params, "api_key", "") or "").strip()
            key_access = (getattr(self.params, "access_key", "") or "").strip()
            secret_access = (getattr(self.params, "access_secret", "") or "").strip()

            # 2) 环境兜底
            # 若 params 中为空，则尝试从本地 secrets 文件里读取（优先于环境变量）
            try:
                if not key_infini:
                    key_infini = (secrets.get("infini_api_key") or "").strip()
                if not key_generic:
                    key_generic = (secrets.get("api_key") or "").strip()
                if not key_access:
                    key_access = (secrets.get("access_key") or "").strip()
                if not secret_access:
                    secret_access = (secrets.get("access_secret") or "").strip()
            except Exception:
                pass

            if not key_infini:
                key_infini = (os.getenv("INFINI_API_KEY") or "").strip()
            if not key_generic:
                key_generic = (os.getenv("API_KEY") or "").strip()
            if not key_access:
                key_access = (os.getenv("INFINI_ACCESS_KEY") or os.getenv("ACCESS_KEY") or os.getenv("AccessKey") or "").strip()
            if not secret_access:
                secret_access = (os.getenv("INFINI_ACCESS_SECRET") or os.getenv("ACCESS_SECRET") or os.getenv("AccessSecret") or "").strip()

            # 3) 无密钥提示
            if not key_infini and not key_generic and not key_access and not secret_access:
                self._debug("未找到密钥，请在 params.infini_api_key/api_key/access_key/access_secret 或环境变量 INFINI_* / ACCESS_* / API_KEY 设置")
                return

            # 4) 注入环境与SDK
            try:
                import infini as _infini  # type: ignore[import]
            except Exception:
                _infini = None

            if key_infini:
                os.environ["INFINI_API_KEY"] = key_infini
                try:
                    setattr(_infini, "api_key", key_infini)
                    setattr(_infini, "API_KEY", key_infini)
                except Exception:
                    pass
            if key_generic:
                os.environ["API_KEY"] = key_generic
            if key_access:
                os.environ["INFINI_ACCESS_KEY"] = key_access
                os.environ.setdefault("ACCESS_KEY", key_access)
                try:
                    setattr(_infini, "access_key", key_access)
                    setattr(_infini, "ACCESS_KEY", key_access)
                except Exception:
                    pass
            if secret_access:
                os.environ["INFINI_ACCESS_SECRET"] = secret_access
                os.environ.setdefault("ACCESS_SECRET", secret_access)
                try:
                    setattr(_infini, "access_secret", secret_access)
                    setattr(_infini, "ACCESS_SECRET", secret_access)
                except Exception:
                    pass

            # 额外：如果本地存在 pythongo.infini（占位模块），也把凭证注入到该模块，
            # 以便策略进程中使用本地占位模块时能读取到最新凭证
            try:
                import importlib
                try:
                    _py_inf = importlib.import_module('pythongo.infini')
                except Exception:
                    _py_inf = None
                if _py_inf is not None:
                    try:
                        if key_infini:
                            setattr(_py_inf, 'api_key', key_infini)
                            setattr(_py_inf, 'API_KEY', key_infini)
                        if key_access:
                            setattr(_py_inf, 'access_key', key_access)
                            setattr(_py_inf, 'ACCESS_KEY', key_access)
                        if secret_access:
                            setattr(_py_inf, 'access_secret', secret_access)
                            setattr(_py_inf, 'ACCESS_SECRET', secret_access)
                        # 也直接更新模块变量（保险）
                        try:
                            _py_inf.api_key = key_infini or getattr(_py_inf, 'api_key', None)
                            _py_inf.access_key = key_access or getattr(_py_inf, 'access_key', None)
                            _py_inf.access_secret = secret_access or getattr(_py_inf, 'access_secret', None)
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

            self._debug("API Key 已注入到环境变量和SDK")
        except Exception as e:
            try:
                self.output(f"加载 API Key 失败: {e}")
            except Exception:
                pass

    # 预设配置：一键轻量模式
    def apply_profile(self, name: str) -> None:
        """应用运行预设。目前支持 'lite'"""
        try:
            if name and name.strip().lower() == "lite":
                lite_values = {
                    "subscription_batch_size": 20,
                    "subscription_interval": 1,
                    "subscription_backoff_factor": 1.5,
                    "history_minutes": 120,
                    "enable_doc_examples": False,
                    "pause_unsubscribe_all": True,
                    "auto_load_history": True,
                    # 扩容交易所与品种：覆盖中金所、上期所、郑商所、大商所及已知期权品种
                    "exchanges": "CFFEX,SHFE,DCE,CZCE",
                    "future_products": "IF,IH,IC,RB,CU,AL,ZN,AU,AG,M,Y,A,J,JM,I,CF,SR,MA,TA",
                    "option_products": "IO,HO,MO",
                    "include_future_products_for_options": True,
                }
                for k, v in lite_values.items():
                    try:
                        if hasattr(self.params, k):
                            setattr(self.params, k, v)
                    except Exception:
                        pass
                self._profile_applied = True
                try:
                    self.output("已应用运行预设 lite")
                except Exception:
                    pass
        except Exception:
            # 安静失败，不影响主流逻辑
            pass

    def use_lite_profile(self) -> None:
        """一键切换到 Lite 预设（可在启动前调用）"""
        self.params.run_profile = "lite"
        self.apply_profile("lite")

    def _write_local_log(self, msg: str) -> None:
        """将日志追加写入本地文件，用于启动过程记录"""
        try:
            path = getattr(self, "log_file_path", None) or getattr(self.params, "log_file_path", None)
            if not path:
                return
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            # 本地日志失败不影响主流程
            pass

    def _is_backtest_context(self) -> bool:
        """判定是否处于回测模式，兼容多种标志/配置。"""
        try:
            if bool(getattr(self.params, "backtest_tick_mode", False)):
                return True
            rp = str(getattr(self.params, "run_profile", "")).lower()
            if rp in ("backtest", "bt", "backtesting"):
                return True
            # 常见平台字段兜底
            for name in ("is_backtesting", "backtesting", "in_backtesting", "Backtesting"):
                if bool(getattr(self, name, False)):
                    return True
        except Exception:
            pass
        return False

    def _is_trade_context(self) -> bool:
        """判定是否处于交易模式（输出模式或交易标志）。"""
        try:
            mode = str(getattr(self.params, "output_mode", "debug")).lower()
        except Exception:
            mode = "debug"
        try:
            if mode == "trade":
                return True
            if bool(getattr(self, "my_trading", False)):
                return True
        except Exception:
            pass
        return False

    def _diagnostic_output_allowed(self) -> bool:
        """诊断/测试输出是否允许；交易/回测统一关闭。"""
        try:
            if self._is_trade_context() or self._is_backtest_context():
                return False
        except Exception:
            pass
        try:
            return bool(getattr(self.params, "diagnostic_output", True))
        except Exception:
            return False

    def _enforce_diagnostic_silence(self) -> None:
        """在交易/回测场景下强制关闭测试/诊断输出与测试模式，并记录一次开关状态。"""
        try:
            if self._is_trade_context() or self._is_backtest_context():
                if not getattr(self, "_diag_silence_logged", False):
                    try:
                        mode = str(getattr(self.params, "output_mode", "debug")).lower()
                        rp = str(getattr(self.params, "run_profile", "")).lower()
                        dbg = getattr(self.params, "debug_output", None)
                        diag = getattr(self.params, "diagnostic_output", None)
                        test = getattr(self.params, "test_mode", None)
                        self.output(
                            f"[调试] 交易/回测强制静默生效: output_mode={mode}, run_profile={rp}, "
                            f"debug_output->{dbg}, diagnostic_output->{diag}, test_mode->{test}",
                            force=True,
                        )
                    except Exception:
                        pass
                    setattr(self, "_diag_silence_logged", True)
                try:
                    setattr(self.params, "diagnostic_output", False)
                except Exception:
                    pass
                try:
                    setattr(self.params, "debug_output", False)
                except Exception:
                    pass
                try:
                    setattr(self.params, "test_mode", False)
                except Exception:
                    pass
        except Exception:
            pass

    def output(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """输出到平台并写入本地文件；受调试开关控制，交易/强制信息不受限"""
        try:
            mode = str(getattr(self.params, "output_mode", "debug")).lower()
        except Exception:
            mode = "debug"

        is_trade_msg = bool(kwargs.get("trade", False))
        is_force_msg = bool(kwargs.get("force", False))

        # 诊断/测试输出自动识别关键字，可显式传入 diag=True
        is_diag_msg = bool(kwargs.pop("diag", False))
        try:
            if "调试" in msg or "诊断" in msg or "sanity" in msg.lower() or "验证" in msg:
                is_diag_msg = True
        except Exception:
            pass

        diag_enabled = self._diagnostic_output_allowed()
        if is_diag_msg and not diag_enabled and not is_trade_msg and not is_force_msg:
            return

        # 在调试模式下，默认开启调试输出开关
        dbg_enabled = diag_enabled and (bool(getattr(self.params, "debug_output", False)) or mode == "debug")

        # 交易模式下，仅输出交易或强制信息
        if mode == "trade" and not is_trade_msg and not is_force_msg:
            return
        # 非交易模式，若调试开关关闭且非强制，跳过输出
        if not is_trade_msg and not is_force_msg and not dbg_enabled:
            return

        try:
            super().output(msg)
        except Exception:
            pass
        # 本地日志仅在调试开关开启，或交易/强制信息时写入
        if dbg_enabled or is_trade_msg or is_force_msg:
            self._write_local_log(msg)

    def set_output_mode(self, mode: str) -> None:
        try:
            m = str(mode).lower()
            if m not in ("debug", "trade"):
                self.output(f"无效输出模式: {mode}", force=True)
                return
            setattr(self.params, "output_mode", m)
            # 在调试模式下强制开启调试输出（硬编码保障）
            if m == "debug":
                setattr(self.params, "debug_output", True)
                setattr(self.params, "diagnostic_output", True)
            elif m == "trade":
                # 交易模式可按需关闭调试输出（保持显式）
                setattr(self.params, "debug_output", False)
                setattr(self.params, "diagnostic_output", False)
            # 若界面已启动，调度一次样式刷新
            try:
                self._schedule_output_mode_ui_refresh()
            except Exception:
                pass
            self.output(f"输出模式切换为: {m}", force=True)
        except Exception as e:
            self.output(f"切换输出模式失败: {e}", force=True)

    def set_auto_trading_mode(self, auto: bool) -> None:
        """切换自动/手动交易标志；手动模式下停止自动交易逻辑。"""
        try:
            # 日重置手动计数
            self._reset_manual_limits_if_new_day()
            target_auto = bool(auto)

            # 若切换到手动交易，检查上午/下午次数限制（默认各1次，可由参数控制）
            if not target_auto:
                session_key = self._current_session_half()
                limit_per_half = max(1, int(getattr(self.params, "manual_trade_limit_per_half", 1) or 1))
                if session_key:
                    with self._lock:
                        if self.manual_trade_attempts.get(session_key, 0) >= limit_per_half:
                            self.output("手动交易已经超额！", force=True)
                            return
                        self.manual_trade_attempts[session_key] = self.manual_trade_attempts.get(session_key, 0) + 1

            self.auto_trading_enabled = target_auto
            setattr(self.params, "auto_trading_enabled", self.auto_trading_enabled)
            if self.auto_trading_enabled:
                self.output("已切换为自动交易模式，自动交易逻辑启用", force=True)
            else:
                self.output("已切换为手动交易模式，自动交易逻辑停止", force=True)
            try:
                self._schedule_output_mode_ui_refresh()
            except Exception:
                pass
        except Exception as e:
            self.output(f"切换自动/手动交易模式失败: {e}", force=True)

    def _reset_daily_trades_if_new_day(self) -> None:
        """跨天重置交易明细容器。"""
        try:
            today = datetime.now().date()
            if getattr(self, "_daily_trade_date", None) != today:
                self.daily_trade_events = []
                self._daily_trade_date = today
        except Exception:
            pass

    def _reset_manual_limits_if_new_day(self) -> None:
        """跨天重置手动交易次数限制。"""
        try:
            today = datetime.now().date()
            if getattr(self, "manual_trade_date", None) != today:
                with self._lock:
                    self.manual_trade_attempts = {"morning": 0, "afternoon": 0}
                self.manual_trade_date = today
        except Exception:
            pass

    def _current_session_half(self) -> Optional[str]:
        """返回当前是上午或下午时段，用于手动交易限次。"""
        try:
            now = datetime.now()
            try:
                split = int(getattr(self.params, "morning_afternoon_split_hour", 12) or 12)
            except Exception:
                split = 12
            return "morning" if now.hour < split else "afternoon"
        except Exception:
            return None

    def _resolve_param_table_path(self) -> str:
        """解析参数表绝对路径；优先用户配置，相对路径以脚本目录为基准，最终回退同目录 param_table.json。"""
        try:
            raw = getattr(self.params, "param_override_table", "") or ""
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = []
            if isinstance(raw, str) and raw.strip():
                if os.path.isabs(raw):
                    candidates.append(raw)
                else:
                    candidates.append(os.path.join(base_dir, raw))
            candidates.append(os.path.join(base_dir, "param_table.json"))
            # 优先返回存在的文件，其次回退首个候选，避免指向不存在路径
            for p in candidates:
                if p and os.path.exists(p):
                    return p
            return candidates[0] if candidates else ""
        except Exception:
            return ""

    def _reset_param_edit_quota_if_new_month(self) -> None:
        """跨月重置参数表修改次数。"""
        try:
            current_month = datetime.now().strftime("%Y-%m")
            if getattr(self, "param_edit_month", None) != current_month:
                self.param_edit_month = current_month
                self.param_edit_count = 0
        except Exception:
            pass

    def _load_param_table(self) -> Dict[str, Any]:
        """从参数表字段解析JSON，失败返回空表。"""
        try:
            path = self._resolve_param_table_path()
            self.output(f"[调试] 参数表路径: {path}")
            self.output(f"[调试] 参数表文件是否存在: {path and os.path.exists(path)}")
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.output(f"[调试] 参数表加载成功，共{len(data)}个键")
                        return data
                except Exception as e:
                    self.output(f"[调试] 参数表JSON解析失败: {e}")
                    pass
            else:
                self.output(f"[调试] 参数表文件不存在，使用空表")
            
            raw = getattr(self.params, "param_override_table", "") or ""
            self.output(f"[调试] 参数表原始值: {raw[:100] if len(raw) > 100 else raw}")
            if isinstance(raw, dict):
                self.output(f"[调试] 参数表是dict类型，共{len(raw)}个键")
                return dict(raw)
            if isinstance(raw, str) and raw.strip():
                try:
                    data = json.loads(raw)
                    self.output(f"[调试] 参数表字符串解析成功，共{len(data)}个键")
                    return data
                except Exception as e:
                    self.output(f"[调试] 参数表字符串解析失败: {e}")
                    pass
            self.output(f"[调试] 参数表为空，返回空字典")
            return {}
        except Exception as e:
            self.output(f"[调试] 参数表加载异常: {e}")
            return {}

    def _apply_param_overrides_for_debug(self) -> None:
        """调试模式下根据开关加载参数表；未启用则保持硬编码。"""
        try:
            self.output(f"[调试] 开始应用参数表覆盖")
            overrides = self._param_override_cache or self._load_param_table()
            self.output(f"[调试] 参数表加载结果: overrides类型={type(overrides)}, 键数量={len(overrides) if isinstance(overrides, dict) else 0}")

            # 允许在参数表的 switches 中声明是否启用覆盖；若未提供则回退到 params 字段
            switches = {}
            if isinstance(overrides, dict):
                switches = overrides.get("switches", {}) if isinstance(overrides.get("switches", {}), dict) else {}
                self.output(f"[调试] 参数表switches: {list(switches.keys())}")

            use_override_flag = bool(getattr(self.params, "use_param_overrides_in_debug", False))
            self.output(f"[调试] use_param_overrides_in_debug参数: {use_override_flag}")
            if not use_override_flag:
                use_override_flag = bool(switches.get("use_param_overrides_in_debug", False))
                self.output(f"[调试] switches中的use_param_overrides_in_debug: {use_override_flag}")

            if not use_override_flag:
                self.output("调试模式使用硬编码参数", force=True)
                return

            if not overrides:
                self.output("参数表为空，调试模式继续使用硬编码", force=True)
                return

            # 支持两种结构：扁平（直接是键值对）与嵌套（{params:{...}}）
            param_map = overrides.get("params") if isinstance(overrides, dict) else None
            if not isinstance(param_map, dict):
                param_map = overrides if isinstance(overrides, dict) else {}

            applied = 0
            for k, v in param_map.items():
                if hasattr(self.params, k):
                    try:
                        setattr(self.params, k, v)
                        applied += 1
                    except Exception:
                        pass
            # month_mapping 可能位于 params 或 original_defaults，显式兜底注入
            if hasattr(self.params, "month_mapping"):
                mm_override = None
                if isinstance(overrides, dict):
                    mm_override = overrides.get("month_mapping")
                    if isinstance(overrides.get("params"), dict) and overrides["params"].get("month_mapping") is not None:
                        mm_override = overrides["params"].get("month_mapping")
                    if isinstance(overrides.get("original_defaults"), dict) and overrides["original_defaults"].get("month_mapping") is not None:
                        mm_override = overrides["original_defaults"].get("month_mapping")
                if mm_override is not None:
                    try:
                        setattr(self.params, "month_mapping", mm_override)
                    except Exception:
                        pass
            self._param_override_cache = overrides
            self.output(f"调试模式已应用参数表（{applied} 项）", force=True)
        except Exception as e:
            self.output(f"调试参数表应用失败: {e}", force=True)

    def _on_param_modify_click(self) -> None:
        """参数按钮：若未超额则进入参数表编辑；超额则提示。"""
        try:
            self._reset_param_edit_quota_if_new_month()
            limit = max(1, int(getattr(self.params, "param_edit_limit_per_month", 1) or 1))
            if self.param_edit_count >= limit:
                self.output("已经超额！", force=True)
                return
            self._open_param_editor()
        except Exception as e:
            self.output(f"参数表操作失败: {e}", force=True)

    def _on_backtest_click(self) -> None:
        """打开回测参数编辑器，支持保存/放弃并回写参数表。"""
        try:
            self._open_backtest_editor()
        except Exception as e:
            self.output(f"回测参数操作失败: {e}", force=True)

    def _get_current_param_json_string(self) -> str:
        try:
            path = self._resolve_param_table_path()
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except Exception:
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception:
                        pass

            raw = getattr(self.params, "param_override_table", "") or ""
            if isinstance(raw, dict):
                return json.dumps(raw, ensure_ascii=False, indent=2)
            if isinstance(raw, str) and raw.strip().startswith("{"):
                try:
                    data = json.loads(raw)
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except Exception:
                    return raw
            return "{}"
        except Exception:
            return "{}"

    def _open_param_editor(self) -> None:
        try:
            import tkinter as tk
            from tkinter import messagebox
        except Exception:
            self.output("参数编辑器依赖tkinter，不可用", force=True)
            return

        try:
            top = tk.Toplevel(self._ui_root) if hasattr(self, "_ui_root") and self._ui_root else tk.Tk()
            top.title("参数表编辑")
            top.geometry("600x500")

            frm = tk.Frame(top)
            frm.pack(fill="both", expand=True, padx=10, pady=10)

            lbl = tk.Label(frm, text="编辑参数表(JSON)：")
            lbl.pack(anchor="w")

            tk.Label(
                frm,
                justify="left",
                wraplength=560,
                fg="#444",
                text=(
                    "常用字段注释：\n"
                    "- exchange/future_product/option_product：默认交易所与品种。\n"
                    "- load_history_options：加载历史时是否包含期权。\n"
                    "- subscribe_only_specified_month_*：仅订阅指定月/下月合约。\n"
                    "- option_buy_lots_min/max：期权开仓手数上下限。\n"
                    "- option_contract_multiplier：期权合约乘数。\n"
                    "- position_limit_*：开仓资金限额设置（最大小时数、默认小时、比例上限、最小金额）。\n"
                    "- option_order_*：开仓委托 CTP 参数。\n"
                    "- close_*：平仓/追单/止盈/时间窗口参数（见回测参数区更详细说明）。"
                ),
            ).pack(anchor="w", pady=(0, 4))

            txt = tk.Text(frm, wrap="none", height=25)
            txt.pack(fill="both", expand=False)
            txt.insert("1.0", self._get_current_param_json_string())

            btn_bar = tk.Frame(frm)
            btn_bar.pack(fill="x", pady=(8,0))

            def do_cancel():
                try:
                    top.destroy()
                except Exception:
                    pass

            def do_save():
                try:
                    # 再次校验配额
                    self._reset_param_edit_quota_if_new_month()
                    limit = max(1, int(getattr(self.params, "param_edit_limit_per_month", 1) or 1))
                    if self.param_edit_count >= limit:
                        try:
                            messagebox.showinfo("提示", "已经超额！")
                        except Exception:
                            pass
                        self.output("已经超额！", force=True)
                        return

                    content = txt.get("1.0", "end").strip()
                    try:
                        data = json.loads(content) if content else {}
                    except Exception as e:
                        try:
                            messagebox.showerror("错误", f"JSON 解析失败: {e}")
                        except Exception:
                            pass
                        return

                    # 保存到文件或参数
                    target = self._resolve_param_table_path()
                    saved_to_file = False
                    if isinstance(target, str) and target.strip():
                        try:
                            os.makedirs(os.path.dirname(target), exist_ok=True)
                            with open(target, "w", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)
                            saved_to_file = True
                        except Exception as e:
                            self.output(f"参数表写文件失败: {e}", force=True)

                    if not saved_to_file:
                        # 回写到参数字段为JSON字符串
                        try:
                            setattr(self.params, "param_override_table", json.dumps(data, ensure_ascii=False))
                        except Exception:
                            pass

                    # 更新缓存与计数
                    self._param_override_cache = data if isinstance(data, dict) else {}
                    self.param_edit_count += 1
                    self.output(f"参数表已保存（{len(self._param_override_cache)} 项），本月已用 {self.param_edit_count}/{limit}", force=True)
                    try:
                        messagebox.showinfo("成功", "参数表已保存")
                    except Exception:
                        pass
                    try:
                        top.destroy()
                    except Exception:
                        pass
                except Exception as e:
                    self.output(f"保存参数失败: {e}", force=True)

            btn_save = tk.Button(btn_bar, text="保存", command=do_save)
            btn_save.pack(side="left", padx=(0,8))
            btn_cancel = tk.Button(btn_bar, text="取消", command=do_cancel)
            btn_cancel.pack(side="left")

        except Exception as e:
            self.output(f"参数编辑器打开失败: {e}", force=True)

    def _open_backtest_editor(self) -> None:
        """回测参数编辑：编辑 backtest_params，保存时写回参数表并应用；关闭询问是否保存。"""
        try:
            import tkinter as tk
            from tkinter import messagebox
        except Exception:
            self.output("回测参数编辑器依赖tkinter，不可用", force=True)
            return

        # 载入参数表并备份原始内容，便于放弃时还原
        try:
            table = self._load_param_table()
            if not isinstance(table, dict):
                table = {}
        except Exception:
            table = {}
        try:
            original_table = json.loads(json.dumps(table))  # 深拷贝
        except Exception:
            original_table = table.copy() if isinstance(table, dict) else {}

        # 捕获“回测开始”时的全量参数快照（按参数表路径区分），便于随时恢复
        try:
            current_param_path = None
            try:
                current_param_path = self._resolve_param_table_path()
            except Exception:
                pass

            need_snapshot = not hasattr(self, "_backtest_session_snapshot")
            if not need_snapshot:
                prev_path = getattr(self, "_backtest_session_path", None)
                if current_param_path != prev_path:
                    need_snapshot = True

            if need_snapshot:
                snapshot_copy = json.loads(json.dumps(table)) if isinstance(table, dict) else {}
                self._backtest_session_snapshot = snapshot_copy
                self._backtest_session_path = current_param_path
                self._backtest_session_time = datetime.now()
        except Exception:
            pass

        backtest_params = {}
        try:
            backtest_params = table.get("backtest_params", {}) if isinstance(table, dict) else {}
            if not isinstance(backtest_params, dict):
                backtest_params = {}
        except Exception:
            backtest_params = {}

        # 将开仓/风控参数预填入回测参数区，便于一并调参
        try:
            open_risk_keys = [
                "option_buy_lots_min",
                "option_buy_lots_max",
                "option_contract_multiplier",
                "position_limit_valid_hours_max",
                "position_limit_default_valid_hours",
                "position_limit_max_ratio",
                "position_limit_min_amount",
                "option_order_price_type",
                "option_order_time_condition",
                "option_order_volume_condition",
                "option_order_contingent_condition",
                "option_order_force_close_reason",
                "option_order_hedge_flag",
                "option_order_min_volume",
                "option_order_business_unit",
                "option_order_is_auto_suspend",
                "option_order_user_force_close",
                "option_order_is_swap",
                "close_take_profit_ratio",
                "close_overnight_check_time",
                "close_daycut_time",
                "close_max_hold_days",
                "close_overnight_loss_threshold",
                "close_overnight_profit_threshold",
                "close_max_chase_attempts",
                "close_chase_interval_seconds",
                "close_chase_task_timeout_seconds",
                "close_delayed_timeout_seconds",
                "close_delayed_max_retries",
                "close_order_price_type",
            ]
            for key in open_risk_keys:
                if key not in backtest_params and hasattr(self.params, key):
                    backtest_params[key] = getattr(self.params, key)
        except Exception:
            pass

        def dumps_pretty(obj: Any) -> str:
            try:
                return json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                return "{}"

        try:
            top = tk.Toplevel(self._ui_root) if hasattr(self, "_ui_root") and self._ui_root else tk.Tk()
            top.title("回测参数")
            top.geometry("620x520")

            frm = tk.Frame(top)
            frm.pack(fill="both", expand=True, padx=10, pady=10)

            tk.Label(frm, text="回测参数(JSON，可新增字段)：").pack(anchor="w")
            tk.Label(
                frm,
                justify="left",
                wraplength=580,
                text=(
                    "使用说明：\n"
                    "1) 编辑框内容为 JSON 对象，保存即写回参数表 backtest_params 并应用到当前 params 中已存在的字段。\n"
                    "2) 可新增/修改开仓/风控/平仓参数（已预填 option_* / position_limit_* / close_*）。\n"
                    "3) 不支持 JSON 注释；如需临时关闭某字段，可删除或改为空值。\n"
                    "4) 保存并应用按钮会立即生效；关闭若选择不保存则恢复打开前状态。\n"
                    "5) “恢复回测起点”会回到本次回测开始时捕获的全量参数快照。"
                ),
            ).pack(anchor="w", pady=(0, 4))

            tk.Label(
                frm,
                justify="left",
                wraplength=580,
                fg="#444",
                text=(
                    "字段注释(回测常用)：\n"
                    "- option_buy_lots_min/max：期权开仓手数上下限。\n"
                    "- option_contract_multiplier：期权合约乘数。\n"
                    "- position_limit_*：开仓资金限额（有效小时/默认小时/资金比例上限/最小金额）。\n"
                    "- option_order_*：开仓委托 CTP 参数。\n"
                    "- close_take_profit_ratio：止盈倍数，开仓价*倍数触发。\n"
                    "- close_overnight_check_time / close_daycut_time：隔夜检查、日内平仓时间。\n"
                    "- close_max_hold_days：持仓天数≥阈值时平仓。\n"
                    "- close_overnight_loss_threshold / close_overnight_profit_threshold：隔夜亏损/盈利触发平仓的收益率阈值。\n"
                    "- close_max_chase_attempts / close_chase_interval_seconds：追单最大次数与间隔秒数。\n"
                    "- close_chase_task_timeout_seconds：单次追单任务超时时间。\n"
                    "- close_delayed_timeout_seconds / close_delayed_max_retries：延迟平仓超时与最大重试次数。\n"
                    "- close_order_price_type：平仓委托价类型（CTP代码，默认限价2）。"
                ),
            ).pack(anchor="w", pady=(0, 6))

            # 文本区 + 滚动条，压缩高度以保证底部按钮可见
            text_frame = tk.Frame(frm)
            text_frame.pack(fill="both", expand=True)

            vbar = tk.Scrollbar(text_frame, orient="vertical")
            vbar.pack(side="right", fill="y")
            hbar = tk.Scrollbar(text_frame, orient="horizontal")
            hbar.pack(side="bottom", fill="x")

            txt = tk.Text(text_frame, wrap="none", height=18, yscrollcommand=vbar.set, xscrollcommand=hbar.set)
            txt.pack(side="left", fill="both", expand=True)
            vbar.config(command=txt.yview)
            hbar.config(command=txt.xview)
            txt.insert("1.0", dumps_pretty(backtest_params))

            btn_bar = tk.Frame(frm)
            btn_bar.pack(fill="x", pady=(8,0))

            def apply_and_save(content: str) -> bool:
                """保存到参数表并应用到当前 params，返回是否成功。"""
                try:
                    data = json.loads(content) if content.strip() else {}
                    if not isinstance(data, dict):
                        raise ValueError("回测参数需为JSON对象")

                    # 写回表
                    table_new = table if isinstance(table, dict) else {}
                    table_new["backtest_params"] = data

                    target = self._resolve_param_table_path()
                    saved_to_file = False
                    if isinstance(target, str) and target.strip():
                        try:
                            os.makedirs(os.path.dirname(target), exist_ok=True)
                            with open(target, "w", encoding="utf-8") as f:
                                json.dump(table_new, f, ensure_ascii=False, indent=2)
                            saved_to_file = True
                        except Exception as e:
                            self.output(f"回测参数写文件失败: {e}", force=True)

                    if not saved_to_file:
                        try:
                            setattr(self.params, "param_override_table", json.dumps(table_new, ensure_ascii=False))
                        except Exception:
                            pass

                    # 更新缓存并应用到当前 params（存在的字段才设置）
                    self._param_override_cache = table_new
                    applied = 0
                    for k, v in data.items():
                        if hasattr(self.params, k):
                            try:
                                setattr(self.params, k, v)
                                applied += 1
                            except Exception:
                                pass
                    self.output(f"回测参数已保存，应用 {applied} 项（只对已存在字段）", force=True)
                    return True
                except Exception as e:
                    try:
                        messagebox.showerror("错误", f"保存失败: {e}")
                    except Exception:
                        pass
                    self.output(f"回测参数保存失败: {e}", force=True)
                    return False

            def restore_original():
                """恢复回测参数为打开前状态并回写。"""
                try:
                    target = self._resolve_param_table_path()
                    if isinstance(target, str) and target.strip():
                        try:
                            os.makedirs(os.path.dirname(target), exist_ok=True)
                            with open(target, "w", encoding="utf-8") as f:
                                json.dump(original_table, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            self.output(f"回测参数恢复写文件失败: {e}", force=True)
                    else:
                        try:
                            setattr(self.params, "param_override_table", json.dumps(original_table, ensure_ascii=False))
                        except Exception:
                            pass
                    self._param_override_cache = original_table if isinstance(original_table, dict) else {}
                    # 还原当前 params 中已有的回测字段
                    try:
                        orig_bt = original_table.get("backtest_params", {}) if isinstance(original_table, dict) else {}
                        if isinstance(orig_bt, dict):
                            for k, v in orig_bt.items():
                                if hasattr(self.params, k):
                                    setattr(self.params, k, v)
                    except Exception:
                        pass
                except Exception:
                    pass

            def restore_session_start():
                """恢复到“回测开始”快照并回写。"""
                nonlocal backtest_params
                try:
                    snapshot = getattr(self, "_backtest_session_snapshot", None)
                    if not isinstance(snapshot, dict):
                        try:
                            messagebox.showwarning("提示", "当前进程未捕获回测起点快照")
                        except Exception:
                            pass
                        return

                    snapshot_copy = json.loads(json.dumps(snapshot))
                    target = self._resolve_param_table_path()
                    if isinstance(target, str) and target.strip():
                        try:
                            os.makedirs(os.path.dirname(target), exist_ok=True)
                            with open(target, "w", encoding="utf-8") as f:
                                json.dump(snapshot_copy, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            self.output(f"回测起点恢复写文件失败: {e}", force=True)
                    else:
                        try:
                            setattr(self.params, "param_override_table", json.dumps(snapshot_copy, ensure_ascii=False))
                        except Exception:
                            pass

                    self._param_override_cache = snapshot_copy

                    try:
                        snap_bt = snapshot_copy.get("backtest_params", {}) if isinstance(snapshot_copy, dict) else {}
                        if isinstance(snap_bt, dict):
                            for k, v in snap_bt.items():
                                if hasattr(self.params, k):
                                    setattr(self.params, k, v)
                    except Exception:
                        pass

                    # 刷新编辑框为快照内容，方便继续查看/调整
                    try:
                        snap_bt = snapshot_copy.get("backtest_params", {}) if isinstance(snapshot_copy, dict) else {}
                        txt.delete("1.0", "end")
                        txt.insert("1.0", dumps_pretty(snap_bt if isinstance(snap_bt, dict) else {}))
                        backtest_params = snap_bt if isinstance(snap_bt, dict) else {}
                    except Exception:
                        pass

                    try:
                        messagebox.showinfo("成功", "已恢复到回测起点快照")
                    except Exception:
                        pass
                except Exception as e:
                    self.output(f"回测起点恢复失败: {e}", force=True)

            def do_save():
                content = txt.get("1.0", "end").strip()
                if apply_and_save(content):
                    try:
                        messagebox.showinfo("成功", "已保存并应用")
                    except Exception:
                        pass
                    try:
                        top.destroy()
                    except Exception:
                        pass

            def do_cancel():
                content = txt.get("1.0", "end").strip()
                changed = content.strip() != dumps_pretty(backtest_params).strip()
                if changed:
                    try:
                        res = messagebox.askyesno("保存确认", "是否保存并应用到参数表？选择否将恢复到打开前状态。")
                    except Exception:
                        res = False
                    if res:
                        if apply_and_save(content):
                            try:
                                top.destroy()
                            except Exception:
                                pass
                        return
                    else:
                        restore_original()
                try:
                    top.destroy()
                except Exception:
                    pass

            btn_save = tk.Button(btn_bar, text="保存并应用", command=do_save)
            btn_save.pack(side="left", padx=(0,8))
            btn_restore = tk.Button(btn_bar, text="恢复回测起点", command=restore_session_start)
            btn_restore.pack(side="left", padx=(0,8))
            btn_cancel = tk.Button(btn_bar, text="关闭", command=do_cancel)
            btn_cancel.pack(side="left")

            # 关闭窗口事件：同取消逻辑
            try:
                def _on_close():
                    do_cancel()
                top.protocol("WM_DELETE_WINDOW", _on_close)
            except Exception:
                pass

        except Exception as e:
            self.output(f"回测参数编辑器打开失败: {e}", force=True)

    def record_trade_event(
        self,
        side: str,
        offset: str,
        instrument_id: str,
        exchange: str,
        price: float,
        volume: int,
        account_id: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录当日交易明细，用于日结输出。"""
        try:
            self._reset_daily_trades_if_new_day()
            evt = {
                "ts": datetime.now(),
                "side": side,
                "offset": offset,
                "instrument_id": instrument_id,
                "exchange": exchange,
                "price": float(price),
                "volume": int(volume),
                "account_id": account_id or "",
            }
            if extra and isinstance(extra, dict):
                evt.update(extra)
            self.daily_trade_events.append(evt)
        except Exception:
            pass

    def _start_output_mode_ui(self) -> None:
        """启动简易输出模式界面（调试/交易按钮）"""
        try:
            import threading
            import tkinter as tk
        except Exception:
            self.output("界面库不可用，跳过输出模式界面", force=True)
            return

        cls = self.__class__

        # 遗留窗口自检：若存在全局 root 但标记失真，复用或清理，防止重启时双窗
        try:
            existing_root = getattr(cls, "_ui_global_root", None)
            if existing_root:
                try:
                    if existing_root.winfo_exists():
                        self._ui_root = existing_root
                        self._schedule_bring_output_mode_ui_front()
                        setattr(cls, "_ui_global_running", True)
                        setattr(cls, "_ui_global_creating", False)
                        self.output("检测到遗留输出界面，已复用并置前", force=True)
                        return
                except Exception:
                    pass
                setattr(cls, "_ui_global_root", None)
                setattr(cls, "_ui_global_running", False)
                setattr(cls, "_ui_global_creating", False)
        except Exception:
            pass

        # 进程内单例保护：如已有任何实例在运行，直接置前返回
        try:
            if getattr(cls, "_ui_global_running", False) or getattr(cls, "_ui_global_creating", False):
                try:
                    # 若实例未持有 root，也尝试复用全局 root 聚焦
                    if not getattr(self, "_ui_root", None):
                        self._ui_root = getattr(cls, "_ui_global_root", None)
                    self._schedule_bring_output_mode_ui_front()
                except Exception:
                    pass
                self.output("输出模式界面已在运行（全局），已聚焦现有窗口", force=True)
                return
        except Exception:
            pass

        # 并发保护：如界面创建中或已存在，不重复创建；改为聚焦现有窗口
        try:
            if getattr(self, "_ui_creating", False) or getattr(self, "_ui_running", False) or (hasattr(self, "_ui_root") and self._ui_root):
                try:
                    self._schedule_bring_output_mode_ui_front()
                except Exception:
                    pass
                self.output("输出模式界面已在运行，已聚焦现有窗口", force=True)
                return
        except Exception:
            pass

        def _ui_thread():
            try:
                root = tk.Tk()
                root.title("输出模式控制")
                try:
                    w = int(getattr(self.params, "ui_window_width", 260) or 260)
                    h = int(getattr(self.params, "ui_window_height", 240) or 240)
                except Exception:
                    w, h = 260, 240
                root.geometry(f"{w}x{h}")
                # 顶部状态标签
                lbl = tk.Label(root, text=f"当前模式: {getattr(self.params, 'output_mode', 'debug')}")
                lbl.pack(pady=8)

                # 按钮容器（模式切换）
                btn_frame = tk.Frame(root)
                btn_frame.pack(fill="x", padx=12, pady=6)
                btn_debug = tk.Button(btn_frame, text="调试")
                btn_trade = tk.Button(btn_frame, text="交易")
                btn_backtest_mode = tk.Button(btn_frame, text="回测")
                btn_debug.pack(side="left", expand=True, fill="x", padx=(0, 4))
                btn_trade.pack(side="left", expand=True, fill="x", padx=(4, 4))
                btn_backtest_mode.pack(side="left", expand=True, fill="x", padx=(4, 0))

                # 自动/手动交易按钮
                auto_frame = tk.Frame(root)
                auto_frame.pack(fill="x", padx=12, pady=6)
                btn_auto = tk.Button(auto_frame, text="自动交易")
                btn_manual = tk.Button(auto_frame, text="手动交易")
                btn_auto.pack(side="left", expand=True, fill="x", padx=(0, 6))
                btn_manual.pack(side="left", expand=True, fill="x", padx=(6, 0))

                # 日结输出按钮（手动触发15:01汇总）
                btn_daily = tk.Button(root, text="日结输出 (15:01)")
                btn_daily.pack(fill="x", padx=12, pady=(0, 8))

                # 参数按钮（每月限改次数，载入参数表）
                btn_param = tk.Button(root, text="参数")
                btn_param.pack(fill="x", padx=12, pady=(0, 8))

                # 回测参数按钮
                btn_backtest = tk.Button(root, text="回测参数")
                btn_backtest.pack(fill="x", padx=12, pady=(0, 8))

                # 事件处理：切换模式并刷新样式
                def _to_debug():
                    # 开启调试输出，再应用参数表以便可见提示
                    try:
                        setattr(self.params, "debug_output", True)
                    except Exception:
                        pass
                    # 调试模式：根据开关自动应用参数表或保持硬编码
                    self._apply_param_overrides_for_debug()
                    self.set_output_mode("debug")
                    self._refresh_output_mode_ui_styles()
                def _to_trade():
                    # 关闭调试输出，切换到交易模式
                    try:
                        setattr(self.params, "debug_output", False)
                    except Exception:
                        pass
                    self.set_output_mode("trade")
                    self._refresh_output_mode_ui_styles()

                def _to_backtest_mode():
                    # 回测模式：设置回测档位并静默诊断输出
                    try:
                        setattr(self.params, "run_profile", "backtest")
                        setattr(self.params, "backtest_tick_mode", True)
                        setattr(self.params, "output_mode", "debug")
                        setattr(self.params, "debug_output", False)
                        setattr(self.params, "diagnostic_output", False)
                        self._enforce_diagnostic_silence()
                    except Exception:
                        pass
                    self._refresh_output_mode_ui_styles()

                def _to_auto_trading():
                    self.set_auto_trading_mode(True)
                    self._refresh_output_mode_ui_styles()

                def _to_manual_trading():
                    # 手动模式：停止自动交易逻辑
                    self.set_auto_trading_mode(False)
                    self._refresh_output_mode_ui_styles()

                def _daily_summary():
                    try:
                        # 手动触发日结输出，且不重复调度下一次
                        self._output_daily_signal_summary(skip_reschedule=True)
                    except Exception as e:
                        try:
                            self.output(f"日结输出触发失败: {e}", force=True)
                        except Exception:
                            pass

                def _param_modify():
                    self._on_param_modify_click()

                def _backtest_modify():
                    self._on_backtest_click()

                btn_debug.config(command=_to_debug)
                btn_trade.config(command=_to_trade)
                btn_backtest_mode.config(command=_to_backtest_mode)
                btn_auto.config(command=_to_auto_trading)
                btn_manual.config(command=_to_manual_trading)
                btn_daily.config(command=_daily_summary)
                btn_param.config(command=_param_modify)
                btn_backtest.config(command=_backtest_modify)

                # 记录控件引用以便跨线程刷新
                try:
                    self._ui_root = root
                    self._ui_lbl = lbl
                    self._ui_btn_debug = btn_debug
                    self._ui_btn_trade = btn_trade
                    self._ui_btn_auto = btn_auto
                    self._ui_btn_manual = btn_manual
                    self._ui_btn_daily = btn_daily
                    self._ui_btn_param = btn_param
                    self._ui_btn_backtest_mode = btn_backtest_mode
                    self._ui_btn_backtest = btn_backtest
                    self._ui_running = True
                    self._ui_creating = False
                    self._ui_global_running = True
                    # 进程级单例指针
                    setattr(cls, "_ui_global_root", root)
                    setattr(cls, "_ui_global_running", True)
                    setattr(cls, "_ui_global_creating", False)
                except Exception:
                    pass

                # 初始样式同步当前模式
                self._refresh_output_mode_ui_styles()

                # 窗口关闭时重置运行标记并清理引用
                def _on_close():
                    try:
                        self._ui_running = False
                        self._ui_global_running = False
                        self._ui_root = None
                        self._ui_lbl = None
                        self._ui_btn_debug = None
                        self._ui_btn_trade = None
                        self._ui_btn_auto = btn_auto
                        self._ui_btn_manual = btn_manual
                        self._ui_btn_param = btn_param
                        self._ui_btn_auto = None
                        self._ui_btn_manual = None
                        self._ui_btn_daily = None
                        self._ui_btn_param = None
                        self._ui_btn_backtest_mode = None
                        self._ui_btn_backtest = None
                        # 清空进程级单例指针
                        setattr(cls, "_ui_global_root", None)
                        setattr(cls, "_ui_global_running", False)
                        setattr(cls, "_ui_global_creating", False)
                    except Exception:
                        pass
                    try:
                        root.destroy()
                    except Exception:
                        pass
                try:
                    root.protocol("WM_DELETE_WINDOW", _on_close)
                except Exception:
                    pass

                root.mainloop()
            except Exception as e:
                self.output(f"输出模式界面失败: {e}", force=True)
                try:
                    self._ui_creating = False
                except Exception:
                    pass

        try:
            # 标记创建中，避免并发双启动
            self._ui_creating = True
            setattr(cls, "_ui_global_creating", True)
            self._ui_thread = threading.Thread(target=_ui_thread, daemon=True)
            self._ui_thread.start()
            self.output("输出模式界面已启动（调试/交易）", force=True)
        except Exception as e:
            self.output(f"输出模式界面线程启动失败: {e}", force=True)
            try:
                self._ui_creating = False
                setattr(cls, "_ui_global_creating", False)
            except Exception:
                pass

    def _schedule_bring_output_mode_ui_front(self) -> None:
        """将现有UI窗口置前并聚焦。"""
        try:
            root = None
            try:
                root = getattr(self, "_ui_root", None) or getattr(self.__class__, "_ui_global_root", None)
            except Exception:
                root = getattr(self, "_ui_root", None)
            if root:
                try:
                    def _bring():
                        try:
                            root.deiconify()
                            root.lift()
                            root.focus_force()
                        except Exception:
                            pass
                    root.after(0, _bring)
                except Exception:
                    # 如果after不可用，尽力直接调用（可能同线程）
                    try:
                        root.deiconify()
                        root.lift()
                        root.focus_force()
                    except Exception:
                        pass
        except Exception:
            pass

    def _refresh_output_mode_ui_styles(self) -> None:
        """刷新输出模式界面样式与标签文本；在UI线程内调用或使用after调度。"""
        try:
            # 若UI未启动，直接返回
            if not hasattr(self, "_ui_root") or not getattr(self, "_ui_root"):
                return
            import tkinter as tk
            cur = str(getattr(self.params, 'output_mode', 'debug')).lower()
            # 更新标签
            try:
                if hasattr(self, "_ui_lbl") and self._ui_lbl:
                    self._ui_lbl.config(text=f"当前模式: {cur}")
            except Exception:
                pass
            # 更新按钮样式：选中高亮，未选弱化
            try:
                try:
                    f_lg = int(getattr(self.params, "ui_font_large", 11) or 11)
                    f_sm = int(getattr(self.params, "ui_font_small", 10) or 10)
                except Exception:
                    f_lg, f_sm = 11, 10
                if cur == 'debug':
                    if hasattr(self, "_ui_btn_debug") and self._ui_btn_debug:
                        self._ui_btn_debug.config(relief=tk.SUNKEN, bg="#2e7d32", fg="white", activebackground="#1b5e20", activeforeground="white", font=("Microsoft YaHei", f_lg, "bold"))
                    if hasattr(self, "_ui_btn_trade") and self._ui_btn_trade:
                        self._ui_btn_trade.config(relief=tk.RAISED, bg="#f0f0f0", fg="black", activebackground="#d9d9d9", activeforeground="black", font=("Microsoft YaHei", f_sm))
                    if hasattr(self, "_ui_btn_backtest_mode") and self._ui_btn_backtest_mode:
                        self._ui_btn_backtest_mode.config(relief=tk.RAISED, bg="#f0f0f0", fg="black", activebackground="#d9d9d9", activeforeground="black", font=("Microsoft YaHei", f_sm))
                else:
                    if hasattr(self, "_ui_btn_trade") and self._ui_btn_trade:
                        self._ui_btn_trade.config(relief=tk.SUNKEN, bg="#2e7d32", fg="white", activebackground="#1b5e20", activeforeground="white", font=("Microsoft YaHei", f_lg, "bold"))
                    if hasattr(self, "_ui_btn_debug") and self._ui_btn_debug:
                        self._ui_btn_debug.config(relief=tk.RAISED, bg="#f0f0f0", fg="black", activebackground="#d9d9d9", activeforeground="black", font=("Microsoft YaHei", f_sm))
                    if hasattr(self, "_ui_btn_backtest_mode") and self._ui_btn_backtest_mode:
                        self._ui_btn_backtest_mode.config(relief=tk.RAISED, bg="#f0f0f0", fg="black", activebackground="#d9d9d9", activeforeground="black", font=("Microsoft YaHei", f_sm))

                # 回测模式按钮样式：基于 run_profile/backtest_tick_mode
                is_backtest = False
                try:
                    rp = str(getattr(self.params, "run_profile", "")).lower()
                    is_backtest = rp in ("backtest", "bt", "backtesting") or bool(getattr(self.params, "backtest_tick_mode", False))
                except Exception:
                    pass
                if hasattr(self, "_ui_btn_backtest_mode") and self._ui_btn_backtest_mode:
                    if is_backtest:
                        self._ui_btn_backtest_mode.config(relief=tk.SUNKEN, bg="#6a1b9a", fg="white", activebackground="#4a148c", activeforeground="white", font=("Microsoft YaHei", f_lg, "bold"))
                    else:
                        self._ui_btn_backtest_mode.config(relief=tk.RAISED, bg="#f0f0f0", fg="black", activebackground="#d9d9d9", activeforeground="black", font=("Microsoft YaHei", f_sm))

                # 自动/手动交易按钮样式
                auto_on = bool(getattr(self, "auto_trading_enabled", True))
                if hasattr(self, "_ui_btn_auto") and self._ui_btn_auto:
                    if auto_on:
                        self._ui_btn_auto.config(relief=tk.SUNKEN, bg="#1565c0", fg="white", activebackground="#0d47a1", activeforeground="white", font=("Microsoft YaHei", f_lg, "bold"))
                    else:
                        self._ui_btn_auto.config(relief=tk.RAISED, bg="#f0f0f0", fg="black", activebackground="#d9d9d9", activeforeground="black", font=("Microsoft YaHei", f_sm))
                if hasattr(self, "_ui_btn_manual") and self._ui_btn_manual:
                    if not auto_on:
                        self._ui_btn_manual.config(relief=tk.SUNKEN, bg="#c62828", fg="white", activebackground="#b71c1c", activeforeground="white", font=("Microsoft YaHei", f_lg, "bold"))
                    else:
                        self._ui_btn_manual.config(relief=tk.RAISED, bg="#f0f0f0", fg="black", activebackground="#d9d9d9", activeforeground="black", font=("Microsoft YaHei", f_sm))
            except Exception:
                pass
        except Exception:
            pass

    def _schedule_output_mode_ui_refresh(self) -> None:
        """将样式刷新调度到UI线程，避免跨线程直接操作Tk控件。"""
        try:
            if hasattr(self, "_ui_root") and self._ui_root:
                try:
                    self._ui_root.after(0, self._refresh_output_mode_ui_styles)
                except Exception:
                    # 如果after不可用则直接尝试刷新（可能在同线程）
                    self._refresh_output_mode_ui_styles()
        except Exception:
            pass

    def _schedule_daily_signal_summary(self) -> None:
        """调度下一次 15:01:00 (北京时间) 的日终信号汇总输出。"""
        try:
            now = datetime.now()
            try:
                hh = int(getattr(self.params, "daily_summary_hour", 15) or 15)
                mm = int(getattr(self.params, "daily_summary_minute", 1) or 1)
            except Exception:
                hh, mm = 15, 1
            target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if now >= target:
                # 已过今天 15:01，则安排到明天
                target = target + timedelta(days=1)
            delay = max(1, int((target - now).total_seconds()))
            self._safe_add_once_job(
                job_id="daily_signal_summary",
                func=self._output_daily_signal_summary,
                delay_seconds=delay,
                kwargs={}
            )
        except Exception as e:
            self.output(f"调度日终信号汇总失败: {e}", force=True)

    def _output_daily_signal_summary(self, skip_reschedule: bool = False) -> None:
        """在 15:01 输出最新的交易信号表（同UI内容），并保存日志；可选择不重调度。"""
        try:
            # 直接使用最近一次缓存的交易模式表格
            lines = getattr(self, "last_trade_table_lines", None) or []
            ts = getattr(self, "last_trade_table_timestamp", None)
            if lines:
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else ""
                self.output("15:01 日结：交易模式TOP信号（沿用最新缓存）", force=True)
                if ts_str:
                    self.output(f"最新信号时间: {ts_str}", force=True)
                for line in lines:
                    self.output(line, force=True)
            else:
                self.output("15:01 日结：当前无交易信号（历史为空）", force=True)

            # 输出当日交易明细（开仓/平仓）
            try:
                self._reset_daily_trades_if_new_day()
                today = datetime.now().date()
                trades = [e for e in self.daily_trade_events if isinstance(e.get("ts"), datetime) and e["ts"].date() == today]
                if trades:
                    self.output("15:01 日结：当日交易明细", force=True)
                    headers = ["时间", "账户", "交易所", "合约", "买卖", "开平", "手数", "价格", "金额"]
                    rows: List[List[str]] = []
                    for evt in trades:
                        tstr = evt["ts"].strftime("%H:%M:%S") if isinstance(evt.get("ts"), datetime) else ""
                        amt = 0.0
                        try:
                            amt = float(evt.get("price", 0.0)) * float(evt.get("volume", 0))
                            cm = evt.get("contract_multiplier")
                            if cm:
                                amt *= float(cm)
                        except Exception:
                            pass
                        rows.append([
                            tstr,
                            str(evt.get("account_id", "")),
                            str(evt.get("exchange", "")),
                            str(evt.get("instrument_id", "")),
                            str(evt.get("side", "")),
                            str(evt.get("offset", "")),
                            str(evt.get("volume", "")),
                            f"{float(evt.get('price', 0.0)):.2f}",
                            f"{amt:.2f}",
                        ])
                    col_widths = []
                    for i in range(len(headers)):
                        max_cell = max([len(r[i]) for r in rows]) if rows else 0
                        col_widths.append(max(len(headers[i]), max_cell))
                    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
                    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
                    table_lines: List[str] = [sep, header_line, sep]
                    for r in rows:
                        row_line = "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
                        table_lines.append(row_line)
                    table_lines.append(sep)
                    for line in table_lines:
                        self.output(line, force=True)
                else:
                    self.output("15:01 日结：当日无交易明细", force=True)
            except Exception as e:
                self.output(f"15:01 日结：交易明细输出失败: {e}", force=True)
        except Exception as e:
            self.output(f"15:01 日结输出失败: {e}", force=True)
        finally:
            # 日结执行后，调度下一次（除非手动触发要求跳过）
            if not skip_reschedule:
                try:
                    self._schedule_daily_signal_summary()
                except Exception:
                    pass

    def _safe_add_interval_job(self, job_id: str, func, seconds: int, kwargs: Optional[dict] = None) -> None:
        """安全添加 interval 定时任务：如同名任务已存在则先移除再添加，避免Job ID 冲突."""
        try:
            if not getattr(self.params, "enable_scheduler", True):
                self._debug(f"跳过定时任务 {job_id}（enable_scheduler=False，回测模式）")
                return
            if not getattr(self, "scheduler", None):
                self._debug(f"跳过定时任务 {job_id}（scheduler 不可用）")
                return
            # 已销毁未运行暂停/交易关闭时不再添加任务，避免残留日志
            if self._is_paused_or_stopped():
                return
            # 先尝试移除已有同名任务（如果不存在会静默忽略）
            try:
                self.scheduler.remove_job(job_id)
            except Exception:
                pass

            # 添加新任务
            self.scheduler.add_job(
                func=func,
                trigger="interval",
                id=job_id,
                seconds=seconds,
                kwargs=kwargs or {}
            )
        except Exception as e:
            self.output(f"添加定时任务失败 {job_id}: {e}")

    def _safe_add_once_job(self, job_id: str, func, delay_seconds: int, kwargs: Optional[dict] = None) -> None:
        """安全添加一次性定时任务（trigger='date'），避免订阅批次重复执行造成阻塞"""
        try:
            if not getattr(self.params, "enable_scheduler", True):
                self._debug(f"跳过一次性任务 {job_id}（enable_scheduler=False，回测模式）")
                return
            if not getattr(self, "scheduler", None):
                self._debug(f"跳过一次性任务 {job_id}（scheduler 不可用）")
                return
            # 已销毁未运行暂停/交易关闭时不再添加任务，避免残留日志
            if self._is_paused_or_stopped():
                return
            try:
                self.scheduler.remove_job(job_id)
            except Exception:
                pass

            run_time = datetime.now() + timedelta(seconds=delay_seconds)
            self.scheduler.add_job(
                func=func,
                trigger="date",
                id=job_id,
                run_date=run_time,
                kwargs=kwargs or {}
            )
            # 记录订阅任务ID，便于后续清理
            self.subscription_job_ids.add(job_id)
        except Exception as e:
            self.output(f"添加一次性任务失败{job_id}: {e}")

    def _remove_job_silent(self, job_id: str) -> None:
        """静默移除任务ID，不抛异常"""
        try:
            self.scheduler.remove_job(job_id)
        except Exception:
            pass

    def _debug(self, msg: str) -> None:
        """受控调试输出，便于排查无信号原因"""
        try:
            if not self._diagnostic_output_allowed():
                return
            debug_output = getattr(self.params, "debug_output", False)
            if debug_output:
                self.output(msg, diag=True)
        except Exception as e:
            self.output(f"调试输出异常: {e}", force=True)

    def _is_paused_or_stopped(self) -> bool:
        """综合判断是否处于暂停/停止态，兼容平台可能使用的不同标志位"""
        try:
            # 本策略自有标志
            if getattr(self, "my_destroyed", False):
                return True
            if getattr(self, "my_is_paused", False):
                return True
            if not getattr(self, "my_is_running", False):
                return True
            if getattr(self, "my_trading", True) is False:
                return True
            # 平台/外部可能的状态字段
            state = str(getattr(self, "my_state", "")).lower()
            if state in ("paused", "stopped", "stopping", "destroyed"):
                return True
            # 其他常见命名兼容（若平台以布尔字段暴露）
            for name in ("Paused", "IsPaused", "isPause", "PauseFlag", "pause_flag"):
                val = getattr(self, name, None)
                if isinstance(val, bool) and val:
                    return True
            # 环境变量开关（紧急兜底）
            if str(os.getenv("STRATEGY_PAUSED", "")).strip() == "1":
                return True
        except Exception:
            pass
        return False

    def _set_strategy_state(self, paused: bool) -> None:
        """统一设置策略的暂停/运行状态，同步更新所有相关标志"""
        try:
            if paused:
                self.my_is_running = False
                self.my_is_paused = True
                self.my_trading = False
                self.my_state = "paused"
            else:
                self.my_is_running = True
                self.my_is_paused = False
                self.my_trading = True
                self.my_state = "running"
                self.my_destroyed = False
        except Exception as e:
            self.output(f"设置策略状态失败 {e}")

    def _to_light_kline(self, bar: Any) -> Any:
        """将任意bar 转为轻量K线对象（仅包含open/high/low/close/volume 属性）
        避免直接构造KLineData 以规避环境构造参数不兼容的问题"""
        try:
            def _get(o: Any, names: list, default: float = 0.0) -> float:
                for n in names:
                    # 兼容对象属性与字典键
                    try:
                        if isinstance(o, dict) and n in o:
                            v = o[n]
                        else:
                            v = getattr(o, n, None)
                        if v not in (None, ""):
                            try:
                                return float(v)
                            except Exception:
                                continue
                    except Exception:
                        continue
                return float(default)

            o = _get(bar, ['open', 'Open', 'o'])
            h = _get(bar, ['high', 'High', 'h'])
            l = _get(bar, ['low', 'Low', 'l'])
            c = _get(bar, ['close', 'Close', 'last', 'last_price', 'LastPrice', 'Last', 'c', 'C', 'price', 'Price'])
            v = _get(bar, ['volume', 'Volume', 'vol', 'Vol', 'v'])

            # 尽量保留原始时间戳，缺失则用当前时间
            ts = None
            ts_candidates = ['datetime', 'DateTime', 'timestamp', 'Timestamp', 'time', 'Time']
            for name in ts_candidates:
                try:
                    val = bar[name] if isinstance(bar, dict) and name in bar else getattr(bar, name, None)
                except Exception:
                    val = None
                if val is None:
                    continue
                if isinstance(val, datetime):
                    ts = val
                    break
                if isinstance(val, (int, float)):
                    try:
                        ts = datetime.fromtimestamp(val)
                        break
                    except Exception:
                        pass
                if isinstance(val, str):
                    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y%m%d %H:%M:%S"):
                        try:
                            ts = datetime.fromisoformat(val) if fmt is None else datetime.strptime(val, fmt)
                            break
                        except Exception:
                            continue
                    if ts:
                        break

            if ts is None:
                ts = datetime.now()

            # 若close缺失但其他价有效，用open或high/low兜底
            if c <= 0:
                if o > 0:
                    c = o
                elif max(h, l) > 0:
                    c = max(h, l)

            return types.SimpleNamespace(open=o, high=h, low=l, close=c, volume=v, datetime=ts)
        except Exception:
            return types.SimpleNamespace(open=0, high=0, low=0, close=0, volume=0, datetime=datetime.now())
    
    def on_init(self, *args, **kwargs) -> None:
        """策略初始化回调"""
        try:
            super().on_init(*args, **kwargs)
            self.output("期权宽度信号生成器初始化")
            self.my_state = "initialized"
            self.my_started = False
            self.my_is_running = False
            self.my_is_paused = False
            self.my_destroyed = False
            self.my_trading = False
            self._log_status_snapshot("after on_init")
        except Exception as exc:
            self.output(f"[严重] 策略初始化失败: {exc}")
            raise

        # 避免卡在初始化：移除任何自触发 on_start 的逻辑，保持生命周期由平台驱动


    def on_start(self, *args: Any, **kwargs: Any) -> None:
        """策略启动 - 平台入口点"""
        try:
            super().on_start(*args, **kwargs)
            self.output("[调试] on_start() 方法开始执行")
            self._instruments_ready = False

            # 将诊断放入后台线程，避免阻塞 on_start 导致平台短暂卡死
            def _run_start_diagnostics() -> None:
                try:
                    self.output("=== 开始诊断数据通路 ===")
                    try:
                        sanity = infini.get_instruments_by_product(exchange="CFFEX", product_id="IF")
                        count = len(sanity) if sanity else 0
                        self.output(f"[sanity] infini返回 {count} 条 IF 合约", force=True)
                    except Exception as exc:
                        self.output(f"[sanity] infini拉取失败: {exc}", force=True)

                    try:
                        mc_sample = self.market_center.get_instruments(exchange="CFFEX")
                        count = len(mc_sample) if mc_sample else 0
                        self.output(f"[sanity] MarketCenter 返回 {count} 条 CFFEX 合约", force=True)
                    except AttributeError as exc:
                        self.output(f"[sanity] MarketCenter 方法不存在: {exc}. 正在检查可用方法...", force=True)
                        try:
                            attrs = [name for name in dir(self.market_center) if not name.startswith('_')]
                            self.output(f"[sanity] MarketCenter 对象属性: {attrs}", force=True)
                        except Exception as inner_exc:
                            self.output(f"[sanity] 无法列出 MarketCenter 属性: {inner_exc}", force=True)
                    except Exception as exc:
                        self.output(f"[sanity] MarketCenter 拉取失败: {exc}", force=True)

                    try:
                        # [Fix] 使用底层 MarketCenter 获取合约 sanity check
                        probe = None
                        if getattr(self, "market_center", None) and hasattr(self.market_center, "get_instrument_data"):
                            probe = self.market_center.get_instrument_data(exchange="SHFE", instrument_id="ag2406")
                        
                        status = "成功" if probe else "返回空"
                        self.output(f"[sanity] MarketCenter.get_instrument_data 调用: {status}", force=True)
                    except Exception as exc:
                        self.output(f"[sanity] MarketCenter.get_instrument_data 调用异常: {exc}", force=True)

                    self.output("=== 诊断结束 ===")
                except Exception as exc:
                    try:
                        self.output(f"[sanity] 诊断线程异常: {exc}", force=True)
                    except Exception:
                        pass

            threading.Thread(target=_run_start_diagnostics, daemon=True).start()
            self.output("[调试] 诊断任务已在后台线程启动，on_start 将继续返回")

            self.my_started = True
            self.my_is_running = True
            self.my_state = "running"
            self.my_trading = True
            self.my_is_paused = False
            self.my_destroyed = False

            try:
                self._set_strategy_state(paused=False)
            except Exception:
                self.my_is_running = True
                self.my_is_paused = False
                self.my_trading = True
                self.my_state = "running"
                self.my_destroyed = False

            self.output(f"[调试] 我的业务状态已设置: started={self.my_started}, trading={self.my_trading}")

            # 启动输出模式界面（调试/交易 + 日结按钮），仅启动一次
            try:
                if bool(getattr(self.params, "enable_output_mode_ui", True)) and not (getattr(self, "_ui_running", False) or getattr(self, "_ui_creating", False)):
                    self._start_output_mode_ui()
            except Exception as exc:
                try:
                    self.output(f"输出模式界面启动失败: {exc}", force=True)
                except Exception:
                    pass

            # 启动平仓管理器
            try:
                if hasattr(self, "position_manager") and self.position_manager:
                    self.position_manager.start()
                    self.output("平仓管理器已启动", force=True)
            except Exception as exc:
                try:
                    self.output(f"平仓管理器启动失败: {exc}", force=True)
                except Exception:
                    pass

            # 诊断：输出当前实例的关键状态属性
            self.output("=== on_start - 状态快照 ===")
            self._log_status_snapshot("on_start")

            # 在单独的线程中执行 start() 方法，避免阻塞 on_start
            self.output("[调试] on_start() - 即将在后台线程中调用 start() 方法")

            def _start_in_thread() -> None:
                try:
                    self.start()
                    self.my_state = "running"
                    self.my_trading = True
                    self.output("[调试] start() 方法在后台线程中执行完成")
                except Exception as exc:
                    self.output(f"start() 方法在后台线程中执行失败: {exc}\n{traceback.format_exc()}")
                    self.my_state = "error"
                    self.my_is_running = False
                    self.my_trading = False

            threading.Thread(target=_start_in_thread, daemon=True).start()
            self.output("[调试] on_start() - 已在后台线程中启动 start() 方法，on_start 即将返回")

            self.output("=== 2026-01-09 23:50 修改版本已加载 ===")
            self.output("=== on_start 执行完成，策略应已启动 ===")

            try:
                self._load_api_key()
            except Exception as exc:
                self.output(f"加载 API Key 失败: {exc}")
        except Exception as exc:
            self.my_state = "error"
            self.my_is_running = False
            self.my_trading = False
            try:
                self.output(f"[严重] on_start() 执行失败: {exc}\n{traceback.format_exc()}")
            except Exception:
                pass
            raise

    def _load_external_config(self) -> None:
        """从本地配置文件加载参数覆盖值，避免每次修改程序
        支持文件：demo/config/strategy_settings.json 或Strategy20260105_9.json
        仅覆盖已有的 Params 字段"""
        try:
            base_dir = os.path.dirname(__file__)
            cfg_dir = os.path.join(base_dir, "config")
            candidates = [
                os.path.join(cfg_dir, "strategy_settings.json"),
                os.path.join(cfg_dir, "Strategy20260105_9.json"),
            ]
            cfg_path = next((p for p in candidates if os.path.isfile(p)), None)
            if not cfg_path:
                return

            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return

            applied: Dict[str, Any] = {}
            # 正常键覆盖
            for k, v in data.items():
                if hasattr(self.params, k):
                    try:
                        setattr(self.params, k, v)
                        applied[k] = v
                    except Exception:
                        pass
            # 特殊键名映射（JSON里使用环境变量风格）
            if "INFINI_API_KEY" in data and data["INFINI_API_KEY"]:
                try:
                    self.params.infini_api_key = data["INFINI_API_KEY"]
                    applied["infini_api_key"] = data["INFINI_API_KEY"]
                except Exception:
                    pass
            if "API_KEY" in data and data["API_KEY"]:
                try:
                    self.params.api_key = data["API_KEY"]
                    applied["api_key"] = data["API_KEY"]
                except Exception:
                    pass
            # AccessKey/AccessSecret 映射
            for key_name in ("INFINI_ACCESS_KEY", "ACCESS_KEY", "AccessKey"):
                if key_name in data and data[key_name]:
                    try:
                        self.params.access_key = data[key_name]
                        applied["access_key"] = data[key_name]
                        break
                    except Exception:
                        pass
            for key_name in ("INFINI_ACCESS_SECRET", "ACCESS_SECRET", "AccessSecret"):
                if key_name in data and data[key_name]:
                    try:
                        self.params.access_secret = data[key_name]
                        applied["access_secret"] = data[key_name]
                        break
                    except Exception:
                        pass

            if applied:
                self.output(f"已加载本地配置覆盖项: {sorted(applied.keys())}")
        except Exception as e:
            self.output(f"加载本地配置失败: {e}")

    def on_pause(self, *args: Any, **kwargs: Any) -> None:
        """策略暂停（平台按键触发）"""
        try:
            self.output("=" * 60)
            self.output("【调试】on_pause 方法被调用")
            self.output(f"【调试】调用参数 args={args}, kwargs={kwargs}")
            self.output(
                f"【调试】调用前状态 my_is_running={self.my_is_running} my_is_paused={self.my_is_paused} "
                f"my_trading={self.my_trading} my_destroyed={self.my_destroyed}"
            )
            # 先调用父类方法，确保平台底层框架接收暂停信号
            try:
                self.output("【调试】准备调用 super().on_pause(*args, **kwargs)")
                super().on_pause(*args, **kwargs)
                self.output("【调试】super().on_pause(*args, **kwargs) 调用完成")
            except Exception as e:
                self.output(f"【调试】super().on_pause(*args, **kwargs) 调用失败: {e}")
            # 然后执行策略自身的暂停逻辑
            self.output("【调试】准备调用 self.pause_strategy()")
            self.pause_strategy()
            self.output(
                f"【调试】调用后状态 my_is_running={self.my_is_running} my_is_paused={self.my_is_paused} "
                f"my_trading={self.my_trading} my_destroyed={self.my_destroyed}"
            )
            self.output("=" * 60)
        except Exception as e:
            self.output(f"策略暂停失败: {e}\n{traceback.format_exc()}")

    def on_resume(self, *args: Any, **kwargs: Any) -> None:
        """策略恢复（平台按键触发）"""
        try:
            self.output("=" * 60)
            self.output("【调试】on_resume 方法被调用")
            self.output(f"【调试】调用参数 args={args}, kwargs={kwargs}")
            self.output(
                f"【调试】调用前状态 my_is_running={self.my_is_running} my_is_paused={self.my_is_paused} "
                f"my_trading={self.my_trading} my_destroyed={self.my_destroyed}"
            )
            # 先调用父类方法，确保平台底层框架接收恢复信号
            try:
                self.output("【调试】准备调用 super().on_resume(*args, **kwargs)")
                super().on_resume(*args, **kwargs)
                self.output("【调试】super().on_resume(*args, **kwargs) 调用完成")
            except Exception as e:
                self.output(f"【调试】super().on_resume(*args, **kwargs) 调用失败: {e}")
            # 然后执行策略自身的恢复逻辑
            self.output("【调试】准备调用 self.resume_strategy()")
            self.resume_strategy()
            self.output(
                f"【调试】调用后状态 my_is_running={self.my_is_running} my_is_paused={self.my_is_paused} "
                f"my_trading={self.my_trading} my_destroyed={self.my_destroyed}"
            )
            self.output("=" * 60)
        except Exception as e:
            self.output(f"策略恢复失败: {e}\n{traceback.format_exc()}")

    # 兼容平台可能直接调用 pause()/resume() 而非 on_pause/on_resume
    def pause(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.output("收到 pause 调用")
            self.pause_strategy()
            try:
                super().on_pause(*args, **kwargs)
            except Exception:
                pass
        except Exception as e:
            self.output(f"策略暂停失败: {e}\n{traceback.format_exc()}")

    def resume(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.output("收到 resume 调用")
            self.resume_strategy()
            try:
                super().on_resume(*args, **kwargs)
            except Exception:
                pass
        except Exception as e:
            self.output(f"策略恢复失败: {e}\n{traceback.format_exc()}")

    # 兼容其他可能被平台调用的命名
    def pauseStrategy(self, *args: Any, **kwargs: Any) -> None:
        self.pause(*args, **kwargs)

    def resumeStrategy(self, *args: Any, **kwargs: Any) -> None:
        self.resume(*args, **kwargs)

    # 进一步兼容可能的命名变体（部分平台大小写或onX命名）
    def onPause(self, *args: Any, **kwargs: Any) -> None:
        self.pause(*args, **kwargs)

    def onResume(self, *args: Any, **kwargs: Any) -> None:
        self.resume(*args, **kwargs)

    def Pause(self, *args: Any, **kwargs: Any) -> None:
        self.pause(*args, **kwargs)

    def Resume(self, *args: Any, **kwargs: Any) -> None:
        self.resume(*args, **kwargs)

    # 兼容平台可能调用 onStop/Stop（区分暂停与停止由参数控制）
    def onStop(self, *args: Any, **kwargs: Any) -> None:
        self.on_stop(*args, **kwargs)

    def Stop(self, *args: Any, **kwargs: Any) -> None:
        if bool(getattr(self.params, "pause_on_stop", True)):
            self.on_stop(*args, **kwargs)
        else:
            self.stop(*args, **kwargs)

    def onStopStrategy(self, *args: Any, **kwargs: Any) -> None:
        """平台可能调用的变体：将on_stop 映射暂停/停止"""
        self.on_stop(*args, **kwargs)

    # 兼容平台生命周期：将创建/加载/初始化别名映射到 on_init（其内部已自动触发on_start）
    def on_create(self, *args: Any, **kwargs: Any) -> None:
        self.on_init(*args, **kwargs)

    def Create(self, *args: Any, **kwargs: Any) -> None:
        self.on_init(*args, **kwargs)

    def onLoad(self, *args: Any, **kwargs: Any) -> None:
        self.on_init(*args, **kwargs)

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        self.on_init(*args, **kwargs)

    def initStrategy(self, *args: Any, **kwargs: Any) -> None:
        self.on_init(*args, **kwargs)

    # 兼容启动别名：若未启动则调用 on_start
    def onStart(self, *args: Any, **kwargs: Any) -> None:
        self.on_start(*args, **kwargs)

    def Start(self, *args: Any, **kwargs: Any) -> None:
        self.on_start(*args, **kwargs)

    # 兼容平台可能直接调用 startStrategy/stopStrategy
    def startStrategy(self, *args: Any, **kwargs: Any) -> None:
        if not self.my_started:
            self.on_start(*args, **kwargs)

    def stopStrategy(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.stop()
        finally:
            self.my_state = "stopped"
            self.my_is_running = False
            self.my_trading = False

    # 销毁钩子与兼容别名，确保实例删除后彻底停止
    def __del__(self):
        try:
            # 避免重复输出，销毁时只做彻底清理
            self.stop()
            try:
                self.output("实例销毁：已停止调度器并清理所有任务")
            except Exception:
                pass
        except Exception:
            pass

    def on_destroy(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.stop()
        except Exception:
            pass

    def destroyStrategy(self, *args: Any, **kwargs: Any) -> None:
        self.on_destroy(*args, **kwargs)

    def Destroy(self, *args: Any, **kwargs: Any) -> None:
        self.on_destroy(*args, **kwargs)
    
    def start(self) -> None:
        """策略启动逻辑 - 核心业务逻辑"""
        try:
            # --- 关键修改：移除在 start 方法中设置状态标志的代码 ---
            # 状态标志应由 on_start 统一管理

            # 检查是否已经执行过，避免重复执行 (可选，取决于具体需求)
            if getattr(self, "_start_executed", False):
                 self.output("[调试] start() 方法已经执行过，跳过重复执行")
                 return
            # 标记 start() 方法已经执行过 (可选)
            self._start_executed = True

            start_ts = datetime.now()
            self.output(f"=== start 方法开始执行，时间 {start_ts.strftime('%H:%M:%S')} ===")
            self.output(
                f"[调试] start() 内部，当前状态: my_is_running={self.my_is_running}, "
                f"my_state={self.my_state}, my_trading={self.my_trading}"
            )

            output_mode = str(getattr(self.params, "output_mode", "debug")).lower()
            tick_backtest = bool(getattr(self.params, "backtest_tick_mode", False))

            # 调试模式下禁用 Tick 合成，避免调试日志阶段额外工作量
            if output_mode == "debug":
                try:
                    setattr(self.params, "use_tick_kline_generator", False)
                    self._debug("调试模式：关闭 Tick→K线 生成")
                except Exception:
                    pass
            elif tick_backtest and not getattr(self.params, "use_tick_kline_generator", False):
                try:
                    setattr(self.params, "use_tick_kline_generator", True)
                    self._debug("tick回测模式：自动开启 Tick→K线 生成")
                except Exception:
                    pass

            # 启动调度器
            try:
                self.scheduler.start()
                self.output("调度器已启动")
            except Exception as e:
                self.output(f"调度器启动失败 {e}")

            # 调度每日 15:01 日结输出
            try:
                self._schedule_daily_signal_summary()
            except Exception as e:
                self.output(f"日结调度失败: {e}", force=True)

            # 应用预设
            if str(getattr(self.params, "run_profile", "full")).lower() == "lite" and not self._profile_applied:
                self.apply_profile("lite")

            # 诊断参数设置（可选）：仅在 force_debug_on_start=True 时打开
            try:
                self._diag_backup = {
                    "debug_output": getattr(self.params, "debug_output", False),
                    "ignore_otm_filter": getattr(self.params, "ignore_otm_filter", False),
                    "allow_minimal_signal": getattr(self.params, "allow_minimal_signal", False),
                    "min_option_width": getattr(self.params, "min_option_width", None),
                }
                if bool(getattr(self.params, "force_debug_on_start", False)):
                    setattr(self.params, "debug_output", True)
                    self._debug("启动阶段按配置开启调试输出 debug_output=True")
            except Exception:
                pass
            self.output(f"[调试] 启动时future_instruments: {len(self.future_instruments)} option_instruments: {len(self.option_instruments)}")

            # 只有在合约数据未加载时才重新加载
            if not self.data_loaded:
                step_ts = datetime.now()
                self.output(f"[start] 加载合约数据... {step_ts.strftime('%H:%M:%S')}")
                self.load_all_instruments()
                self.output(f"[start] 合约数据加载完成，用时{(datetime.now()-step_ts).total_seconds():.2f}s")
            else:
                self.output("=== 合约数据已加载，跳过 ===")

            self.output(f"[调试] 合约加载时 future_instruments: {len(self.future_instruments)} option_instruments: {len(self.option_instruments)}")

            # 启动时打印合约统计，便于确认订阅对象
            self.output("=== 开始打印合约统计 ===")
            self.output(f"已加载期货 {len(self.future_instruments)} 个")
            self.output(f"期权分组数 {len(self.option_instruments)} 个")
            try:
                option_total = sum(len(v) for v in self.option_instruments.values())
                self.output(f"期权总数: {option_total} 个")
            except Exception as e:
                self.output(f"计算期权总数失败: {e}")
                self.output(f"期权总数: 0 个")

            # 分批订阅
            self.output("=== 开始分批订阅合约 ===")
            self._subscribe_in_batches()
            # 合约与订阅准备完成
            self._instruments_ready = True

            # 加载历史K线 (如果需要)，支持异步以缩短启动耗时
            if tick_backtest:
                self._debug("tick回测模式：跳过历史K线加载，等待Tick实时合成")
            elif getattr(self.params, "auto_load_history", True) and not self.history_loaded:
                if bool(getattr(self.params, "async_history_load", True)):
                    self.output("=== 异步加载历史K线（不阻塞启动） ===")
                    import threading as _th
                    def _load_hist_async():
                        hist_ts = datetime.now()
                        try:
                            self.load_historical_klines()
                            self.history_loaded = True
                            self.output(f"=== 异步历史K线加载完成，用时 {(datetime.now()-hist_ts).total_seconds():.2f}s ===")
                        except Exception as e:
                            self.output(f"异步历史K线加载失败 {e}")
                    _th.Thread(target=_load_hist_async, daemon=True).start()
                else:
                    hist_ts = datetime.now()
                    self.output("=== 开始加载历史K线数据（同步） ===")
                    try:
                        self.load_historical_klines()
                        self.history_loaded = True
                        self.output(f"=== 历史K线加载完成，用时 {(datetime.now()-hist_ts).total_seconds():.2f}s ===")
                    except Exception as e:
                        self.output(f"历史K线加载失败 {e}")

            # 启动阶段打印K线与准备度快照（可选）
            if bool(getattr(self.params, "print_start_snapshots", False)):
                try:
                    snap_ts = datetime.now()
                    self.output("=== 开始打印K 线快照 ===")
                    self.print_kline_counts(limit=10)
                    self.print_commodity_option_readiness(limit=10)
                    self.output(f"=== K 线快照打印结束，用时 {(datetime.now()-snap_ts).total_seconds():.2f}s ===")
                except Exception as e:
                    self.output(f"打印 K 线快照失败 {e}")

            # 启动定时任务（回测可禁用）
            if getattr(self.params, "enable_scheduler", True):
                self.output("=== 开始添加定时任务 ===")
                self._safe_add_interval_job(
                    job_id="calculate_all_option_widths",
                    func=self.calculate_all_option_widths,
                    seconds=self.calculation_interval
                )
                self.output(f"已启动定时任务，每隔 {self.calculation_interval} 秒计算一次期权宽度")
            else:
                self._debug("跳过定时任务（enable_scheduler=False，回测模式）")

            # 立即计算一次
            self.output(f"[调试] 准备调用 calculate_all_option_widths")
            try:
                self.calculate_all_option_widths()
                self.output(f"[调试] calculate_all_option_widths（首次立即计算）调用完成")
            except Exception as e:
                self._debug(f"首次立即计算失败: {e}")

            self.output("=== start 方法执行完成 ===")
            self.output(f"=== start 总耗时 {(datetime.now()-start_ts).total_seconds():.2f}s ===")
            # 不再在 start 方法末尾修改状态

        except Exception as e:
            self.output(f"策略 start 逻辑失败: {e}\n{traceback.format_exc()}")
            # 不在此处修改状态，由 on_start 统一处理错误状态

    def _log_status_snapshot(self, tag: str) -> None:
        """输出当前状态快照，辅助定位卡在初始化的原因"""
        try:
            snap = {
                "tag": tag,
                "state": getattr(self, "state", None),
                "is_running": getattr(self, "is_running", None),
                "is_paused": getattr(self, "is_paused", None),
                "started": getattr(self, "started", None),
                "trading": getattr(self, "trading", None),
                "destroyed": getattr(self, "destroyed", None),
                "my_state": getattr(self, "my_state", None),
                "my_is_running": getattr(self, "my_is_running", None),
                "my_is_paused": getattr(self, "my_is_paused", None),
                "my_started": getattr(self, "my_started", None),
                "my_trading": getattr(self, "my_trading", None),
                "my_destroyed": getattr(self, "my_destroyed", None),
                "jobs": len(getattr(self, "subscription_job_ids", set()) or []),
            }
            self.output(f"[状态快照] {snap}")
        except Exception:
            pass

    def install_emergency_pause_solution(self) -> None:
        """安装紧急暂停解决方案：通过网络监听等方式实现暂停"""
        try:
            self.output("=" * 80)
            self.output("【紧急暂停解决方案】开始安装")
            self.output("=" * 80)
            
            # 1. 创建紧急暂停文件监控
            self._setup_emergency_file_watch()
            
            # 2. 设置紧急网络监听
            self._setup_emergency_network_listener()
            
            # 3. 创建紧急信号
            self._setup_emergency_signals()
            
            self.output("【紧急暂停解决方案】安装完成")
            self.output("使用方法：")
            self.output("  1. 文件方式：创建emergency_pause.flag 文件暂停，删除恢复")
            self.output("  2. 网络方式：向 127.0.0.1:9999 发送'PAUSE' 暂停，'RESUME' 恢复")
            self.output("=" * 80)
            
        except Exception as e:
            self.output(f"【紧急暂停解决方案】安装失败 {e}\n{traceback.format_exc()}")

    def _setup_emergency_file_watch(self) -> None:
        """通过文件监控实现暂停"""
        try:
            import os
            import time
            import threading
            
            pause_file = os.path.join(os.getcwd(), "emergency_pause.flag")
            
            def check_pause_file():
                while True:
                    try:
                        if os.path.exists(pause_file):
                            if not getattr(self, "_emergency_paused", False):
                                pass
                                self.pause_strategy()
                                self._emergency_paused = True
                        else:
                            if getattr(self, "_emergency_paused", False):
                                pass
                                self.resume_strategy()
                                self._emergency_paused = False
                    except Exception as e:
                        self._debug(f"文件监控失败: {e}")
                    time.sleep(1)
            
            thread = threading.Thread(target=check_pause_file, daemon=True)
            thread.start()
            pass
            
        except Exception as e:
            pass

    def _setup_emergency_network_listener(self) -> None:
        """通过网络监听实现暂停"""
        try:
            import socket
            import threading
            
            def listen_for_pause():
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    sock.bind(('127.0.0.1', 9999))
                    pass
                except Exception as e:
                    return
                    return
                
                while True:
                    try:
                        data, addr = sock.recvfrom(1024)
                        message = data.decode('utf-8').strip()
                        
                        if message == 'PAUSE':
                            pass
                            self.pause_strategy()
                            self._emergency_paused = True
                        elif message == 'RESUME':
                            pass
                            self.resume_strategy()
                            self._emergency_paused = False
                        else:
                            pass
                    except Exception as e:
                        self._debug(f"网络监听失败: {e}")
            
            thread = threading.Thread(target=listen_for_pause, daemon=True)
            thread.start()
            
        except Exception as e:
            pass

    def _setup_emergency_signals(self) -> None:
        """设置紧急信号"""
        try:
            self._emergency_paused = False
        except Exception as e:
            pass
    
    def _subscribe_in_batches(self) -> None:
        """使用定时器分批订阅合约，避免在启动时阻塞主线程"""
        try:
            self.output("=== 开始分批订阅合约 ===")
            start_time = datetime.now()
            # 在构建订阅队列前，先尝试对调试映射进行格式对齐修正
            try:
                self._align_month_mapping_to_loaded_futures()
            except Exception:
                pass
            
            # 构建订阅队列
            self.subscription_queue = []
            
            filter_specified_futures = self._resolve_subscribe_flag(
                "subscribe_only_specified_month_futures",
                "subscribe_only_current_next_futures",
                False
            )
            
            # 使用动态加载的期货合约列表（可选：仅限指定月/指定下月 CFFEX IF/IH/IC）
            fut_included = 0
            fut_skipped = 0
            self.output("[调试] 即将首次调用 calculate_all_option_widths")
            for future in self.future_instruments:
                exchange = future.get("ExchangeID", "")
                instrument_id = future.get("InstrumentID", "")
                if not exchange or not instrument_id:
                    continue
                instrument_norm = self._normalize_future_id(instrument_id)
                # 跳过非月份类综合合约（如 IFMain/IF_Weighted）
                if not self._is_real_month_contract(instrument_norm):
                    fut_skipped += 1
                    continue
                # 当参数开启时，仅订阅指定月和指定下月的期货（适用于所有交易所）
                if filter_specified_futures and (not self._is_symbol_specified_or_next(instrument_norm)):
                    fut_skipped += 1
                    continue
                
                self.subscription_queue.append({
                    "exchange": exchange,
                    "instrument_id": instrument_id,
                    "type": "future"
                })
                fut_included += 1
            
            # 如果需要订阅期权，则添加期权到订阅队列（可选：仅限指定月）
            seen_opt_keys: Set[str] = set()
            included_options = 0
            skipped_options = 0
            if self.params.subscribe_options:
                try:
                    self._normalize_option_group_keys()
                except Exception:
                    pass
                filter_specified_options = self._resolve_subscribe_flag(
                    "subscribe_only_specified_month_options",
                    "subscribe_only_current_next_options",
                    False
                )

                allowed_future_symbols: Set[str] = set()
                if filter_specified_options:
                    # 按日期直接筛选option_instruments 的分组键（不依赖已加载期货）
                    for fid in list(self.option_instruments.keys()):
                        fid_norm = self._normalize_future_id(fid)
                        if self._is_symbol_specified_or_next(fid_norm.upper()):
                            allowed_future_symbols.add(fid_norm.upper())
                    self._debug(f"启用过滤：仅订阅指定月/指定下月期权，允许的分组键数量{len(allowed_future_symbols)}")

                global_seen_opts: Set[str] = set()

                for future_symbol, options in self.option_instruments.items():
                    future_symbol_norm = self._normalize_future_id(future_symbol)
                    if filter_specified_options and future_symbol_norm.upper() not in allowed_future_symbols:
                        skipped_options += len(options)
                        continue
                    for option in options:
                        opt_exchange = option.get("ExchangeID", "")
                        opt_instrument = option.get("InstrumentID", "")
                        if not opt_exchange or not opt_instrument:
                            continue
                        # 期权也使用严格指定月/指定下月过滤
                        opt_norm = self._normalize_future_id(str(opt_instrument))
                        if filter_specified_options and (not self._is_symbol_specified_or_next(opt_norm.upper())):
                            skipped_options += 1
                            continue
                        opt_key = f"{opt_exchange}_{opt_instrument}"
                        if opt_key in seen_opt_keys:
                            continue
                        seen_opt_keys.add(opt_key)
                        global_key = f"{opt_exchange}_{opt_norm}"
                        if global_key in global_seen_opts:
                            continue
                        global_seen_opts.add(global_key)
                        self.subscription_queue.append({
                            "exchange": opt_exchange,
                            "instrument_id": opt_instrument,
                            "type": "option"
                        })
                        included_options += 1

            # 额外调试：检查是否包含指定商品（如CU）的期货/期权
            try:
                if getattr(self.params, "debug_output", False):
                    target_prefixes = ["CU"]
                    fut_list = [f"{f.get('ExchangeID','')}.{f.get('InstrumentID','')}" for f in self.future_instruments if any(str(f.get('InstrumentID','')).upper().startswith(p) for p in target_prefixes)]
                    opt_list = []
                    for fsym, opts in self.option_instruments.items():
                        if any(fsym.upper().startswith(p) for p in target_prefixes):
                            opt_list.extend([f"{o.get('ExchangeID','')}.{o.get('InstrumentID','')}" for o in opts])
                    self._debug(f"[调试] 目标品种期货数量: {len(fut_list)} 示例: {fut_list[:5]}")
                    self._debug(f"[调试] 目标品种期权数量: {len(opt_list)} 示例: {opt_list[:5]}")
            except Exception:
                pass
            
            build_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.output(f"订阅队列已构建 {len(self.subscription_queue)} 个合约（期货: {fut_included} 个，期权: {included_options} 个），耗时 {build_ms:.1f}ms")
            # 统计过滤效果
            if self._resolve_subscribe_flag(
                "subscribe_only_specified_month_futures",
                "subscribe_only_current_next_futures",
                False
            ):
                self._debug(f"期货队列统计: 包含 {fut_included} 条，过滤跳过 {fut_skipped} 条")
            if self.params.subscribe_options:
                self._debug(f"期权队列统计: 包含 {included_options} 条，过滤跳过 {skipped_options} 条")
            
            # 立即订阅第一批合约（不使用定时器）
            self._subscribe_next_batch(0)
            
            # 使用Scheduler订阅下一批
            next_batch_index = self.subscription_batch_size
            if next_batch_index < len(self.subscription_queue):
                # 使用安全添加，避免同名任务重复添加
                next_job_id = f"subscribe_batch_{next_batch_index // self.subscription_batch_size + 1}"
                # 动态退避：后续批次延时 = 基础间隔 * (1 + backoff_factor * (批次序号-1))
                batch_seq = next_batch_index // self.subscription_batch_size + 1
                dynamic_delay = int(max(1, self.subscription_interval * (1 + self.subscription_backoff_factor * (batch_seq - 1))))
                self._safe_add_once_job(
                    job_id=next_job_id,
                    func=lambda: self._subscribe_next_batch(next_batch_index, next_job_id),
                    delay_seconds=dynamic_delay
                )
                self.output(f"已安排下一批订阅，批次 {next_batch_index // self.subscription_batch_size + 1}")
            
            self.output(f"=== 分批订阅完成，共订阅 {len(self.subscription_queue)} 个合约 ===")
        except Exception as e:
            self.output(f"分批订阅失败: {e}\n{traceback.format_exc()}")
    
    def _subscribe_next_batch(self, batch_index: int, job_id: Optional[str] = None) -> None:
        """订阅下一批合约"""
        # 暂停或非运行状态下不再继续订阅，也不再级联安排后续批次（避免暂停期间刷日志）
        if (not self.my_is_running) or self.my_is_paused or (self.my_trading is False) or getattr(self, "my_destroyed", False):
            if job_id:
                self._remove_job_silent(job_id)
            self._debug("已暂停/非运行，跳过本批次订阅并停止级联安排")
            return
        self.output(f"=== _subscribe_next_batch 被调用，batch_index={batch_index} ===")
        if batch_index >= len(self.subscription_queue):
            self._debug("所有合约已订阅完成")
            if job_id:
                self._remove_job_silent(job_id)
            return
        
        # 订阅当前批次
        batch = self.subscription_queue[batch_index:batch_index + self.subscription_batch_size]
        success_cnt = 0
        fail_cnt = 0
        dup_cnt = 0
        for item in batch:
            try:
                sub_key = f"{item['exchange']}|{item['instrument_id']}"
                if sub_key in self.subscribed_instruments:
                    dup_cnt += 1
                    continue
                self.sub_market_data(exchange=item["exchange"], instrument_id=item["instrument_id"])
                self.subscribed_instruments.add(sub_key)
                success_cnt += 1
                self.output(f"已订阅{item['type']}: {item['exchange']}.{item['instrument_id']}")
            except Exception as e:
                fail_cnt += 1
                self.output(f"订阅失败 {item['exchange']}.{item['instrument_id']}: {e}")
        
        # 使用Scheduler订阅下一批
        next_batch_index = batch_index + self.subscription_batch_size
        self._debug(f"订阅批次完成: 成功 {success_cnt}，失败{fail_cnt}，重复{dup_cnt}")
        if job_id:
            self._remove_job_silent(job_id)
        if next_batch_index < len(self.subscription_queue):
            next_job_id = f"subscribe_batch_{next_batch_index // self.subscription_batch_size + 1}"
            batch_seq = next_batch_index // self.subscription_batch_size + 1
            dynamic_delay = int(max(1, self.subscription_interval * (1 + self.subscription_backoff_factor * (batch_seq - 1))))
            self._safe_add_once_job(
                job_id=next_job_id,
                func=lambda: self._subscribe_next_batch(next_batch_index, next_job_id),
                delay_seconds=dynamic_delay
            )
            self._debug(f"已安排下一批订阅，批次 {next_batch_index // self.subscription_batch_size + 1}")

    def on_stop(self, *args: Any, **kwargs: Any) -> None:
        """平台回调 on_stop：默认映射为"暂停"，可通过参数切换为"停止"."""
        try:
            self.output("=" * 60)
            self.output("【调试】on_stop 方法被调用")
            self.output(f"【调试】调用参数 args={args}, kwargs={kwargs}")
            self.output(
                f"【调试】调用前状态 my_is_running={self.my_is_running} my_is_paused={self.my_is_paused} "
                f"my_trading={self.my_trading} my_destroyed={self.my_destroyed}"
            )
            self.output(f"【调试】pause_on_stop 参数={getattr(self.params, 'pause_on_stop', True)}")
            try:
                self.output("【调试】准备调用 super().on_stop(*args, **kwargs)")
                super().on_stop(*args, **kwargs)
                self.output("【调试】super().on_stop(*args, **kwargs) 调用完成")
            except Exception as e:
                self.output(f"【调试】super().on_stop(*args, **kwargs) 调用失败: {e}")
            try:
                if bool(getattr(self.params, "pause_on_stop", True)):
                    try:
                        self.output("【调试】on_stop 映射为暂停，准备调用 self.pause_strategy()")
                    except Exception:
                        pass
                    self.pause_strategy()
                else:
                    try:
                        self.output("【调试】on_stop 映射为停止，准备调用 self.stop()")
                    except Exception:
                        pass
                    self.stop()
                self.output(
                    f"【调试】调用后状态 my_is_running={self.my_is_running} my_is_paused={self.my_is_paused} "
                    f"my_trading={self.my_trading} my_destroyed={self.my_destroyed}"
                )
            except Exception as e:
                self.output(f"on_stop 处理失败: {e}\n{traceback.format_exc()}")

            # 深度清理运行态缓存，避免停启复用的脏数据
            try:
                self.option_width_results = {}
                self.kline_data = {}
                self.option_type_cache = {}
                self.out_of_money_cache = {}
                self.subscribed_instruments = set()
                self._instruments_ready = False
            except Exception:
                pass
            self.output("=" * 60)
        except Exception as e:
            self.output(f"on_stop 处理失败: {e}\n{traceback.format_exc()}")

    def stop(self) -> None:
        """策略停止逻辑"""
        # 防重复调用保护：如果已经停止过，则跳过
        if self.my_destroyed:
            self._debug("stop() 已被调用过，跳过重复执行")
            return
        
        # 停止平仓管理器
        try:
            if hasattr(self, 'position_manager') and self.position_manager:
                self.position_manager.stop()
                self.output("平仓管理器已停止", force=True)
        except Exception as e:
            try:
                self.output(f"平仓管理器停止失败: {e}", force=True)
            except Exception:
                pass
        
        self.unsubscribe_all()
        self.my_is_running = False
        self.my_is_paused = False
        # 重置 start() 执行标志，允许重新启动时再次执行
        self._start_executed = False
        # 同步关闭交易与标记已销毁，阻断后续调度与回调
        self.my_trading = False
        self.my_destroyed = True
        self.my_state = "stopped"
        # 恢复诊断前的参数设置，避免影响后续正式运行
        try:
            if isinstance(self._diag_backup, dict):
                for k, v in self._diag_backup.items():
                    try:
                        setattr(self.params, k, v)
                    except Exception:
                        pass
                self._diag_backup = None
        except Exception:
            pass
        # 清理订阅批次任务与宽度定时任务，避免残留阻塞
        try:
            for job_id in list(self.subscription_job_ids):
                self._remove_job_silent(job_id)
            self.subscription_job_ids.clear()
            self._remove_job_silent("calculate_all_option_widths")
            # 清空订阅队列与缓存，避免后续误用
            try:
                self.subscription_queue = []
                self.subscribed_instruments.clear()
            except Exception:
                pass
            # 清理文档示例订阅
            for key in list(self._doc_demo_instruments):
                try:
                    exch, inst = key.split("|", 1)
                    self.unsub_market_data(exchange=exch, instrument_id=inst)
                except Exception:
                    pass
            self._doc_demo_instruments.clear()
        except Exception:
            pass
        # 彻底关闭调度器，避免实例删除后仍继续输出
        try:
            # 常见API：shutdown / stop / remove_all_jobs / clear / cancel_all
            if hasattr(self.scheduler, "remove_all_jobs"):
                try:
                    self.scheduler.remove_all_jobs()
                except Exception:
                    pass
            for m in ("shutdown", "stop", "clear", "cancel_all"):
                try:
                    getattr(self.scheduler, m)()
                except Exception:
                    pass
        except Exception:
            pass
        self.output("策略停止")

    def place_order(
        self,
        exchange: str,
        instrument_id: str,
        direction: str,
        offset_flag: str,
        price: float,
        volume: int,
        order_price_type: str = "2"
    ) -> Optional[str]:
        """
        发出订单委托（包装器方法，供平仓管理器使用）
        
        参数：
            exchange: 交易所代码
            instrument_id: 合约代码
            direction: 方向 ("0"-买/"1"-卖)
            offset_flag: 开平标志 ("0"-开/"1"-平)
            price: 委托价格
            volume: 委托数量
            order_price_type: 委托价格类型 ("2"-限价)
        
        返回：
            订单ID（成功）或 None（失败）
        """
        try:
            # 构建订单请求
            order_request = {
                "ExchangeID": exchange,
                "InstrumentID": instrument_id,
                "Direction": direction,
                "OffsetFlag": offset_flag,
                "OrderPriceType": order_price_type,
                "LimitPrice": float(price),
                "VolumeTotalOriginal": int(volume),
                "TimeCondition": "3",
                "VolumeCondition": "1",
                "ContingentCondition": "1",
                "ForceCloseReason": "0",
                "HedgeFlag": "1",
                "MinVolume": 1,
                "IsAutoSuspend": 0,
                "UserForceClose": 0,
                "IsSwapOrder": 0,
                "BusinessUnit": "1",
            }
            
            # 调用父类的 send_order 方法
            if not hasattr(self, "send_order"):
                self.output("错误：策略缺少 send_order 方法", force=True)
                return None
            
            order_result = self.send_order(order_request)
            
            # 检查返回结果
            if not isinstance(order_result, dict):
                self.output(f"错误：委托返回格式错误 {instrument_id}", force=True)
                return None
            
            if not order_result.get("success", False):
                error_msg = order_result.get("message", "委托失败")
                self.output(f"错误：{instrument_id} 委托失败 - {error_msg}", force=True)
                return None
            
            # 返回订单ID
            order_id = order_result.get("order_id", "")
            if order_id:
                self.output(f"成功：{instrument_id} 委托已提交，订单ID={order_id}")
            return order_id
            
        except Exception as e:
            self.output(f"错误：{instrument_id} 委托异常 - {e}", force=True)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        撤销订单委托（包装器方法，供平仓管理器使用）
        
        参数：
            order_id: 订单ID
        
        返回：
            True（成功）或 False（失败）
        """
        try:
            # 调用父类的 cancel_order 方法
            try:
                result = super().cancel_order(order_id)
            except Exception as e:
                self.output(f"警告：父类 cancel_order 调用失败 - {e}", force=True)
                return False
            
            if result:
                self.output(f"成功：订单 {order_id} 已撤销")
            else:
                self.output(f"错误：订单 {order_id} 撤销失败", force=True)
            
            return result
            
        except Exception as e:
            self.output(f"错误：订单 {order_id} 撤销异常 - {e}", force=True)
            return False

    def get_recent_m1_kline(self, exchange: str, instrument_id: str, count: int = 10) -> List[KLineData]:
        """获取最近N根分钟K线（优先使用 MarketCenter.get_kline_data 的count 参数）"""
        try:
            mc_get_kline = getattr(self.market_center, "get_kline_data", None)
            bars = []
            if callable(mc_get_kline):
                try:
                    # 优先按说明文档使用count=-N
                    bars = mc_get_kline(
                        exchange=exchange,
                        instrument_id=instrument_id,
                        style="M1",
                        count=-(abs(count))
                    )
                except TypeError:
                    # 回退：以当前时间为终点，向前 count 分钟
                    end_dt = datetime.now()
                    start_dt = end_dt - timedelta(minutes=abs(count))
                    bars = mc_get_kline(
                        exchange=exchange,
                        instrument_id=instrument_id,
                        style="M1",
                        start_time=start_dt,
                        end_time=end_dt
                    )
            # 转换并落盘到 kline_data（两种键格式）
            result: List[KLineData] = []
            for bar in bars or []:
                # 使用轻量K线对象，避免构造KLineData 触发参数不兼容错误
                lk = self._to_light_kline(bar)
                # 过滤零价格
                if getattr(lk, 'close', 0) and lk.close > 0:
                    result.append(lk)
            if result:
                for key_fmt in (f"{exchange}_{instrument_id}", f"{exchange}|{instrument_id}"):
                    if key_fmt not in self.kline_data:
                        self.kline_data[key_fmt] = {'generator': None, 'data': []}
                    data_list = self.kline_data[key_fmt].get('data', [])
                    if isinstance(data_list, list):
                        self.kline_data[key_fmt]['data'].extend(result)
                        if len(self.kline_data[key_fmt]['data']) > self.params.max_kline:
                            self.kline_data[key_fmt]['data'] = self.kline_data[key_fmt]['data'][-self.params.max_kline:]
            return result
        except Exception as e:
            self.output(f"获取最近M1失败 {exchange}.{instrument_id}: {e}")
            return []

    def get_m1_kline_range(self, exchange: str, instrument_id: str, start_time: datetime, end_time: datetime) -> List[KLineData]:
        """按时间区间获取分钟K线，调用 MarketCenter.get_kline_data(start_time, end_time)"""
        try:
            mc_get_kline = getattr(self.market_center, "get_kline_data", None)
            bars = []
            if callable(mc_get_kline):
                try:
                    bars = mc_get_kline(
                        exchange=exchange,
                        instrument_id=instrument_id,
                        style="M1",
                        start_time=start_time,
                        end_time=end_time
                    )
                except Exception as e:
                    self.output(f"时间区间获取失败 {exchange}.{instrument_id}: {e}")
                    bars = []
            # 转换并落盘到 kline_data（两种键格式）
            result: List[KLineData] = []
            for bar in bars or []:
                # 使用轻量K线对象，避免构造KLineData 触发参数不兼容错误
                lk = self._to_light_kline(bar)
                if getattr(lk, 'close', 0) and lk.close > 0:
                    result.append(lk)
            if result:
                for key_fmt in (f"{exchange}_{instrument_id}", f"{exchange}|{instrument_id}"):
                    if key_fmt not in self.kline_data:
                        self.kline_data[key_fmt] = {'generator': None, 'data': []}
                    data_list = self.kline_data[key_fmt].get('data', [])
                    if isinstance(data_list, list):
                        self.kline_data[key_fmt]['data'].extend(result)
                        if len(self.kline_data[key_fmt]['data']) > self.params.max_kline:
                            self.kline_data[key_fmt]['data'] = self.kline_data[key_fmt]['data'][-self.params.max_kline:]
            return result
        except Exception as e:
            self.output(f"时间区间M1失败 {exchange}.{instrument_id}: {e}")
            return []

    def pause_strategy(self) -> None:
        """暂停策略：不退订已订阅的合约，但停止计算与后续批次订阅"""
        try:
            # 步骤1：收到暂停按钮
            self.output("暂停：收到暂停按钮")
            try:
                pending = len(getattr(self, "subscription_queue", []) or [])
            except Exception:
                pending = None
            self._debug(
                f"暂停前状态 my_is_running={self.my_is_running} my_is_paused={self.my_is_paused} "
                f"my_trading={self.my_trading} my_destroyed={self.my_destroyed} pending_subscribe={pending}"
            )
            # 仅切换运行状态，忽略后续行情推送
            self._set_strategy_state(paused=True)

            # 步骤2：停止订阅任务调度
            try:
                self.output("暂停：停止订阅任务调度")
                removed_count = 0
                for job_id in list(self.subscription_job_ids):
                    self._remove_job_silent(job_id)
                    removed_count += 1
                self.subscription_job_ids.clear()
                self.output(f"暂停：已清理订阅任务 {removed_count} 个")
                # 步骤3：停止定时计算任务
                self.output("暂停：停止定时计算任务")
                self._remove_job_silent("calculate_all_option_widths")
                # 兜底：清理调度器所有任务，防止残留导致"暂停无效"
                try:
                    self.scheduler.remove_all_jobs()
                    self._debug("暂停：调度器所有任务已清理")
                except Exception:
                    pass
                # 可选：强制停止调度器，彻底阻断后续任务触发（resume时重启）
                try:
                    if getattr(self.params, "pause_force_stop_scheduler", True):
                        self.output("暂停：准备强制停止调度器")
                        for m in ("shutdown", "stop", "clear", "cancel_all"):
                            try:
                                self.output(f"暂停：尝试调用 scheduler.{m}()")
                                getattr(self.scheduler, m)()
                                self.output(f"暂停：scheduler.{m}() 调用成功")
                            except Exception as e:
                                self.output(f"暂停：scheduler.{m}() 调用失败: {e}")
                        setattr(self, "_scheduler_stopped_by_pause", True)
                        self._debug("暂停：调度器已强制停止")
                except Exception:
                    pass
            except Exception:
                pass

            # 按用户需求：暂停即完全安静（可配置），退订所有行情
            if getattr(self.params, "pause_unsubscribe_all", True):
                try:
                    # 步骤4：退订全部（可选）
                    self.output("暂停：退订所有已订阅行情（配置开启）")
                    self.unsubscribe_all()
                    # 清空已订阅记录，以便恢复时重新订阅
                    self.subscribed_instruments.clear()
                    self.output("暂停：退订完成")
                except Exception:
                    pass
            else:
                self.output("暂停：保留已订阅行情（配置关闭）")

            # 同步清理文档示例的订阅
            try:
                for key in list(self._doc_demo_instruments):
                    try:
                        exch, inst = key.split("|", 1)
                        self.unsub_market_data(exchange=exch, instrument_id=inst)
                    except Exception:
                        pass
                self._doc_demo_instruments.clear()
            except Exception:
                pass

            # 步骤5：进入paused 状态
            self.output("暂停：进入paused 状态")
            try:
                pending_after = len(getattr(self, "subscription_queue", []) or [])
            except Exception:
                pending_after = None
            self._debug(
                f"暂停后状态 my_is_running={self.my_is_running} my_is_paused={self.my_is_paused} "
                f"my_trading={self.my_trading} my_destroyed={self.my_destroyed} pending_subscribe={pending_after}"
            )
            self.output("策略暂停")
        except Exception as e:
            self.output(f"策略暂停逻辑失败: {e}\n{traceback.format_exc()}")

    def resume_strategy(self) -> None:
        """恢复策略：继续接收行情回调并恢复宽度定时计算"""
        try:
            # 统一设置策略状态为运行
            self._set_strategy_state(paused=False)

            # 恢复定时计算任务
            if getattr(self.params, "enable_scheduler", True):
                self._safe_add_interval_job(
                    job_id="calculate_all_option_widths",
                    func=self.calculate_all_option_widths,
                    seconds=self.calculation_interval
                )
            else:
                self._debug("恢复：enable_scheduler=False，跳过定时计算任务")

            # 若暂停时强制停止过调度器，则尝试重启
            try:
                if getattr(self, "_scheduler_stopped_by_pause", False):
                    try:
                        self.output("恢复：准备重启调度器")
                        self.scheduler.start()
                        self.output("恢复：调度器已重启")
                        self._debug("恢复：调度器已重启")
                    except Exception as e:
                        self.output(f"恢复：调度器重启失败: {e}")
                    self._scheduler_stopped_by_pause = False
                else:
                    self.output("恢复：调度器未被暂停时停止，无需重启")
            except Exception:
                pass

            # 恢复行情订阅（使用批次订阅以避免阻塞），仅当暂停时进行了退订
            if getattr(self.params, "pause_unsubscribe_all", True):
                try:
                    self._subscribe_in_batches()
                except Exception as e:
                    self.output(f"恢复订阅失败: {e}")

            # 恢复时输出并复位暂停期间的丢弃计数，便于一次性核对
            try:
                td = int(self.paused_drop_counts.get("tick", 0))
                kd = int(self.paused_drop_counts.get("kline", 0))
                self._debug(f"恢复：暂停期间丢弃 tick={td} kline={kd}")
                self.paused_drop_counts["tick"] = 0
                self.paused_drop_counts["kline"] = 0
            except Exception:
                pass

            self.output("策略恢复")
        except Exception as e:
            self.output(f"策略恢复失败: {e}\n{traceback.format_exc()}")

    def on_tick(self, tick: TickData, *args, **kwargs) -> None:
        """Tick 回调：用于合成K 线并刷新宽度计算"""
        try:
            # 暂停/未运行/销毁或交易关闭时立即返回，避免任何日志输出
            if self._is_paused_or_stopped():
                # 统计丢弃计数
                try:
                    self.paused_drop_counts["tick"] = self.paused_drop_counts.get("tick", 0) + 1
                except Exception:
                    pass
                return
            # 监控平台可能修改的属性（每100次tick输出一次，避免日志过多）
            try:
                self.tick_count = getattr(self, "tick_count", 0) + 1
                if self.tick_count % 100 == 0:
                    self.output("=" * 60)
                    self.output(
                        "  my_flags="
                        f"is_running={self.my_is_running} "
                        f"is_paused={self.my_is_paused} "
                        f"state={self.my_state} "
                        f"trading={self.my_trading} "
                        f"destroyed={self.my_destroyed} "
                        f"started={self.my_started}"
                    )
                    self.output("=" * 60)
            except Exception as e:
                self._debug(f"平台属性监控失败 {e}")
            
            # 暂停/未运行/销毁或交易关闭时忽略后续处理（上方已返回，一般不会到此）
            if self._is_paused_or_stopped():
                return
            try:
                super().on_tick(tick, *args, **kwargs)
            except Exception:
                pass

            if (not getattr(self, "is_running", False)) or getattr(self, "is_paused", False) or (getattr(self, "trading", True) is False) or getattr(self, "destroyed", False):
                return

            tick_inst = getattr(tick, "instrument_id", "") or getattr(tick, "InstrumentID", "") or ""
            tick_exch = getattr(tick, "exchange", "") or getattr(tick, "ExchangeID", "") or ""
            if not self._is_instrument_allowed(tick_inst, tick_exch):
                return

            # 若平台已推送K线，可关闭Tick合成K线生成器以减少冗余
            if bool(getattr(self.params, "use_tick_kline_generator", False)) and str(getattr(self.params, "output_mode", "debug")).lower() != "debug":
                self.update_tick_data(tick)

            # 集成平仓管理器：处理 tick 回调
            try:
                if hasattr(self, 'position_manager') and self.position_manager:
                    if self._should_forward_to_position_manager(tick_inst, tick_exch):
                        self.position_manager.handle_tick(tick)
            except Exception as e:
                self._debug(f"平仓管理器 tick 处理失败: {e}")
        except Exception as e:
            self._debug(f"on_tick 处理失败: {e}")

    def on_trade(self, trade_data: Any, *args, **kwargs) -> None:
        """成交回报回调 - 集成平仓管理器"""
        try:
            # 暂停/未运行/销毁或交易关闭时立即返回
            if self._is_paused_or_stopped():
                return

            # 调用父类的 on_trade
            try:
                super().on_trade(trade_data, *args, **kwargs)
            except Exception:
                pass

            # 集成平仓管理器：处理新开仓
            try:
                if hasattr(self, 'position_manager') and self.position_manager:
                    # 检查是否为开仓（offset_flag == "0"）
                    offset_flag = getattr(trade_data, 'offset_flag', None)
                    if offset_flag == "0":
                        inst_id = getattr(trade_data, 'instrument_id', None) or getattr(trade_data, 'InstrumentID', None) or ""
                        exch = getattr(trade_data, 'exchange', None) or getattr(trade_data, 'ExchangeID', None) or ""
                        if self._should_forward_to_position_manager(inst_id, exch):
                            self.position_manager.handle_new_position(trade_data)
            except Exception as e:
                self._debug(f"平仓管理器 on_trade 处理失败: {e}")
        except Exception as e:
            self._debug(f"on_trade 处理失败: {e}")

    def on_order_trade(self, trade_data: Any, *args, **kwargs) -> None:
        """委托单成交回报回调 - 集成平仓管理器"""
        try:
            # 暂停/未运行/销毁或交易关闭时立即返回
            if self._is_paused_or_stopped():
                return

            # 调用父类的 on_order_trade
            try:
                super().on_order_trade(trade_data, *args, **kwargs)
            except Exception:
                pass

            # 集成平仓管理器：处理订单成交
            try:
                if hasattr(self, 'position_manager') and self.position_manager:
                    inst_id = getattr(trade_data, 'instrument_id', None) or getattr(trade_data, 'InstrumentID', None) or ""
                    exch = getattr(trade_data, 'exchange', None) or getattr(trade_data, 'ExchangeID', None) or ""
                    if self._should_forward_to_position_manager(inst_id, exch):
                        self.position_manager.handle_order_trade(trade_data)
            except Exception as e:
                self._debug(f"平仓管理器 on_order_trade 处理失败: {e}")
        except Exception as e:
            self._debug(f"on_order_trade 处理失败: {e}")

    def on_order(self, order_data: Any, *args, **kwargs) -> None:
        """订单状态回调 - 集成平仓管理器"""
        try:
            # 暂停/未运行/销毁或交易关闭时立即返回
            if self._is_paused_or_stopped():
                return

            # 调用父类的 on_order
            try:
                super().on_order(order_data, *args, **kwargs)
            except Exception:
                pass

            # 集成平仓管理器：处理订单状态
            try:
                if hasattr(self, 'position_manager') and self.position_manager:
                    inst_id = getattr(order_data, 'instrument_id', None) or getattr(order_data, 'InstrumentID', None) or ""
                    exch = getattr(order_data, 'exchange', None) or getattr(order_data, 'ExchangeID', None) or ""
                    if self._should_forward_to_position_manager(inst_id, exch):
                        self.position_manager.handle_order(order_data)
            except Exception as e:
                self._debug(f"平仓管理器 on_order 处理失败: {e}")
        except Exception as e:
            self._debug(f"on_order 处理失败: {e}")

    def _get_tick_price(self, tick: TickData) -> Optional[float]:
        """兼容不同 TickData 字段，优先使用last/price，其次用一档双边均价"""
        try:
            for field in ("last", "last_price", "price", "LastPrice", "Last", "close", "Close"):
                val = getattr(tick, field, None)
                if val not in (None, ""):
                    try:
                        return float(val)
                    except Exception:
                        continue

            bid = getattr(tick, "bid", None) or getattr(tick, "BidPrice1", None)
            ask = getattr(tick, "ask", None) or getattr(tick, "AskPrice1", None)
            try:
                if bid not in (None, "") and ask not in (None, ""):
                    return (float(bid) + float(ask)) / 2
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _on_kline_from_tick(self, kline: Any) -> None:
        """KLineGenerator 回调：将合成的K 线写入缓存并触发计算"""
        try:
            exch = getattr(kline, "exchange", "")
            inst = getattr(kline, "instrument_id", "")
            style = getattr(kline, "style", getattr(self.params, "kline_style", "M1"))
            light_bar = self._to_light_kline(kline)
            light_bar.exchange = exch
            light_bar.instrument_id = inst
            light_bar.style = style
            self._process_kline_data(exch, inst, style, light_bar)
            self._trigger_width_calc_for_kline(light_bar)
        except Exception as e:
            self._debug(f"Tick 合成 K 线处理失败{e}")

    def update_tick_data(self, tick: TickData) -> None:
        """由Tick 合成 K 线，保证没有平台 K 线推送时也能更新"""
        try:
            exchange = getattr(tick, "exchange", getattr(tick, "ExchangeID", ""))
            instrument_id = getattr(tick, "instrument_id", getattr(tick, "InstrumentID", ""))
            if not (exchange and instrument_id):
                return

            style = getattr(self.params, "kline_style", "M1")
            key_variants = (f"{exchange}_{instrument_id}", f"{exchange}|{instrument_id}")

            # 确保生成器存在
            generator = None
            for key in key_variants:
                entry = self.kline_data.get(key)
                if entry is None:
                    entry = {"generator": None, "data": []}
                    self.kline_data[key] = entry
                if entry.get("generator") is None:
                    try:
                        entry["generator"] = KLineGenerator(
                            callback=self._on_kline_from_tick,
                            exchange=exchange,
                            instrument_id=instrument_id,
                            style=style
                        )
                    except Exception:
                        entry["generator"] = None
                if entry.get("generator") is not None and generator is None:
                    generator = entry.get("generator")

            # 尝试使用生成器合成K 线
            used_generator = False
            if generator is not None:
                try:
                    generator.tick_to_kline(tick)
                    used_generator = True
                except Exception:
                    used_generator = False

            # 若没有生成器或调用失败，则直接以 tick 价格拼接一根轻量K 线
            if not used_generator:
                price = self._get_tick_price(tick)
                if price is None:
                    return
                bar = types.SimpleNamespace(
                    exchange=exchange,
                    instrument_id=instrument_id,
                    style=style,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=getattr(tick, "volume", getattr(tick, "Volume", 0))
                )
                self._on_kline_from_tick(bar)
        except Exception as e:
            self._debug(f"update_tick_data 异常: {e}")

    def _process_kline_data(self, exchange: str, instrument_id: str, frequency: str, kline: KLineData) -> None:
        """处理K线数据并落盘缓存，由平台K线推送直接调用"""
        try:
            # 将无效/零价格K线视为0值写入，避免提前返回导致后续行权价遍历中断
            try:
                close_val = float(getattr(kline, 'close', 0) or 0)
            except Exception:
                close_val = 0.0
            if close_val <= 0:
                try:
                    setattr(kline, 'close', 0.0)
                except Exception:
                    pass
            else:
                try:
                    setattr(kline, 'close', close_val)
                except Exception:
                    pass

            # 确保每根K线具备时间戳，缺失则赋当前时间，便于后续新鲜度判断
            try:
                if not getattr(kline, "datetime", None):
                    setattr(kline, "datetime", datetime.now())
            except Exception:
                pass
            inst_upper = str(instrument_id).upper()
            exch_upper = str(exchange).upper()
            keys = (
                f"{exchange}|{instrument_id}",
                f"{exchange}_{instrument_id}",
                f"{exch_upper}|{inst_upper}",
                f"{exch_upper}_{inst_upper}",
            )
            seen = set()
            for key in keys:
                if key in seen:
                    continue
                seen.add(key)
                if key not in self.kline_data:
                    self.kline_data[key] = {"generator": None, "data": []}
                if frequency not in self.kline_data[key]:
                    self.kline_data[key][frequency] = []

                self.kline_data[key][frequency].append(kline)
                if len(self.kline_data[key][frequency]) > self.params.max_kline:
                    self.kline_data[key][frequency] = self.kline_data[key][frequency][-self.params.max_kline:]

                # 同步 data 列表，兼容旧结构
                data_list = self.kline_data[key].get("data")
                if isinstance(data_list, list):
                    data_list.append(kline)
                    if len(data_list) > self.params.max_kline:
                        self.kline_data[key]["data"] = data_list[-self.params.max_kline:]
                else:
                    self.kline_data[key]["data"] = [kline]
        except Exception as e:
            self.output(f"处理K线数据失败 {e}\n{traceback.format_exc()}")

    def _get_kline_series(self, exchange: str, instrument_id: str) -> List[KLineData]:
        """统一获取K线列表，兼容 pipe/underscore 键和值结构差异"""
        freq = getattr(self.params, "kline_style", "M1")
        inst_upper = str(instrument_id).upper()
        exch_upper = str(exchange).upper()
        key_variants = [
            f"{exchange}_{instrument_id}",
            f"{exchange}|{instrument_id}",
            f"{exch_upper}_{inst_upper}",
            f"{exch_upper}|{inst_upper}",
        ]

        # 去重保持原顺序
        seen_keys = set()
        key_variants = [k for k in key_variants if not (k in seen_keys or seen_keys.add(k))]

        for key in key_variants:
            series = self.kline_data.get(key)
            if series is None:
                continue

            # 结构1：{"generator": None, "data": [...]}
            if isinstance(series, dict):
                data_list = series.get("data")
                if isinstance(data_list, list):
                    return data_list

                freq_list = series.get(freq)
                if isinstance(freq_list, list):
                    return freq_list

            # 结构2：直接存储为列表
            if isinstance(series, list):
                return series

        return []
    
    def _previous_price_from_klines(self, klines: List[Any]) -> float:
        """根据当前时间和 K 线序列选择上一根 K 线的 close 值。
        规则：
        - 若处于开盘时间内，上一根为序列中的前一根；
        - 若处于开盘时间外，上一根为上一次开盘时段结束时刻对应的最后一根 K 线（找不到则退回倒数第二根）；
        - 若上一根成交量为 0 或不存在，则返回 0。
        """
        try:
            if not klines or len(klines) == 0:
                return 0

            def _get_bar_ts(bar: Any) -> Optional[datetime]:
                for name in ("datetime", "DateTime", "timestamp", "Timestamp", "time", "Time"):
                    try:
                        val = getattr(bar, name, None)
                    except Exception:
                        val = None
                    if val is None:
                        continue
                    if isinstance(val, datetime):
                        return val
                    if isinstance(val, (int, float)):
                        try:
                            return datetime.fromtimestamp(val)
                        except Exception:
                            continue
                    if isinstance(val, str):
                        for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y%m%d %H:%M:%S"):
                            try:
                                return datetime.fromisoformat(val) if fmt is None else datetime.strptime(val, fmt)
                            except Exception:
                                continue
                return None

            now = datetime.now()
            default_exch = getattr(self.params, "exchange", None)
            sessions = self._get_trading_sessions(now, default_exch)
            in_open = any(s <= now <= e for s, e in sessions)

            # 开盘时间内：直接取倒数第二根（安全取法）
            if in_open:
                prev = klines[-2] if len(klines) >= 2 else None
                if prev is None:
                    return 0
                prev_vol = getattr(prev, 'volume', getattr(prev, 'Volume', 0)) or 0
                if prev_vol == 0:
                    return 0
                return getattr(prev, 'close', 0) or 0

            # 非开盘时间：找到最近一个已结束的交易时段终点
            past_ends = [e for _, e in sessions if e <= now]
            if not past_ends:
                return 0
            prev_end = max(past_ends)

            candidate = None
            for b in reversed(klines):
                bts = _get_bar_ts(b)
                if bts and bts <= prev_end:
                    candidate = b
                    break
            if candidate is None:
                candidate = klines[-2] if len(klines) >= 2 else None
            if candidate is None:
                return 0
            prev_vol = getattr(candidate, 'volume', getattr(candidate, 'Volume', 0)) or 0
            if prev_vol == 0:
                return 0
            return getattr(candidate, 'close', 0) or 0
        except Exception:
            return 0
    
    def load_all_instruments(self) -> None:
        """加载所有期货和期权合约"""
        try:
            self._option_fetch_failures = set()
            # 品种到交易所的映射关系
            PRODUCT_EXCHANGE_MAP = {
                # 股指期货和期权- CFFEX
                "IF": "CFFEX", "IH": "CFFEX", "IC": "CFFEX", "IM": "CFFEX",
                "IO": "CFFEX", "HO": "CFFEX", "MO": "CFFEX", "EO": "CFFEX",
                
                # 有色金属 - SHFE
                "CU": "SHFE", "AL": "SHFE", "ZN": "SHFE",
                "NI": "SHFE", "SN": "SHFE", "PB": "SHFE",
                
                # 黑色金属 - SHFE
                "RB": "SHFE", "HC": "SHFE", "WR": "SHFE", "SS": "SHFE",
                
                # 贵金属 - SHFE
                "AU": "SHFE", "AG": "SHFE",
                
                # 能源化工 - SHFE
                "FU": "SHFE", "BU": "SHFE", "RU": "SHFE", "SP": "SHFE",
                
                # 农产品 - DCE
                "M": "DCE", "Y": "DCE", "A": "DCE", "P": "DCE",
                "C": "DCE", "CS": "DCE", "JD": "DCE", "L": "DCE",
                "V": "DCE", "PP": "DCE", "EB": "DCE", "EG": "DCE",
                "PG": "DCE", "LH": "DCE", "FB": "DCE", "BB": "DCE",
                
                # 钢铁 - DCE
                "J": "DCE", "JM": "DCE", "I": "DCE",
                
                # 化工/建材 - CZCE
                "MA": "CZCE", "FG": "CZCE", "TA": "CZCE",

                # 农产品 - CZCE
                "CF": "CZCE", "SR": "CZCE", "OI": "CZCE", "RM": "CZCE",
                "ZC": "CZCE", "WH": "CZCE", "PM": "CZCE", "RI": "CZCE",
                "JR": "CZCE", "LR": "CZCE", "SF": "CZCE", "SM": "CZCE",
                "UR": "CZCE", "SA": "CZCE", "PK": "CZCE", "AP": "CZCE",
                "CY": "CZCE", "CJ": "CZCE", "PF": "CZCE",
            }
            
            # 解析新的多交易所、多品种参数
            exchanges_str = getattr(self.params, "exchanges", "CFFEX") or "CFFEX"
            future_products_str = str(getattr(self.params, "future_products", "") or "")
            option_products_str = str(getattr(self.params, "option_products", "") or "")
            include_fut_as_opt = bool(getattr(self.params, "include_future_products_for_options", True))
            load_all = getattr(self.params, "load_all_products", True)
            self.output(f"=== load_all = {load_all} ===")
            
            # 解析逗号分隔的列表
            exchanges = [self._normalize_exchange_code(e) for e in exchanges_str.split(",") if e.strip()]
            future_products = [p.strip() for p in future_products_str.split(",") if p.strip()]
            option_products = [p.strip() for p in option_products_str.split(",") if p.strip()]
            if not future_products and not option_products:
                try:
                    mm = getattr(self.params, "month_mapping", {}) or {}
                    mm_keys = [str(k).strip().upper() for k in mm.keys() if str(k).strip()]
                    if mm_keys:
                        future_products = list(dict.fromkeys(mm_keys))
                        if include_fut_as_opt:
                            option_products = list(dict.fromkeys(option_products + future_products))
                except Exception:
                    pass
            if include_fut_as_opt:
                # 自动把期货品种也尝试作为期权品种拉取，便于覆盖商品期权（如CU/AL/ZN/RB/AG 等）
                option_products = list(dict.fromkeys(option_products + future_products))
            # 基于映射扩展 option_products
            try:
                mapping = getattr(self, "future_to_option_map", {}) or {}
                mapped_from_future = []
                for fp in future_products:
                    k = str(fp).strip().upper()
                    if not k:
                        continue
                    mapped = mapping.get(k) or k.lower()
                    if mapped and mapped not in option_products and mapped not in mapped_from_future:
                        mapped_from_future.append(mapped)
                effective_option_products = list(dict.fromkeys(option_products + mapped_from_future))
            except Exception:
                effective_option_products = option_products
            
            # 默认交易所（用于兼容）
            default_exchange = exchanges[0] if exchanges else "CFFEX"
            
            self._debug(f"=== 开始加载合约 ===")
            self._debug(f"交易所列表: {exchanges}")
            self._debug(f"期货品种列表: {future_products}")
            self._debug(f"期权品种列表: {effective_option_products} (include_future_products_for_options={include_fut_as_opt})")
            self._debug(f"load_all_products: {load_all}")
            
            fetched: List[dict] = []
            
            def _collect(res: Any, label: str) -> None:
                if res is None:
                    self._debug(f"{label} 返回 None")
                    return
                norm = self._normalize_instruments(res)
                self._debug(f"{label} 原始数量: {len(res) if hasattr(res, '__len__') else '未知'} 归一化后: {len(norm)}")
                if norm:
                    # 立即过滤：股指仍按指定月/指定下月限制，商品放宽以保留远月
                    filtered_norm = []
                    expect_option = "期权" in label
                    for inst in norm:
                        product_class = str(inst.get("ProductClass", "") or "").strip()
                        inst_id = inst.get("InstrumentID", "")
                        normalized_id = str(inst_id or "").upper()

                        # 如果当前是在期权拉取路径，但返回的不是期权（无OptionType/Strike且ProductClass不为期权），直接跳过
                        if expect_option:
                            is_option_like = (
                                product_class in ("2", "h", "H")
                                or inst.get("OptionType") not in (None, "")
                                or inst.get("StrikePrice") not in (None, "")
                                or "-C" in normalized_id
                                or "-P" in normalized_id
                            )
                            if not is_option_like:
                                key_skip = f"{label}:{inst_id}"
                                if key_skip not in self._non_option_return_logged:
                                    self._non_option_return_logged.add(key_skip)
                                    self._debug(f"过滤非期权返回: {inst_id} (label={label})")
                                continue

                        handled = False

                        # 过滤期权：股指和商品都按指定月/指定下月过滤
                        if product_class in ("2", "h", "H"):
                            handled = True
                            future_symbol = self._extract_future_symbol(inst_id)
                            if future_symbol and self._is_symbol_specified_or_next(future_symbol.upper()):
                                filtered_norm.append(inst)
                            else:
                                self._debug(f"过滤期权(非指定月/指定下月或解析失败): {inst_id}")
                            continue

                        # 过滤期货：所有品种统一限制到映射指定的指定月/指定下月
                        if product_class in ("1", "i", "I"):
                            handled = True
                            if self._is_symbol_specified_or_next(normalized_id):
                                filtered_norm.append(inst)
                            else:
                                self._debug(f"过滤期货(非指定月/指定下月): {inst_id}")
                            continue

                        # 基于字段推断缺失ProductClass 的期货/期权类型
                        option_hint = (
                            inst.get("OptionType") not in (None, "")
                            or inst.get("StrikePrice") not in (None, "")
                            or "-C" in normalized_id
                            or "-P" in normalized_id
                        )
                        if not handled and option_hint:
                            handled = True
                            future_symbol = self._extract_future_symbol(inst_id)
                            if future_symbol and self._is_symbol_specified_or_next(future_symbol.upper()):
                                filtered_norm.append(inst)
                            else:
                                self._debug(f"过滤期权(推断类型) {inst_id}: 对应期货不在指定月/指定下月或无法解析")
                            continue

                        if not handled and self._is_real_month_contract(normalized_id):
                            handled = True
                            if self._is_symbol_specified_or_next(normalized_id):
                                filtered_norm.append(inst)
                            else:
                                self._debug(f"过滤期货(推断类型) {inst_id}: 非指定月/指定下月")
                            continue

                        if not handled:
                            self._debug(f"过滤未知类型合约: {inst_id} ProductClass={product_class}")

                    fetched.extend(filtered_norm)
                    self._debug(f"{label} 归一化后已添加到fetched，当前fetched数量: {len(fetched)}")
                else:
                    self._debug(f"{label} 归一化后为空")

            def _fetch_by_product(exchange_code: str, product_code: str, category: str) -> bool:
                """按品种获取合约，若infini 返回空则回退到MarketCenter"""
                primary_res: Any = None
                try:
                    primary_res = infini.get_instruments_by_product(exchange=exchange_code, product_id=product_code)
                    _collect(primary_res, f"获取{category} {exchange_code}.{product_code}")
                except Exception as e:
                    self._debug(f"获取{category} {exchange_code}.{product_code} 失败: {e}")

                if primary_res:
                    return True

                # 回退 1：MarketCenter.get_instruments_by_product
                mc_by_prod = getattr(self.market_center, "get_instruments_by_product", None)
                if callable(mc_by_prod):
                    try:
                        mc_res = mc_by_prod(exchange=exchange_code, product_id=product_code)
                        _collect(mc_res, f"MarketCenter 获取{category} {exchange_code}.{product_code}")
                        if mc_res:
                            self.output(f"{category} {exchange_code}.{product_code} infini 返回空，已用 MarketCenter 兜底 {len(mc_res)} 个")
                            return True
                    except Exception as e:
                        self._debug(f"MarketCenter 获取{category} {exchange_code}.{product_code} 失败: {e}")

                # 回退 2：MarketCenter.get_instruments(exchange=xx) 过滤品种前缀
                mc_get_all = getattr(self.market_center, "get_instruments", None)
                if callable(mc_get_all):
                    try:
                        mc_all = mc_get_all(exchange=exchange_code)
                        filtered = []
                        for inst in self._normalize_instruments(mc_all):
                            inst_id = str(inst.get("InstrumentID", "")).upper()
                            if inst_id.startswith(product_code.upper()):
                                filtered.append(inst)
                        _collect(filtered, f"MarketCenter 过滤{category} {exchange_code}.{product_code}")
                        if filtered:
                            self.output(f"{category} {exchange_code}.{product_code} 使用 MarketCenter.get_instruments 过滤获得 {len(filtered)} 个")
                            return True
                    except Exception as e:
                        self._debug(f"MarketCenter 过滤{category} {exchange_code}.{product_code} 失败: {e}")

                self._debug(f"{category} {exchange_code}.{product_code} 未获取到合约，已尝试 infini 和MarketCenter")
                return False

            # 优先尝试获取全市场合约，确保实盘品种齐全
            if load_all:
                self._debug("load_all_products=True，尝试获取全部合约..")
                try:
                    get_all = getattr(infini, "get_instruments", None)
                    if callable(get_all):
                        try:
                            res = get_all(exchange=None)
                            self._debug(f"infini.get_instruments(exchange=None) 调用成功")
                        except TypeError:
                            res = get_all()
                            self._debug(f"infini.get_instruments() 调用成功")
                        _collect(res, f"infini.get_instruments(全部交易所)")
                except Exception as e:
                    self.output(f"获取全部合约失败: {e}\n{traceback.format_exc()}")

                if not fetched:
                    # 增强诊断：检测是否命中了本地占位 infini 实现
                    try:
                        infini_path = getattr(infini, "__file__", "")
                    except Exception:
                        infini_path = ""
                    if infini_path and ("pyStrategy" in infini_path and "pythongo" in infini_path and infini_path.endswith("infini.py")):
                        pass

                    self._debug("infini.get_instruments 未获取到合约，尝试使用MarketCenter...")
                    try:
                        mc_get_all = getattr(self.market_center, "get_instruments", None)
                        if callable(mc_get_all):
                            try:
                                res = mc_get_all(exchange=None)
                                self._debug(f"MarketCenter.get_instruments(exchange=None) 调用成功")
                            except TypeError:
                                res = mc_get_all()
                                self._debug(f"MarketCenter.get_instruments() 调用成功")
                            _collect(res, f"MarketCenter.get_instruments(全部交易所)")
                    except Exception as e:
                        self.output(f"MarketCenter 获取全部合约失败: {e}\n{traceback.format_exc()}")
                
                if fetched:
                    self._debug(f"全量获取成功，共获取 {len(fetched)} 个合约，跳过按品种拉取")
                    # 覆盖度检查：若全量返回中缺少目标品种，则按品种兜底加载，避免只拿到CFFEX导致商品期权缺失
                    try:
                        all_products_for_coverage = list(dict.fromkeys(future_products + effective_option_products))
                        inst_ids_upper = [str(inst.get("InstrumentID", "")).upper() for inst in fetched if isinstance(inst, dict)]
                        missing_products = [p for p in all_products_for_coverage if not any(iid.startswith(p.upper()) for iid in inst_ids_upper)]
                        if missing_products:
                            self.output(f"全量获取缺少品种 {missing_products}，改为按品种补拉")
                            for prod in missing_products:
                                target_ex = PRODUCT_EXCHANGE_MAP.get(prod.upper())
                                if target_ex and target_ex in exchanges:
                                    _fetch_by_product(target_ex, prod, "补拉合约")
                                else:
                                    for exch in exchanges:
                                        _fetch_by_product(exch, prod, "补拉合约")
                    except Exception as e:
                        self._debug(f"全量覆盖检查失败 {e}")
                else:
                    self._debug(f"全量获取失败或返回空，回退到按品种拉取")

        # 如果全量获取失败或未启用，则按交易所和品种拉取
            if not fetched:
                # 遍历所有期货品种，使用映射的交易所
                for fut_prod in future_products:
                    # 获取品种对应的交易所，如果映射中没有，则使用交易所列表
                    target_exchange = PRODUCT_EXCHANGE_MAP.get(fut_prod.upper())
                    if target_exchange:
                        if target_exchange not in exchanges:
                            self._debug(f"品种 {fut_prod} 的交易所 {target_exchange} 不在交易所列表中，跳过")
                            continue
                        _fetch_by_product(target_exchange, fut_prod, "期货合约")
                    else:
                        # 如果映射中没有，尝试从所有交易所加载
                        for exch in exchanges:
                            _fetch_by_product(exch, fut_prod, "期货合约")

                # 遍历所有期权品种，使用映射的交易所
            self.output("=== 开始加载期权合约 ===")
            self.output(f"期权品种列表: {effective_option_products}")
            for i, opt_prod in enumerate(effective_option_products):
                self.output(f"=== 开始加载期权品种 {i+1}/{len(effective_option_products)}: {opt_prod} ===")
                # 获取品种对应的交易所，如果映射中没有，则使用交易所列表
                target_exchange = PRODUCT_EXCHANGE_MAP.get(opt_prod.upper())
                self.output(f"  期权品种 {opt_prod} 对应的交易所: {target_exchange}")
                if target_exchange:
                    if target_exchange not in exchanges:
                        self._debug(f"品种 {opt_prod} 的交易所 {target_exchange} 不在交易所列表中，跳过")
                        continue
                    self.output(f"  调用 infini.get_instruments_by_product(exchange={target_exchange}, product_id={opt_prod})")
                    ok = _fetch_by_product(target_exchange, opt_prod, "期权合约")
                    if not ok:
                        self._option_fetch_failures.add(str(opt_prod).upper())
                    self.output(f"  获取 {target_exchange}.{opt_prod} 完成，当前fetched数量: {len(fetched)}")
                else:
                    # 如果映射中没有，尝试从所有交易所加载
                    ok = False
                    for exch in exchanges:
                        ok = _fetch_by_product(exch, opt_prod, "期权合约")
                        if ok:
                            break
                    if not ok:
                        self._option_fetch_failures.add(str(opt_prod).upper())
                self.output(f"=== 期权品种 {opt_prod} 加载完成，当前fetched数量: {len(fetched)} ===")
            self.output("=== 期权合约加载完成 ===")

            # 归一化列表，防御 None 或非可迭代返回值
            all_instruments = fetched or []
            self.output(f"=== 归一化前，fetched: {len(fetched) if fetched else 0} ===")
            try:
                _ = iter(all_instruments)
                self.output("=== 归一化成功，all_instruments 是可迭代的 ===")
            except TypeError:
                all_instruments = []
                self.output("=== 归一化失败，all_instruments 不是可迭代的 ===")

            if not all_instruments:
                # 本地演示数据回退（仅测试模式启用）
                if getattr(self.params, "test_mode", False):
                    try:
                        self.output("未获取到任何合约，启用本地演示数据回退（test_mode=True）")
                        now = datetime.now()
                        y2 = str(now.year)[2:]
                        cur_mon = now.month
                        next_mon = 1 if cur_mon == 12 else cur_mon + 1
                        cur_mon_str = f"{cur_mon:02d}"
                        next_mon_str = f"{next_mon:02d}"

                        # 选择演示的品种与交易所
                        demo_futures = [
                            ("CFFEX", f"IF{y2}{cur_mon_str}"),
                            ("CFFEX", f"IH{y2}{cur_mon_str}"),
                            ("CFFEX", f"IC{y2}{cur_mon_str}"),
                            ("CFFEX", f"IM{y2}{cur_mon_str}"),
                            ("SHFE",  f"CU{y2}{cur_mon_str}"),
                            ("SHFE",  f"RB{y2}{cur_mon_str}"),
                            ("SHFE",  f"AG{y2}{cur_mon_str}"),
                            ("DCE",   f"M{y2}{cur_mon_str}"),
                            ("CZCE",  f"SR{y2}{cur_mon_str}"),
                        ]

                        # 构造期货与期权合约列表
                        synthetic_futures = []
                        synthetic_options = []

                        # 为每个期货生成指定月与指定下月的期权（看涨为主，便于上涨方向演示），以及注入基础K线
                        for exch, fut in demo_futures:
                            # 指定月与指定下月期货ID
                            fut_cur = fut
                            fut_next = f"{fut[:-2]}{next_mon_str}"

                            # 添加期货合约字典
                            synthetic_futures.append({"ExchangeID": exch, "InstrumentID": fut_cur, "ProductClass": "1"})
                            synthetic_futures.append({"ExchangeID": exch, "InstrumentID": fut_next, "ProductClass": "1"})

                            # 注入期货K线（两根，模拟上涨）
                            for fid in (fut_cur, fut_next):
                                key = f"{exch}_{fid}"
                                series = [
                                    self._to_light_kline({"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0, "volume": 1000}),
                                    self._to_light_kline({"open": 101.0, "high": 102.0, "low": 100.5, "close": 101.0, "volume": 1200}),
                                ]
                                self.kline_data[key] = {"generator": None, "data": series}

                            # 期权前缀映射（指数期权）
                            prefix_map = {"IF": "IO", "IH": "HO", "IC": "MO", "IM": "EO"}
                            prod = re.match(r"^([A-Za-z]{1,})\d{2}\d{1,2}$", fut_cur)
                            prod_code = prod.group(1).upper() if prod else fut_cur[:2].upper()
                            opt_prefix = prefix_map.get(prod_code, prod_code)

                            # 基准行权价（略高于期货价，确保看涨为虚值）
                            base_strike_cur = 105.0
                            base_strike_next = 106.0

                            # 构造指定月看涨期权（两档行权价）
                            opt_ids_cur = [
                                f"{opt_prefix}{y2}{cur_mon_str}C{int(base_strike_cur)}",
                                f"{opt_prefix}{y2}{cur_mon_str}C{int(base_strike_cur+2)}",
                            ]
                            # 构造指定下月看涨期权（两档行权价）
                            opt_ids_next = [
                                f"{opt_prefix}{y2}{next_mon_str}C{int(base_strike_next)}",
                                f"{opt_prefix}{y2}{next_mon_str}C{int(base_strike_next+2)}",
                            ]

                            for oid in opt_ids_cur:
                                synthetic_options.append({
                                    "ExchangeID": exch,
                                    "InstrumentID": oid,
                                    "ProductClass": "2",
                                    "OptionType": "C",
                                    "StrikePrice": float(re.search(r"C(\d+(?:\.\d+)?)", oid).group(1)),
                                })
                                # 注入期权K线（上涨）
                                key = f"{exch}_{oid}"
                                series = [
                                    self._to_light_kline({"open": 5.0, "high": 5.2, "low": 4.9, "close": 5.0, "volume": 500}),
                                    self._to_light_kline({"open": 5.1, "high": 5.4, "low": 5.0, "close": 5.3, "volume": 600}),
                                ]
                                self.kline_data[key] = {"generator": None, "data": series}

                            for oid in opt_ids_next:
                                synthetic_options.append({
                                    "ExchangeID": exch,
                                    "InstrumentID": oid,
                                    "ProductClass": "2",
                                    "OptionType": "C",
                                    "StrikePrice": float(re.search(r"C(\d+(?:\.\d+)?)", oid).group(1)),
                                })
                                key = f"{exch}_{oid}"
                                series = [
                                    self._to_light_kline({"open": 4.0, "high": 4.2, "low": 3.9, "close": 4.0, "volume": 450}),
                                    self._to_light_kline({"open": 4.1, "high": 4.3, "low": 4.0, "close": 4.2, "volume": 520}),
                                ]
                                self.kline_data[key] = {"generator": None, "data": series}

                        # 聚合为 all_instruments 供后续通用分离逻辑使用
                        all_instruments = synthetic_futures + synthetic_options
                        self.output(f"本地演示数据生成完成：期货{len(synthetic_futures)}，期权{len(synthetic_options)}")
                    except Exception as e:
                        self.output(f"本地演示数据生成失败: {e}")
                else:
                    self.output("未获取到任何合约，请检查exchange/future_product/option_product 设置")
                    try:
                        infini_path = getattr(infini, "__file__", "")
                    except Exception:
                        infini_path = ""
                    if infini_path and ("pyStrategy" in infini_path and "pythongo" in infini_path and infini_path.endswith("infini.py")):
                        pass
                    self.output("=== 合约数据加载失败 ===")
                    return
            
            self._debug(f"归一化后合约总数: {len(all_instruments)}")
            self.output(f"=== 归一化后合约总数: {len(all_instruments)} ===")
            
            # 打印前10个合约的详细信息，用于调试
            self.output("=== 开始打印前10个合约样例 ===")
            self._debug(f"=== 前10个合约样例 ===")
            for i, inst in enumerate(all_instruments[:10]):
                exchange = inst.get("ExchangeID", "")
                inst_id = inst.get("InstrumentID", "")
                product_class = inst.get("ProductClass", "")
                self._debug(f"  {i+1}. {exchange}.{inst_id} (ProductClass: {product_class})")
            self.output("=== 打印前10个合约样例完成 ===")

            # 分离期货和期权，并统计分类情况
            futures_cnt = 0
            options_cnt = 0
            unknown_cnt = 0
            unknown_samples: List[str] = []
            product_class_counter: Dict[str, int] = {}
            unknown_instruments: List[str] = []

            for instrument in all_instruments:
                if not isinstance(instrument, dict):
                    continue

                product_class = instrument.get("ProductClass", "")
                product_class_counter[product_class] = product_class_counter.get(product_class, 0) + 1
                inst_id = instrument.get("InstrumentID")
                if not inst_id:
                    continue
                
                # 兼容 Exchange 和ExchangeID 字段
                if 'ExchangeID' not in instrument or not instrument['ExchangeID']:
                    instrument['ExchangeID'] = instrument.get('Exchange', default_exchange)

                try:
                    instrument['ExchangeID'] = self._normalize_exchange_code(instrument.get('ExchangeID', ''))
                except Exception:
                    pass
                
                # 支持多种 ProductClass 格式（1/i 表示期货，2/h 表示期权）
                if product_class in ("1", "i", "I"):  # 期货
                    futures_cnt += 1
                    self.future_instruments.append(instrument)
                    try:
                        sym = inst_id.upper()
                        exch_val = self._normalize_exchange_code(instrument.get("ExchangeID", ""))
                        if sym and exch_val:
                            self.future_symbol_to_exchange[sym] = exch_val
                    except Exception:
                        pass
                elif product_class in ("2", "h", "H"):  # 期权
                    options_cnt += 1
                    # 提取期货代码
                    future_symbol = self._extract_future_symbol(inst_id)
                    self._debug(f"[期权加载] 期权合约: {inst_id}, ProductClass: {product_class}, 交易所: {instrument.get('ExchangeID', '')}, 提取的期货代码: {future_symbol}")
                    if future_symbol:
                        if future_symbol not in self.option_instruments:
                            self.option_instruments[future_symbol] = []
                        self.option_instruments[future_symbol].append(instrument)
                        self._debug(f"[期权加载] 期权已添加到option_instruments: {future_symbol}, 交易所: {instrument.get('ExchangeID', '')}")
                    else:
                        self._debug(f"[期权加载] 期权合约无法提取期货代码: {inst_id}, ProductClass: {product_class}, 交易所: {instrument.get('ExchangeID', '')}")

                    # 初始化期权的 kline_data 占位（以便后续查找不会缺失键）
                    exchange = instrument.get('ExchangeID', '')
                    option_id = inst_id
                    if exchange and option_id:
                        key = f"{exchange}_{option_id}"
                        if key not in self.kline_data:
                            self.kline_data[key] = {'generator': None, 'data': []}
                else:
                    unknown_cnt += 1
                    if self.params.debug_output and len(unknown_samples) < 5:
                        unknown_samples.append(str(product_class))
                        unknown_instruments.append(str(instrument)[:200])

            self._debug(
                f"分类统计:期货: {futures_cnt} 期权: {options_cnt} 未识别 {unknown_cnt}"
            )
            self.output(f"=== 分类完成，期货 {futures_cnt} 期权: {options_cnt} 未识别 {unknown_cnt} ===")
            self.output(f"=== 分类完成，期货 {futures_cnt} 期权: {options_cnt} 未识别 {unknown_cnt} ===")
            if unknown_samples:
                self._debug(f"未识别ProductClass 样例: {unknown_samples}")
                self._debug(f"未识别合约样例 {unknown_instruments}")
            if product_class_counter:
                self._debug(f"ProductClass 直方图 {product_class_counter}")
            
            # 打印所有加载的期货合约列表
            if self.future_instruments:
                self._debug(f"=== 已加载的期货合约列表 ===")
                for i, future in enumerate(self.future_instruments):
                    exchange = future.get("ExchangeID", "")
                    inst_id = future.get("InstrumentID", "")
                    product_class = future.get("ProductClass", "")
                    self._debug(f"  {i+1}. {exchange}.{inst_id} (ProductClass: {product_class})")
                self._debug(f"=== 共加载{len(self.future_instruments)} 个期货合约 ===")

            # 归一化期权分组键：将 HO/MO 映射到IH/IC，IO 映射到IF，避免分组键与期货不一致
            self._normalize_option_group_keys()
            self._log_option_month_pair_coverage()

            # 清理空的期权分组，避免“有键无期权”误判
            try:
                before_groups = len(self.option_instruments)
                self.option_instruments = {k: v for k, v in self.option_instruments.items() if v}
                if len(self.option_instruments) != before_groups:
                    self._debug(f"清理空期权分组: {before_groups} -> {len(self.option_instruments)}")
            except Exception:
                pass

            # 仅保留指定月与指定下月的期货（可配置）
            try:
                filter_specified_futures = self._resolve_subscribe_flag(
                    "subscribe_only_specified_month_futures",
                    "subscribe_only_current_next_futures",
                    False
                )
                if filter_specified_futures:
                    before_fut = len(self.future_instruments)
                    self.future_instruments = [
                        f for f in self.future_instruments
                        if self._is_symbol_specified_or_next(str(f.get("InstrumentID", "")).upper())
                    ]
                    self._debug(f"过滤期货至指定月/指定下月: {before_fut} 到{len(self.future_instruments)}")
                    # 同步裁剪期权分组，移除已过滤期货的期权
                    allowed = {self._normalize_future_id(str(f.get("InstrumentID", ""))) for f in self.future_instruments}
                    if allowed:
                        before_opt_groups = len(self.option_instruments)
                        self.option_instruments = {k: v for k, v in self.option_instruments.items() if self._normalize_future_id(k) in allowed}
                        self._debug(f"过滤期权分组至指定月/指定下月: {before_opt_groups} 到{len(self.option_instruments)}")
            except Exception as e:
                self._debug(f"指定月/指定下月过滤失败: {e}")

            # 若期权拉取失败导致缺链，则剔除对应期货，避免“有期货无期权”的中间态
            try:
                if getattr(self.params, "subscribe_options", True):
                    before_fut = len(self.future_instruments)
                    self.future_instruments = [
                        f for f in self.future_instruments
                        if self._has_option_for_product(self._extract_product_code(str(f.get("InstrumentID", ""))))
                    ]
                    if len(self.future_instruments) != before_fut:
                        self._debug(f"剔除无期权链期货: {before_fut} -> {len(self.future_instruments)}")
            except Exception as e:
                self._debug(f"剔除无期权链期货失败: {e}")

            if self._option_fetch_failures:
                self.output(f"[警告] 期权合约加载失败品种: {sorted(self._option_fetch_failures)}")

            self.data_loaded = True
            self.output("=== 合约数据加载完成 ===")
            
            # 打印期权映射关系，用于调试
            self._debug(f"=== 期权映射关系 ===")
            for future_symbol, options in self.option_instruments.items():
                self._debug(f"  {future_symbol}: {len(options)} 个期权")
                if len(options) <= 3:
                    for opt in options:
                        self._debug(f"    - {opt.get('ExchangeID', '')}.{opt.get('InstrumentID', '')}")
            # 打印期货代码和期权代码的对应关系
            self._debug(f"=== 期货代码和期权代码对应关系 ===")
            for future in self.future_instruments:
                exchange = future.get("ExchangeID", "")
                future_id = future.get("InstrumentID", "")
                if future_id:
                    future_id_upper = future_id.upper()
                    options = self.option_instruments.get(future_id_upper, [])
                    if options:
                        self._debug(f"  期货: {exchange}.{future_id} -> 期权数量: {len(options)}")
                        for opt in options[:3]:
                            opt_id = opt.get('InstrumentID', '')
                            self._debug(f"    期权: {exchange}.{opt_id}")
            # 统计各交易所的期权数量
            self._debug(f"=== 各交易所期权统计 ===")
            exchange_stats = {}
            for future_symbol, options in self.option_instruments.items():
                for opt in options:
                    exchange = opt.get('ExchangeID', '')
                    if exchange not in exchange_stats:
                        exchange_stats[exchange] = 0
                    exchange_stats[exchange] += 1
            for exchange, count in exchange_stats.items():
                self._debug(f"  {exchange}: {count} 个期权")
        except Exception as e:
            self.output(f"加载合约失败: {e}\n{traceback.format_exc()}")
            self.output("=== 合约数据加载失败 ===")

    def _normalize_option_group_keys(self) -> None:
        """将已分组的期权键按品种映射到期货键（IO→IF，HO→IH，MO→IC，EO→IM）"""
        try:
            if not self.option_instruments:
                return
            prefix_map = {"IO": "IF", "HO": "IH", "MO": "IC", "EO": "IM"}
            try:
                extra_map = getattr(self.params, "option_prefix_map", {}) or {}
                for k, v in extra_map.items():
                    if k and v:
                        prefix_map[str(k).upper()] = str(v).upper()
            except Exception:
                pass
            normalized: Dict[str, List[Dict[str, Any]]] = {}

            for key, opts in self.option_instruments.items():
                # 去交易所前缀/分隔符、大小写归一
                base_key = self._normalize_future_id(str(key))
                # 先处理郑商所一位年格式
                try:
                    base_key = self._expand_czce_year_month(base_key)
                except Exception:
                    pass
                target_key = base_key

                # 股指期权映射到期货前缀（支持两位字母+两位年+月份）
                m_full = re.match(r"([A-Za-z]{2,})(\d{2})(\d{1,2})", base_key)
                if m_full:
                    prefix = m_full.group(1).upper()
                    mapped = prefix_map.get(prefix)
                    if mapped:
                        target_key = f"{mapped}{m_full.group(2)}{m_full.group(3)}"
                else:
                    # 兜底：CZCE 一位年格式仍保持归一化后的 base_key
                    pass

                if target_key not in normalized:
                    normalized[target_key] = []
                normalized[target_key].extend(opts)

            if normalized != self.option_instruments:
                self.option_instruments = normalized
                self._debug("期权分组键已归一化(IO→IF, HO→IH, MO→IC, EO→IM)")
            self._option_groups_normalized = True
        except Exception:
            pass

    def _log_option_month_pair_coverage(self) -> None:
        """检查指定月/指定下月期权分组是否齐全，输出缺失品种。"""
        try:
            eff_map = self._get_effective_month_mapping()
            if not eff_map:
                return
            missing = []
            for prod, pair in eff_map.items():
                if not isinstance(pair, list) or len(pair) < 2:
                    continue
                cm = self._normalize_future_id(pair[0])
                nm = self._normalize_future_id(pair[1])
                cm_ok = cm in self.option_instruments and bool(self.option_instruments.get(cm))
                nm_ok = nm in self.option_instruments and bool(self.option_instruments.get(nm))
                if not cm_ok or not nm_ok:
                    missing.append((prod, cm if not cm_ok else "", nm if not nm_ok else ""))
            if missing:
                self.output(f"[警告] 期权分组缺失(指定月/指定下月): {missing}")
        except Exception:
            pass

    def _get_commodity_option_exchange(self, product_code: str) -> Optional[str]:
        """根据品种代码获取商品期权对应的交易所"""
        PRODUCT_EXCHANGE_MAP = {
            # 有色金属 - SHFE
            "CU": "SHFE", "AL": "SHFE", "ZN": "SHFE",
            "NI": "SHFE", "SN": "SHFE", "PB": "SHFE",
            # 黑色金属 - SHFE
            "RB": "SHFE", "HC": "SHFE", "WR": "SHFE", "SS": "SHFE",
            # 贵金属 - SHFE
            "AU": "SHFE", "AG": "SHFE",
            # 能源化工 - SHFE
            "FU": "SHFE", "BU": "SHFE", "RU": "SHFE", "SP": "SHFE",
            # 能源化工 - INE
            "SC": "INE", "LU": "INE", "NR": "INE", "BC": "INE", "BR": "INE",
            # 农产品 - DCE
            "M": "DCE", "Y": "DCE", "A": "DCE", "P": "DCE",
            "C": "DCE", "CS": "DCE", "JD": "DCE", "L": "DCE",
            "V": "DCE", "PP": "DCE", "EB": "DCE", "EG": "DCE",
            "PG": "DCE", "LH": "DCE", "FB": "DCE", "BB": "DCE",
            # 钢铁 - DCE
            "J": "DCE", "JM": "DCE", "I": "DCE",
            # 化工 - DCE
            "MA": "DCE", "FG": "DCE", "TA": "DCE",
            # 农产品 - CZCE
            "CF": "CZCE", "SR": "CZCE", "OI": "CZCE", "RM": "CZCE",
            "ZC": "CZCE", "WH": "CZCE", "PM": "CZCE", "RI": "CZCE",
            "JR": "CZCE", "LR": "CZCE", "SF": "CZCE", "SM": "CZCE",
            "UR": "CZCE", "SA": "CZCE", "PK": "CZCE", "AP": "CZCE",
            "CY": "CZCE", "CJ": "CZCE", "PF": "CZCE",
        }
        exch = PRODUCT_EXCHANGE_MAP.get(product_code.upper())
        if exch:
            return exch
        # 动态回退：尝试使用已加载期货合约的交易所
        try:
            prod = str(product_code).upper()
            if prod in self.future_symbol_to_exchange:
                return self.future_symbol_to_exchange.get(prod)
            for sym, ex in self.future_symbol_to_exchange.items():
                if str(sym).upper().startswith(prod):
                    return ex
        except Exception:
            pass
        return None

    def _is_commodity_option(self, product_code: str) -> bool:
        """判断是否为商品期权（非股指期权）"""
        index_options = {"IO", "HO", "MO", "EO"}
        try:
            extra = getattr(self.params, "index_option_prefixes", None)
            if extra:
                index_options = index_options.union({str(x).upper() for x in extra})
        except Exception:
            pass
        return product_code.upper() not in index_options

    def _has_option_for_product(self, product_code: str) -> bool:
        """检测该期货品种是否存在已加载的期权分组"""
        try:
            prefix = str(product_code).upper()
            for k, v in self.option_instruments.items():
                try:
                    if str(k).upper().startswith(prefix) and v:
                        return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    def _is_index_option(self, product_code: str) -> bool:
        """判断是否为股指期权"""
        index_options = {"IO", "HO", "MO", "EO"}
        try:
            extra = getattr(self.params, "index_option_prefixes", None)
            if extra:
                index_options = index_options.union({str(x).upper() for x in extra})
        except Exception:
            pass
        return product_code.upper() in index_options

    def _build_option_groups_by_option_prefix(self) -> Dict[str, List[Dict[str, Any]]]:
        """按期权品种前缀+年月构建分组（如 IO2601、HO2601、M2505）。
        默认处理股指期权交易所，可通过参数 option_group_exchanges 扩展。
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        try:
            allowed_ex = {"CFFEX"}
            try:
                extra_ex = getattr(self.params, "option_group_exchanges", None)
                if extra_ex:
                    allowed_ex = {str(x).upper() for x in extra_ex}
                else:
                    allowed_ex = {"CFFEX", "INE"}
            except Exception:
                allowed_ex = {"CFFEX", "INE"}

            for _, options in self.option_instruments.items():
                for opt in options:
                    exchange = str(opt.get('ExchangeID', '')).strip().upper()
                    if exchange not in allowed_ex:
                        continue
                    oid = str(opt.get('InstrumentID', '')).upper()
                    m = re.search(r"([A-Za-z]{1,})(\d{2})(\d{1,2})", oid)
                    if not m:
                        # 尝试 CZCE 样式：一位年+两位月
                        m2 = re.search(r"([A-Za-z]{1,})(\d)(\d{2})", oid)
                        if not m2:
                            continue
                        prefix = m2.group(1).upper()
                        yy = m2.group(2)
                        mm = m2.group(3)
                        group_id = f"{prefix}{yy}{mm}"
                    else:
                        prefix = m.group(1).upper()
                        yy = m.group(2)
                        mm = m.group(3)
                        group_id = f"{prefix}{yy}{mm}"
                    groups.setdefault(group_id, []).append(opt)
        except Exception:
            pass
        return groups

    def _build_shfe_option_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """按SHFE期权品种前缀+年月构建分组（如 CU2601、RB2601）。
        专门处理SHFE交易所期权，两位年格式。
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        try:
            shfe_options_count = 0
            for _, options in self.option_instruments.items():
                for opt in options:
                    # 只处理SHFE交易所的期权，忽略大小写/空格
                    exchange = str(opt.get('ExchangeID', '')).strip().upper()
                    if exchange != 'SHFE':
                        continue
                    shfe_options_count += 1
                    oid = str(opt.get('InstrumentID', '')).upper()
                    m = re.search(r"([A-Za-z]{1,2})(\d{2})(\d{1,2})", oid)
                    if m:
                        prefix = m.group(1).upper()
                        yy = m.group(2)
                        mm = m.group(3)
                        group_id = f"{prefix}{yy}{mm}"
                        groups.setdefault(group_id, []).append(opt)
            self._debug(f"[SHFE期权分组] 共处理{shfe_options_count}个SHFE期权，生成{len(groups)}个分组")
        except Exception:
            pass
        return groups

    def _build_dce_option_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """按DCE期权品种前缀+年月构建分组（如 M2601、Y2601）。
        专门处理DCE交易所期权，两位年格式。
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        try:
            dce_options_count = 0
            for _, options in self.option_instruments.items():
                for opt in options:
                    # 只处理DCE交易所的期权，忽略大小写/空格
                    exchange = str(opt.get('ExchangeID', '')).strip().upper()
                    if exchange != 'DCE':
                        continue
                    dce_options_count += 1
                    oid = str(opt.get('InstrumentID', '')).upper()
                    m = re.search(r"([A-Za-z]{1,2})(\d{2})(\d{1,2})", oid)
                    if m:
                        prefix = m.group(1).upper()
                        yy = m.group(2)
                        mm = m.group(3)
                        group_id = f"{prefix}{yy}{mm}"
                        groups.setdefault(group_id, []).append(opt)
            self._debug(f"[DCE期权分组] 共处理{dce_options_count}个DCE期权，生成{len(groups)}个分组")
        except Exception:
            pass
        return groups

    def _build_czce_option_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """按CZCE期权品种前缀+年月构建分组（如 SR509、CF509）。
        专门处理CZCE交易所期权，一位年格式。
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        try:
            czce_options_count = 0
            for _, options in self.option_instruments.items():
                for opt in options:
                    # 只处理CZCE交易所的期权，忽略大小写/空格
                    exchange = str(opt.get('ExchangeID', '')).strip().upper()
                    if exchange != 'CZCE':
                        continue
                    czce_options_count += 1
                    oid = str(opt.get('InstrumentID', '')).upper()
                    m = re.search(r"([A-Za-z]{1,2})(\d)(\d{2})", oid)
                    if m:
                        prefix = m.group(1).upper()
                        yy = m.group(2)
                        mm = m.group(3)
                        group_id = f"{prefix}{yy}{mm}"
                        groups.setdefault(group_id, []).append(opt)
            self._debug(f"[CZCE期权分组] 共处理{czce_options_count}个CZCE期权，生成{len(groups)}个分组")
        except Exception:
            pass
        return groups

    def _extract_future_symbol(self, option_symbol: str) -> Optional[str]:
        """从期权代码提取期货代码，支持商品期权的一位品种前缀（如 M/C/P/Y 等）"""
        if not option_symbol:
            return None

        # 将期权品种前缀映射到对应的期货品种前缀（股指特例）
        prefix_map = {"IO": "IF", "HO": "IH", "MO": "IC", "EO": "IM"}

        # 常见格式：字母>=1个 + 两位年份 + 月份(1-2位)，如 IO2601、M2505、c2503
        m = re.search(r"([A-Za-z]{1,})[-_]?(\d{2})[-_]?(\d{1,2})", option_symbol)
        if m:
            product_code = m.group(1).upper()
            year_part = m.group(2)
            month_part = m.group(3)

            # 排除单独的C/P误匹配（纯看涨/看跌标记而非品种）
            if product_code in {"C", "P"}:
                # 检查是否为单独的C/P标记（没有年月）
                if not (year_part and month_part):
                    return None

            # 月份有效性校验
            try:
                month_val = int(month_part)
                if not (1 <= month_val <= 12):
                    return None
            except Exception:
                return None

            mapped_prefix = prefix_map.get(product_code, product_code)
            return self._normalize_future_id(f"{mapped_prefix}{year_part}{str(month_part).zfill(2)}")

        # CZCE 等一位年份+ 两位月份格式（如 SR509C、CF601P）
        m_czce = re.search(r"([A-Za-z]{1,})[-_]?(\d)(\d{2})", option_symbol)
        if m_czce:
            product_code = m_czce.group(1).upper()
            year_digit = m_czce.group(2)
            month_part = m_czce.group(3)

            # 排除单独的C/P误匹配（纯看涨/看跌标记而非品种）
            # 只有当product_code是单个字母C或P，且后面没有数字时，才认为是看涨/看跌标记
            # 例如：C509C -> product_code = "C"（玉米品种，不是看涨/看跌标记）
            # 例如：P509C -> product_code = "P"（棕榈油品种，不是看涨/看跌标记）
            if product_code in {"C", "P"}:
                # 检查是否为单独的C/P标记（没有年月）
                if not (year_digit and month_part):
                    return None
                # 如果有年月，说明是品种代码（如C509C是玉米期权）
                # 继续处理，不返回None

            try:
                month_val = int(month_part)
                if not (1 <= month_val <= 12):
                    return None
            except Exception:
                return None

            mapped_prefix = prefix_map.get(product_code, product_code)
            try:
                y = int(year_digit)
                cur_y2 = datetime.now().year % 100
                decade = (cur_y2 // 10) * 10
                if y > (cur_y2 % 10):
                    decade = max(0, decade - 10)
                y2 = decade + y
            except Exception:
                y2 = int(year_digit)
            return self._normalize_future_id(f"{mapped_prefix}{y2:02d}{str(month_part).zfill(2)}")

        # 兜底：匹配连续字母数字串（允许一位品种代码），再做映射
        m2 = re.search(r"([A-Za-z]{1,}\d{2}\d{1,2})", option_symbol)
        if m2:
            symbol = m2.group(1).upper()
            for prefix, future_prefix in prefix_map.items():
                if symbol.startswith(prefix):
                    symbol = future_prefix + symbol[len(prefix):]
                    break
            return self._normalize_future_id(symbol)
        return None

    def _extract_product_code(self, instrument_id: str) -> str:
        """提取期货品种代码"""
        if not instrument_id:
            return ""
        normalized = self._normalize_future_id(str(instrument_id))
        match = re.match(r"^([A-Z]+)", normalized)
        return match.group(1).upper() if match else ""

    def _cleanup_kline_cache_for_symbol(self, instrument_id: str) -> None:
        """清理某合约的K线缓存，避免无期权品种持续占用内存"""
        try:
            inst_upper = self._normalize_future_id(instrument_id)
            if not inst_upper:
                return
            keys_to_remove = []
            for key in list(self.kline_data.keys()):
                try:
                    if key.endswith(f"_{inst_upper}"):
                        keys_to_remove.append(key)
                except Exception:
                    continue
            for key in keys_to_remove:
                try:
                    del self.kline_data[key]
                except Exception:
                    pass
        except Exception:
            pass

    def _resolve_subscribe_flag(self, primary: str, legacy: str, default: bool = False) -> bool:
        """解析订阅过滤参数优先级：优先使用参数表显式配置，其次使用对象属性，最后回退默认值。"""
        try:
            overrides = getattr(self, "_param_override_cache", None) or self._load_param_table()
            params_block = overrides.get("params") if isinstance(overrides, dict) else None
            if isinstance(params_block, dict):
                if primary in params_block:
                    return bool(params_block.get(primary))
                if legacy in params_block:
                    return bool(params_block.get(legacy))
            if hasattr(self.params, primary):
                return bool(getattr(self.params, primary))
            if hasattr(self.params, legacy):
                return bool(getattr(self.params, legacy))
        except Exception:
            pass
        return bool(default)

    def _normalize_exchange_code(self, exchange: str) -> str:
        """规范化交易所代码，兼容别名。"""
        try:
            exch = (exchange or "").strip().upper()
            alias = {
                "CCFX": "CFFEX",
                "CFFE": "CFFEX",
                "SHFE_TEST": "SHFE",
                "DCE_TEST": "DCE",
                "CZCE_TEST": "CZCE",
            }
            return alias.get(exch, exch)
        except Exception:
            return (exchange or "").strip().upper()

    def _is_instrument_allowed(self, instrument_id: str, exchange: str = "") -> bool:
        """判断某合约是否允许进入主逻辑（需要合约已加载、在指定月、且有期权链）。"""
        try:
            if not instrument_id:
                return False
            if not getattr(self, "data_loaded", False) or not getattr(self, "_instruments_ready", False):
                return False
            inst_upper = self._normalize_future_id(instrument_id)
            if not inst_upper:
                return False

            # 期货合约
            if inst_upper in self.future_symbol_to_exchange or self._is_real_month_contract(inst_upper):
                if not self._is_symbol_current(inst_upper):
                    return False
                prod = self._extract_product_code(inst_upper)
                if prod and not self._has_option_for_product(prod):
                    return False
                return True

            # 期权合约 -> 映射到期货
            fut_symbol = self._extract_future_symbol(inst_upper)
            if fut_symbol:
                if not self._is_symbol_current(fut_symbol):
                    return False
                prod = self._extract_product_code(fut_symbol)
                if prod and not self._has_option_for_product(prod):
                    return False
                return True

            return False
        except Exception:
            return False

    def _should_forward_to_position_manager(self, instrument_id: str, exchange: str = "") -> bool:
        """防止回报/行情通过后门进入平仓管理器（无期权链则阻断）。"""
        try:
            return self._is_instrument_allowed(instrument_id, exchange)
        except Exception:
            return False

    def _expand_czce_year_month(self, symbol: str) -> str:
        """将 CZCE 一位年份格式扩展为两位年份，兼容三/四位数字尾巴：
        - SR603 -> SR2603（1位年 + 2位月）
        - SR6005 -> SR2605（1位年 + 3位数字，取末两位为月）
        对两位年份样式若年份过远也尝试回落到一位年份推断。
        """
        try:
            # 已是两位年份且月份有效、年份不过远则原样返回
            m_two = re.match(r"^([A-Z]{1,})(\d{2})(\d{2})$", symbol)
            if m_two:
                prefix, yy, mm = m_two.group(1), int(m_two.group(2)), int(m_two.group(3))
                if 1 <= mm <= 12:
                    cur_y2 = datetime.now().year % 100
                    try:
                        future_window = int(getattr(self.params, "czce_year_future_window", 10) or 10)
                        past_window = int(getattr(self.params, "czce_year_past_window", 10) or 10)
                    except Exception:
                        future_window = 10
                        past_window = 10
                    # 若年份在可接受窗口内，则认为已是标准格式
                    if (cur_y2 - past_window) <= yy <= (cur_y2 + future_window):
                        return symbol
            # 一位年份 + 两位月份（典型 SR603）
            m_three = re.match(r"^([A-Z]{1,})(\d)(\d{2})$", symbol)
            if m_three:
                prefix, y_digit, mm = m_three.group(1), int(m_three.group(2)), m_three.group(3)
                month = int(mm)
                if 1 <= month <= 12:
                    cur_y2 = datetime.now().year % 100
                    decade = (cur_y2 // 10) * 10
                    if y_digit > (cur_y2 % 10):
                        decade = max(0, decade - 10)
                    y2 = decade + y_digit
                    return f"{prefix}{y2:02d}{mm}"
            # 一位年份 + 三位数字（如 SR6005、MA6005，末两位为月份）
            m_four = re.match(r"^([A-Z]{1,})(\d)(\d{3})$", symbol)
            if m_four:
                prefix, y_digit, tail = m_four.group(1), int(m_four.group(2)), m_four.group(3)
                mm = tail[-2:]
                month = int(mm)
                if 1 <= month <= 12:
                    cur_y2 = datetime.now().year % 100
                    decade = (cur_y2 // 10) * 10
                    if y_digit > (cur_y2 % 10):
                        decade = max(0, decade - 10)
                    y2 = decade + y_digit
                    return f"{prefix}{y2:02d}{mm}"
            return symbol
        except Exception:
            return symbol


    def _normalize_future_id(self, instrument_id: str) -> str:
        """Normalize instrument ID by stripping exchange prefixes, removing separators, uppercasing, and expanding CZCE 一位年份格式。"""
        try:
            if not instrument_id:
                return ""
            normalized = str(instrument_id).upper()
            if "." in normalized:
                normalized = normalized.split(".")[-1]
            # 去除常见分隔符：空格/下划线
            normalized = re.sub(r"[\s_]+", "", normalized)
            normalized = self._expand_czce_year_month(normalized)
            return normalized
        except Exception:
            return ""

    def _normalize_instruments(self, res: Any) -> List[dict]:
        """将infini 返回的合约数据转换为统一的dict 列表"""
        # 首先检查是否为 None
        if res is None:
            return []

        normalized: List[dict] = []
        try:
            # 如果是单个对象，转为列表
            if not isinstance(res, (list, tuple, set)):
                res = [res]

            for obj in res:
                # 如果嵌套可迭代（例如列表套列表），逐个处理
                if isinstance(obj, (list, tuple, set)):
                    for sub in obj:
                        inst = self._to_instrument_dict(sub)
                        if inst and inst.get("InstrumentID"):
                            normalized.append(inst)
                    continue

                inst = self._to_instrument_dict(obj)
                if inst and inst.get("InstrumentID"):
                    normalized.append(inst)
        except Exception:
            # 如果处理过程中出现任何异常，返回空列表
            return []
        return normalized

    def _to_instrument_dict(self, obj: Any) -> Optional[dict]:
        """兼容 dict / InstrumentData 等对象，提取必要字段"""
        try:
            if obj is None:
                return None
            if isinstance(obj, dict):
                # 标准化可选字段名
                inst = dict(obj)
                try:
                    # 规范 StrikePrice 和OptionType
                    sp = inst.get("StrikePrice") or inst.get("strike_price") or inst.get("_strike_price") or inst.get("strikePrice")
                    if sp is not None:
                        inst["StrikePrice"] = sp
                    ot = inst.get("OptionType") or inst.get("option_type") or inst.get("call_put") or inst.get("_option_type") or inst.get("option_kind") or inst.get("_option_kind")
                    if ot is not None:
                        # 归一化为 'C' 或'P'
                        s = str(ot).upper()
                        if s in ("C", "CALL", "1"):
                            inst["OptionType"] = "C"
                        elif s in ("P", "PUT", "2"):
                            inst["OptionType"] = "P"
                        else:
                            inst["OptionType"] = s
                except Exception:
                    pass
                return inst

            # InstrumentData 风格对象
            def _safe_attr(o, name, default=""):
                try:
                    val = getattr(o, name, default)
                    return val if val is not None else default
                except Exception:
                    return default

            instrument_id = _safe_attr(obj, "instrument_id") or _safe_attr(obj, "_instrument_id")
            exchange_id = _safe_attr(obj, "exchange") or _safe_attr(obj, "_exchange") or _safe_attr(obj, "exchange_id")
            product_class_raw = _safe_attr(obj, "_product_type") or _safe_attr(obj, "product_type") or _safe_attr(obj, "product_class") or ""
            strike_price_val = _safe_attr(obj, "strike_price", None) or _safe_attr(obj, "_strike_price", None)
            option_type_val = _safe_attr(obj, "option_type", None) or _safe_attr(obj, "_option_type", None) or _safe_attr(obj, "call_put", None) or _safe_attr(obj, "option_kind", None)

            # 将product_type 映射为原始代码1/2（期货/期权），尽量保持与ProductClass 兼容
            product_class = ""
            upper_pc = str(product_class_raw).upper()
            if upper_pc in ("1", "FUTURE", "FUTURES", "F", "I"):
                product_class = "1"
            elif upper_pc in ("2", "OPTION", "OPTIONS", "O", "OPT", "OPTN", "T", "9", "H"):
                product_class = "2"
            elif "期货" in str(product_class_raw):
                product_class = "1"
            elif "期权" in str(product_class_raw):
                product_class = "2"
            else:
                # 兜底：通过属性或合约代码判断是否为期权
                option_attrs = (
                    "option_type",
                    "_option_type",
                    "option_kind",
                    "_option_kind",
                    "strike_price",
                    "_strike_price",
                    "call_put",
                    "_call_put"
                )
                has_option_attr = any(
                    _safe_attr(obj, attr, None) not in (None, "") for attr in option_attrs
                )
                if has_option_attr:
                    product_class = "2"
                elif instrument_id and re.search(r"[CP][-_]?\d", str(instrument_id)):
                    product_class = "2"
                else:
                    product_class = "1"

            return {
                "ExchangeID": exchange_id,
                "InstrumentID": instrument_id,
                "ProductClass": product_class,
                # 可选字段：若存在则携带，便于虚值判断
                **({"StrikePrice": strike_price_val} if strike_price_val not in (None, "") else {}),
                **({"OptionType": ("C" if str(option_type_val).upper() in ("C", "CALL", "1") else ("P" if str(option_type_val).upper() in ("P", "PUT", "2") else str(option_type_val).upper()))} if option_type_val not in (None, "") else {}),
            }
        except Exception:
            return None

    def _extract_month(self, instrument_id: str) -> Optional[int]:
        """提取合约月份（1-12）。优先识别CZCE样式（一位年+两位月），避免SR611被误解析为“61年1月”。"""
        s = self._normalize_future_id(instrument_id)
        # 优先：CZCE（一位年+两位月），如 SR611 → 月份11
        match = re.search(r"[A-Za-z]+(\d)(\d{2})", s)
        if match:
            try:
                month = int(match.group(2))
                if 1 <= month <= 12:
                    return month
            except Exception:
                pass
        # 其次：通用/股指（两位年 + 1-2位月），如 IF2601 → 月份1/01
        match = re.search(r"[A-Za-z]+(\d{2})(\d{1,2})", s)
        if match:
            try:
                month = int(match.group(2))
                if 1 <= month <= 12:
                    return month
            except Exception:
                pass
        return None
    def _extract_year(self, instrument_id: str) -> Optional[int]:
        """提取合约年份。优先识别CZCE样式（一位年+两位月）。"""
        s = self._normalize_future_id(instrument_id)
        # 优先：CZCE（一位年+两位月），如 SR611 → 年份个位6 → 2026
        match = re.search(r"[A-Za-z]+(\d)(\d{2})", s)
        if match:
            try:
                year_digit = int(match.group(1))
                now = datetime.now()
                current_year_last_digit = now.year % 10
                if current_year_last_digit >= year_digit:
                    return now.year - (current_year_last_digit - year_digit)
                else:
                    return now.year - 10 + (year_digit - current_year_last_digit)
            except Exception:
                pass
        # 其次：通用/股指（两位年 + 1-2位月），如 IF2601 → 2026
        match = re.search(r"[A-Za-z]+(\d{2})(\d{1,2})", s)
        if match:
            try:
                year = int(match.group(1))
                return 2000 + year if year < 50 else 1900 + year
            except Exception:
                pass
        return None

    def _is_real_month_contract(self, instrument_id: str) -> bool:
        """判断是否为真实的月份合约代码，过滤掉 Main/Weighted 等综合合约。支持CFFEX/通用两位年+1-2位月，以及CZCE 一位年+两位月"""
        if not instrument_id:
            return False
        raw = str(instrument_id).upper()
        s = self._normalize_future_id(raw)
        if not s:
            return False
        # 过滤明显的综合/加权/主连等标识
        bad_tags = (
            "MAIN", "INDEX", "WEIGHT", "WEIGHTED", "HOT", "CONT", "CONTINUOUS",
            "NEAR", "THIS", "NEXT", "000", "888", "999"
        )
        if any(tag in raw for tag in bad_tags) or "_" in raw or "-" in raw:
            return False
        # 通用/CFFEX：字母>=1个 + 两位年+ 一/二位月
        if re.match(r"^[A-Z]{1,}\d{2}\d{1,2}$", s):
            return True
        # CZCE：字母>=1个 + 一位年(年份个位) + 两位月
        if re.match(r"^[A-Z]{1,}\d\d{2}$", s):
            return True
        return False

    def _get_debug_month_mapping(self) -> Dict[str, Tuple[str, str]]:
        """调试模式用：从参数表中提取品种到(指定月,指定下月)的映射。
        支持结构：
        - 顶层或 params 下的 {month_mapping|month_map|commodity_months|product_months}
        - 值可为 {current/current_month, next/next_month} 字典，或 [current,next] 列表/元组，或 "current|next" 字符串。
        返回的键为大写品种代码；值为(指定月,指定下月)大写合约。
        """
        mapping: Dict[str, Tuple[str, str]] = {}
        try:
            overrides = getattr(self, "_param_override_cache", None) or self._load_param_table()
            def pick_container(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                if not isinstance(d, dict):
                    return None
                for key in ("month_mapping", "month_map", "commodity_months", "product_months"):
                    val = d.get(key)
                    # 支持字符串存储的JSON
                    if isinstance(val, str):
                        try:
                            parsed = json.loads(val)
                            val = parsed
                        except Exception:
                            val = None
                    if isinstance(val, dict):
                        return val
                return None
            top = pick_container(overrides) if isinstance(overrides, dict) else None
            params_block = None
            if isinstance(overrides, dict):
                p = overrides.get("params")
                if isinstance(p, dict):
                    params_block = pick_container(p)
            orig_block = None
            if isinstance(overrides, dict):
                od = overrides.get("original_defaults")
                if isinstance(od, dict):
                    orig_block = pick_container(od)
            src = None
            for cand in (params_block, orig_block, top):
                if isinstance(cand, dict) and cand:
                    src = cand
                    break
            if not isinstance(src, dict):
                return {}
            for k, v in src.items():
                try:
                    prod = str(k).upper()
                    cm: Optional[str] = None
                    nm: Optional[str] = None
                    if isinstance(v, dict):
                        cm = str(v.get("current") or v.get("current_month") or v.get("cm") or "").upper()
                        nm = str(v.get("next") or v.get("next_month") or v.get("nm") or "").upper()
                    elif isinstance(v, (list, tuple)) and len(v) >= 2:
                        cm = str(v[0]).upper()
                        nm = str(v[1]).upper()
                    elif isinstance(v, str) and v.strip():
                        parts = re.split(r"[,\|;/\s]+", v.strip())
                        if len(parts) >= 2:
                            cm = parts[0].upper()
                            nm = parts[1].upper()
                    if cm and nm:
                        mapping[prod] = (cm, nm)
                except Exception:
                    pass
            return mapping
        except Exception:
            return {}

    def _align_month_mapping_to_loaded_futures(self) -> None:
        """读取已加载期货列表，对调试映射的指定月/指定下月格式进行对齐修正（不改生产参数）。
        逻辑：
        - 若映射中的合约ID已存在于 loaded futures，原样保留。
        - 若不存在，尝试转换为 CZCE 一位年份格式（如 SR2601 -> SR601）。
        - 若仍不存在，尝试去零的单月位格式（如 01 -> 1）。
        - 将修正后的映射写入内存缓存 _param_override_cache（若存在 params.month_mapping 则更新该处，否则更新顶层 month_mapping）。
        """
        try:
            # 构建已加载期货ID集合
            loaded: Set[str] = set()
            prod_to_samples: Dict[str, Set[str]] = {}
            for f in self.future_instruments:
                inst_raw = str(f.get("InstrumentID", "")).upper()
                inst = self._normalize_future_id(inst_raw)
                if not inst:
                    continue
                loaded.add(inst)
                # 同时保留原始格式，便于直接匹配
                loaded.add(inst_raw)
                m = re.match(r"^([A-Z]+)", inst)
                if m:
                    p = m.group(1)
                    prod_to_samples.setdefault(p, set()).add(inst)

            mapping = self._get_debug_month_mapping()
            if not mapping:
                return
            changed = {}

            def to_czce_style(code: str) -> Optional[str]:
                try:
                    p = self._extract_product_code(code).upper()
                    y = self._extract_year(code)
                    mth = self._extract_month(code)
                    if p and y and mth is not None:
                        return f"{p}{str(y)[-1]}{mth:02d}".upper()
                    return None
                except Exception:
                    return None

            def to_single_month_style(code: str) -> Optional[str]:
                try:
                    p = self._extract_product_code(code).upper()
                    yy = re.search(r"[A-Z]+(\d{2})(\d{1,2})", code)
                    if yy:
                        y2 = yy.group(1)
                        mth = int(yy.group(2))
                        return f"{p}{y2}{mth:02d}".upper()
                    cz = re.search(r"[A-Z]+(\d)(\d{2})", code)
                    if cz:
                        y1 = cz.group(1)
                        mth = int(cz.group(2))
                        return f"{p}{y1}{mth:02d}".upper()
                    return None
                except Exception:
                    return None

            for prod, (cm, nm) in mapping.items():
                cm_u = str(cm).upper()
                nm_u = str(nm).upper()
                new_cm = cm_u
                new_nm = nm_u
                # 已存在则保留
                if cm_u not in loaded:
                    alt = to_czce_style(cm_u)
                    if alt and alt in loaded:
                        new_cm = alt
                    else:
                        alt2 = to_single_month_style(cm_u)
                        if alt2 and alt2 in loaded:
                            new_cm = alt2
                if nm_u not in loaded:
                    alt = to_czce_style(nm_u)
                    if alt and alt in loaded:
                        new_nm = alt
                    else:
                        alt2 = to_single_month_style(nm_u)
                        if alt2 and alt2 in loaded:
                            new_nm = alt2
                if (new_cm != cm_u) or (new_nm != nm_u):
                    changed[prod] = (cm_u, nm_u, new_cm, new_nm)

            if not changed:
                return

            # 更新缓存中的映射
            overrides = getattr(self, "_param_override_cache", None) or self._load_param_table()
            updated = False
            def update_container(d: Dict[str, Any]) -> bool:
                if not isinstance(d, dict):
                    return False
                mm = d.get("month_mapping")
                if isinstance(mm, str):
                    try:
                        mm = json.loads(mm)
                    except Exception:
                        mm = {}
                if not isinstance(mm, dict):
                    mm = {}
                    d["month_mapping"] = mm
                try:
                    for prod, (old_cm, old_nm, new_cm, new_nm) in changed.items():
                        val = mm.get(prod)
                        if isinstance(val, dict):
                            val_keys = {k.lower(): k for k in val.keys()}
                            k_cm = val_keys.get("current") or val_keys.get("current_month") or val_keys.get("cm")
                            k_nm = val_keys.get("next") or val_keys.get("next_month") or val_keys.get("nm")
                            if k_cm:
                                val[k_cm] = new_cm
                            if k_nm:
                                val[k_nm] = new_nm
                            else:
                                val["next_month"] = new_nm
                        elif isinstance(val, (list, tuple)):
                            mm[prod] = [new_cm, new_nm]
                        elif isinstance(val, str):
                            mm[prod] = f"{new_cm}|{new_nm}"
                        else:
                            mm[prod] = [new_cm, new_nm]
                    return True
                except Exception:
                    return False

            if isinstance(overrides, dict):
                # 优先 params.month_mapping
                params_block = overrides.get("params") if isinstance(overrides.get("params"), dict) else None
                if isinstance(params_block, dict):
                    if update_container(params_block):
                        updated = True
                if (not updated) and update_container(overrides):
                    updated = True
            if updated:
                self._param_override_cache = overrides
                # 输出简要修正日志
                try:
                    for prod, (old_cm, old_nm, new_cm, new_nm) in changed.items():
                        self.output(f"[调试映射修正] {prod}: {old_cm}->{new_cm} | {old_nm}->{new_nm}")
                except Exception:
                    pass
        except Exception:
            pass

    def _get_effective_month_mapping(self) -> Dict[str, List[str]]:
        """返回标准化的品种->(指定月,指定下月)映射，优先 params.month_mapping，回退参数表调试映射。"""
        try:
            today = datetime.now().date()
            last_date = getattr(self, "_month_mapping_last_refresh_date", None)
            if last_date != today:
                self._param_override_cache = {}
                self._month_mapping_last_refresh_date = today
        except Exception:
            pass
        def normalize_pair(v: Any) -> Optional[List[str]]:
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return [str(v[0]).upper(), str(v[1]).upper()]
            if isinstance(v, str) and v.strip():
                parts = re.split(r"[\|,;\s]+", v.strip())
                if len(parts) >= 2:
                    return [parts[0].upper(), parts[1].upper()]
            if isinstance(v, dict):
                cm = v.get("current") or v.get("current_month") or v.get("cm")
                nm = v.get("next") or v.get("next_month") or v.get("nm")
                if cm and nm:
                    return [str(cm).upper(), str(nm).upper()]
            return None

        mapping: Dict[str, List[str]] = {}
        mm_param = getattr(self.params, "month_mapping", {}) or {}
        if isinstance(mm_param, dict) and mm_param:
            for k, v in mm_param.items():
                pair = normalize_pair(v)
                if pair and len(pair) >= 2:
                    mapping[str(k).upper()] = pair

        # 尝试使用缓存/参数表（包含对齐后的结果），对缺失品种做补全（不覆盖已有品种）
        overrides = getattr(self, "_param_override_cache", None) or self._load_param_table()

        def load_from_container(d: Any) -> Optional[Dict[str, List[str]]]:
            if not isinstance(d, dict):
                return None
            mm_raw = d.get("month_mapping")
            if isinstance(mm_raw, str):
                try:
                    mm_raw = json.loads(mm_raw)
                except Exception:
                    mm_raw = None
            if not isinstance(mm_raw, dict) or not mm_raw:
                return None
            tmp: Dict[str, List[str]] = {}
            for k, v in mm_raw.items():
                pair = normalize_pair(v)
                if pair and len(pair) >= 2:
                    tmp[str(k).upper()] = pair
            return tmp or None

        for cand in (
            overrides if isinstance(overrides, dict) else None,
            overrides.get("params") if isinstance(overrides, dict) else None,
            overrides.get("original_defaults") if isinstance(overrides, dict) else None,
        ):
            mm_tmp = load_from_container(cand)
            if mm_tmp:
                for prod, pair in mm_tmp.items():
                    if prod not in mapping:
                        mapping[prod] = pair

        if getattr(self.params, "debug_output", False) and mapping:
            sample = list(mapping.items())[:3]
            self._debug(f"[月映射] 合并后映射, 示例: {sample}, 总数: {len(mapping)}")

        # 回退调试映射（从参数表解析），仅在缺少 params.month_mapping 时使用
        if not mapping:
            dbg = self._get_debug_month_mapping()
            if dbg:
                for k, v in dbg.items():
                    mapping[str(k).upper()] = [str(v[0]).upper(), str(v[1]).upper()]
                if getattr(self.params, "debug_output", False):
                    sample = list(mapping.items())[:3]
                    self._debug(f"[月映射] 使用参数表映射, 示例: {sample}")
        return mapping

    def _get_next_month_id(self, future_id: str) -> Optional[str]:
        """只从参数表读取指定下月；无配置直接返回 None"""
        try:
            future_clean = self._normalize_future_id(future_id)
            product_code = self._extract_product_code(future_clean).upper()
            if not product_code:
                product_code = self._extract_product_code(future_id or "").upper()

            eff_map = self._get_effective_month_mapping()
            candidate: Optional[str] = None
            lst = eff_map.get(product_code)
            if not lst:
                try:
                    f2o = getattr(self, "future_to_option_map", {}) or {}
                    for fut_code, opt_code in f2o.items():
                        if str(opt_code).strip().upper() == product_code:
                            lst = eff_map.get(str(fut_code).strip().upper())
                            if lst:
                                break
                except Exception:
                    pass
            if isinstance(lst, list) and len(lst) >= 2:
                candidate = self._normalize_future_id((lst[1] or "").strip())
                if candidate:
                    if getattr(self.params, "debug_output", False):
                        self.output(f"计算 {future_id} 的指定下月合约.. 使用有效映射: {candidate}")
                    return candidate

            # 若映射存在但格式不一致，尝试转换为 CZCE 一位年或去零格式
            def to_czce_style(code: str) -> Optional[str]:
                try:
                    p = self._extract_product_code(code).upper()
                    y = self._extract_year(code)
                    mth = self._extract_month(code)
                    if p and y and mth is not None:
                        return f"{p}{str(y)[-1]}{mth:02d}".upper()
                    return None
                except Exception:
                    return None

            def to_single_month_style(code: str) -> Optional[str]:
                try:
                    p = self._extract_product_code(code).upper()
                    yy = re.search(r"[A-Z]+(\d{2})(\d{1,2})", code)
                    if yy:
                        y2 = yy.group(1)
                        mth = int(yy.group(2))
                        return f"{p}{y2}{mth:02d}".upper()
                    cz = re.search(r"[A-Z]+(\d)(\d{2})", code)
                    if cz:
                        y1 = cz.group(1)
                        mth = int(cz.group(2))
                        return f"{p}{y1}{mth:02d}".upper()
                    return None
                except Exception:
                    return None

            if isinstance(lst, list) and len(lst) >= 2:
                candidate = (lst[1] or "").strip().upper()
                for conv in (to_czce_style, to_single_month_style):
                    alt = conv(candidate)
                    if alt:
                        alt = self._normalize_future_id(alt)
                        if getattr(self.params, "debug_output", False):
                            self.output(f"计算 {future_id} 的指定下月合约.. 尝试对齐格式: {candidate}->{alt}")
                        return alt

            # 如果映射存在但长度不足2，且全局下月匹配当前品种，尝试兜底
            if (not candidate) and isinstance(lst, list) and len(lst) < 2:
                nm_spec = (getattr(self.params, "next_specified_month", "") or "").strip().upper()
                if nm_spec and self._extract_product_code(nm_spec).upper() == product_code:
                    return self._normalize_future_id(nm_spec)

            nm_spec = (getattr(self.params, "next_specified_month", "") or "").strip().upper()
            if nm_spec and nm_spec.startswith(product_code):
                if getattr(self.params, "debug_output", False):
                    self.output(f"计算 {future_id} 的指定下月合约.. 使用 next_specified_month: {nm_spec}")
                return nm_spec

            # 若映射缺失或无有效候选且全局下月与当前品种一致，再兜底
            if nm_spec and (not lst or not candidate):
                if self._extract_product_code(nm_spec).upper() == product_code:
                    return self._normalize_future_id(nm_spec)

            return None

        except Exception as e:
            self.output(f"  计算失败: {e}\n{traceback.format_exc()}")
            return None

    def _is_symbol_specified_or_next(self, symbol: str) -> bool:
        """严格判断合约是否为参数表配置的指定月或指定下月"""
        try:
            symbol_norm = self._normalize_future_id(symbol)
            eff_map = self._get_effective_month_mapping()

            if not eff_map:
                specified_month = (getattr(self.params, "specified_month", "") or "").strip().upper()
                next_specified_month = (getattr(self.params, "next_specified_month", "") or "").strip().upper()

                if not specified_month and not next_specified_month:
                    if getattr(self.params, "debug_output", False):
                        self._debug("[严格过滤] 未配置指定月/指定下月，跳过全部合约")
                    return False  # 未配置则全部跳过

                symbol_upper = symbol_norm.upper()
                return symbol_upper in (specified_month, next_specified_month)

            symbol_upper = symbol_norm.upper()
            if getattr(self.params, "debug_output", False):
                self._debug(f"[严格过滤] 检查合约: {symbol_upper}")

            symbol_prod = self._extract_product_code(symbol_upper)
            for product_code, month_list in eff_map.items():
                if not isinstance(month_list, list) or len(month_list) < 2:
                    continue

                product_upper = str(product_code).upper()
                if symbol_prod == product_upper:
                    specified = (month_list[0] or "").strip().upper()
                    next_specified = (month_list[1] or "").strip().upper()
                    if symbol_upper == specified or symbol_upper == next_specified:
                        if getattr(self.params, "debug_output", False):
                            self._debug(f"[严格过滤] 匹配成功: {symbol_upper} -> {product_upper}")
                        return True
                    if getattr(self.params, "debug_output", False):
                        self._debug(f"[严格过滤] 匹配失败: {symbol_upper} 不在 {specified}/{next_specified}")
                    return False  # 品种匹配但月份不符，直接拒绝

            # 有有效映射但未覆盖该品种，回退全局指定月/下月
            specified_month = (getattr(self.params, "specified_month", "") or "").strip().upper()
            next_specified_month = (getattr(self.params, "next_specified_month", "") or "").strip().upper()
            if specified_month or next_specified_month:
                return symbol_upper in (specified_month, next_specified_month)

            if getattr(self.params, "debug_output", False):
                self._debug(f"[严格过滤] 无匹配品种: {symbol_upper}")
            return False

        except Exception as e:
            self._debug(f"[严格过滤] 异常: {e}, symbol={symbol}")
            return False

    def _is_symbol_current_or_next(self, future_id: str) -> bool:
        """仅依据参数表/全局指定月判断是否为指定月或指定下月"""
        if not future_id:
            return False
        norm = self._normalize_future_id(future_id)
        # 1) 优先按参数/映射判断
        if self._is_symbol_specified_or_next(norm):
            return True
        return False

    def _is_symbol_current(self, future_id: str) -> bool:
        """仅依据参数表/全局指定月判断是否为指定月"""
        if not future_id:
            return False
        future_upper = self._normalize_future_id(future_id)
        if not future_upper:
            return False

        # 先看有效映射
        eff_map = self._get_effective_month_mapping()
        if eff_map:
            m = re.match(r'^([A-Z]+)', future_upper)
            if m:
                prod = m.group(1)
                lst = eff_map.get(prod)
                if isinstance(lst, list) and lst:
                    spec = (lst[0] or "").strip().upper()
                    if spec:
                        return future_upper == spec
            # 映射存在但缺少该品种时，严格拒绝，避免被全局参数误放行
            if getattr(self.params, "debug_output", False):
                self._debug(f"[严格过滤] 映射缺失该品种: {future_upper}")
            return False

        # 再看全局指定月（仅当映射不存在时）
        spec_global = (getattr(self.params, "specified_month", "") or "").strip().upper()
        if spec_global:
            return future_upper == spec_global

        return False

        # 未配置则视为不匹配
        if getattr(self.params, "debug_output", False):
            self._debug("[严格过滤] 未配置指定月，跳过当前合约判定")
        return False

    def subscribe_market_data(self, subscribe_options: bool = True) -> None:
        """订阅期货行情，并可选地订阅对应的期权行情

        参数:
            subscribe_options: 是否同时为已知期权订阅行情并创建 K 线生成器（默认True）
        """
        future_subscribed = 0
        option_subscribed = 0
        option_skipped = 0

        futures_skipped = 0

        # 先订阅期货：仅订阅指定月和指定下月的期货
        for future in self.future_instruments:
            exchange = future.get("ExchangeID", "")      
            instrument_id = future.get("InstrumentID", "")

            if not (exchange and instrument_id):
                futures_skipped += 1
                self._debug("跳过无效期货合约")
                continue
            instrument_norm = self._normalize_future_id(instrument_id)
            
            # 仅订阅指定月/指定下月期货
            if not self._is_symbol_current_or_next(instrument_norm.upper()):
                self._debug(f"跳过非指定月/指定下月期货: {exchange}.{instrument_id}")
                futures_skipped += 1
                continue

            try:
                self.sub_market_data(exchange=exchange, instrument_id=instrument_id)
                future_subscribed += 1
            except Exception as e:
                self.output(f"订阅期货失败 {exchange}.{instrument_id}: {e}")

        # 可选：订阅期权行情（可配置是否仅指定月/指定下月）
        if subscribe_options:
            seen_opt_keys = set()
            filter_opts = self._resolve_subscribe_flag(
                "subscribe_only_specified_month_options",
                "subscribe_only_current_next_options",
                False
            )
            allowed_future_symbols: Set[str] = set()
            if filter_opts:
                for fid in list(self.option_instruments.keys()):
                    fid_norm = self._normalize_future_id(fid)
                    if self._is_symbol_specified_or_next(fid_norm.upper()):
                        allowed_future_symbols.add(fid_norm.upper())
            for future_symbol, options in self.option_instruments.items():
                future_symbol_norm = self._normalize_future_id(future_symbol)
                if filter_opts and future_symbol_norm.upper() not in allowed_future_symbols:
                    option_skipped += len(options)
                    continue
                for option in options:
                    opt_exchange = option.get('ExchangeID', '')
                    opt_instrument = option.get('InstrumentID', '')
                    if not opt_exchange or not opt_instrument:
                        option_skipped += 1
                        continue

                    opt_norm = self._normalize_future_id(str(opt_instrument))
                    if filter_opts and (not self._is_symbol_specified_or_next(opt_norm.upper())):
                        option_skipped += 1
                        continue

                    opt_key = f"{opt_exchange}_{opt_instrument}"
                    if opt_key in seen_opt_keys:
                        continue
                    seen_opt_keys.add(opt_key)

                    try:
                        self.sub_market_data(exchange=opt_exchange, instrument_id=opt_instrument)
                        option_subscribed += 1
                    except Exception as e:
                        self.output(f"订阅期权失败 {opt_exchange}.{opt_instrument}: {e}")
                        continue

        self._debug(
            f"订阅完成: 期货 {future_subscribed} 个，期货跳过 {futures_skipped} 个，期权 {option_subscribed} 个，期权跳过 {option_skipped} 个"
        )
        # 定时任务统一在start() 中通过 _safe_add_interval_job 添加，避免重复与残留
        self._debug("订阅完成：定时计算任务将在start() 阶段统一启动")

    def unsubscribe_all(self) -> None:
        """取消订阅所有合约"""
        for future in self.future_instruments:
            exchange = future.get("ExchangeID", "")
            instrument_id = future.get("InstrumentID", "")
            
            if exchange and instrument_id:
                try:
                    self.unsub_market_data(exchange=exchange, instrument_id=instrument_id)
                except Exception as e:
                    self.output(f"取消订阅期货失败 {exchange}.{instrument_id}: {e}")
        
        # 同步取消已订阅的期权，避免残留订阅
        for _, options in self.option_instruments.items():
            for option in options:
                opt_exchange = option.get("ExchangeID", "")
                opt_instrument = option.get("InstrumentID", "")
                if opt_exchange and opt_instrument:
                    try:
                        self.unsub_market_data(exchange=opt_exchange, instrument_id=opt_instrument)
                    except Exception as e:
                        self.output(f"取消订阅期权失败 {opt_exchange}.{opt_instrument}: {e}")

    def on_kline(self, kline: KLineData) -> None:
        """K线数据回调"""
        try:
            # 暂停/未运行/销毁或交易关闭时忽略回调
            paused = getattr(self, "is_paused", False)
            if (not getattr(self, "is_running", False)) or paused or (getattr(self, "trading", True) is False) or getattr(self, "destroyed", False):
                if paused:
                    try:
                        self.paused_drop_counts["kline"] = self.paused_drop_counts.get("kline", 0) + 1
                        c = self.paused_drop_counts["kline"]
                        if c in (1, 10) or c % 100 == 0:
                            exch = getattr(kline, "exchange", "")
                            inst = getattr(kline, "instrument_id", "")
                            self._debug(f"暂停中丢弃kline 回调 {c} 次示例{exch}.{inst}")
                    except Exception:
                        pass
                return
            if not getattr(self, "_instruments_ready", False):
                return
            frequency = getattr(kline, "style", getattr(self.params, "kline_style", "M1"))
            self._process_kline_data(kline.exchange, kline.instrument_id, frequency, kline)
            # 到达任意相关合约的K线后尝试触发宽度计算
            self._trigger_width_calc_for_kline(kline)
        except Exception as e:
            self.output(f"处理K线数据失败 {e}\n{traceback.format_exc()}")

    def _trigger_width_calc_for_kline(self, kline: KLineData) -> None:
        """在收到K线后触发对应期货的宽度计算（统一使用优化版）"""
        try:
            # CORE_LOCK_START_KLINE_TRIGGER
            # 暂停/未运行/销毁或交易关闭时不触发计算
            if (not getattr(self, "is_running", False)) or getattr(self, "is_paused", False) or (getattr(self, "trading", True) is False) or getattr(self, "destroyed", False):
                return
            inst_id = getattr(kline, "instrument_id", "")
            exch = getattr(kline, "exchange", "")
            if not inst_id:
                return

            base_id = self._normalize_future_id(inst_id)
            if not self._is_instrument_allowed(base_id, exch):
                return
            # 若自身就是期货，直接计算（所有品种仅指定月触发宽度计算；指定下月仅作为指定月组件）
            if base_id in self.future_symbol_to_exchange:
                fut_exch = self.future_symbol_to_exchange.get(base_id, exch)
                prod = self._extract_product_code(base_id)
                if prod and not self._has_option_for_product(prod):
                    self._debug(f"[调试] K线触发跳过无期权品种: {fut_exch}.{base_id}")
                    self._cleanup_kline_cache_for_symbol(base_id)
                    return
                if not self._is_symbol_current(base_id):
                    self._debug(f"[调试] K线触发跳过非指定月/指定下月商品合约: {fut_exch}.{base_id}")
                    try:
                        if base_id in self.option_width_results:
                            del self.option_width_results[base_id]
                    except Exception:
                        pass
                    return
                self.calculate_option_width_optimized(fut_exch, base_id)
                return

            # 若是期权，通过期权代码映射回期货
            fut_symbol = self._extract_future_symbol(base_id)
            if fut_symbol:
                fut_exch = self.future_symbol_to_exchange.get(fut_symbol, exch)
                prod = self._extract_product_code(fut_symbol)
                if prod and not self._has_option_for_product(prod):
                    self._debug(f"[调试] K线触发跳过无期权品种: {fut_exch}.{fut_symbol}")
                    self._cleanup_kline_cache_for_symbol(fut_symbol)
                    return
                if not self._is_symbol_current(fut_symbol):
                    self._debug(f"[调试] K线触发跳过非指定月/指定下月商品合约: {fut_exch}.{fut_symbol}")
                    try:
                        if fut_symbol in self.option_width_results:
                            del self.option_width_results[fut_symbol]
                    except Exception:
                        pass
                    return
                self.calculate_option_width_optimized(fut_exch, fut_symbol)
            # CORE_LOCK_END_KLINE_TRIGGER
        except Exception as e:
            self._debug(f"触发宽度计算失败 {exch}.{inst_id}: {e}")

    # 订阅/退订安全封装：暂停/销毁时阻断订阅，防止暂停后仍产生日志
    def sub_market_data(self, exchange: Optional[str] = None, instrument_id: Optional[str] = None, *args, **kwargs) -> None:
        try:
            if (not getattr(self, "is_running", False)) or getattr(self, "is_paused", False) or (getattr(self, "trading", True) is False) or getattr(self, "destroyed", False):
                self._debug(f"已暂停/未运行/销毁，忽略订阅 {exchange}.{instrument_id}")
                return
            try:
                # 兼容父类可能的无参订阅调用
                if exchange is None or instrument_id is None:
                    super().sub_market_data(*args, **kwargs)
                else:
                    super().sub_market_data(exchange=exchange, instrument_id=instrument_id)
            except Exception:
                # 某些平台使用不同签名
                if exchange is None or instrument_id is None:
                    super().sub_market_data(*args, **kwargs)
                else:
                    super().sub_market_data(exchange, instrument_id)
        except Exception as e:
            self.output(f"订阅失败 {exchange}.{instrument_id}: {e}")

    def unsub_market_data(self, exchange: Optional[str] = None, instrument_id: Optional[str] = None, *args, **kwargs) -> None:
        try:
            # 父类可能不存在unsub_market_data；在本地测试环境下进行守卫
            target = getattr(super(), "unsub_market_data", None)
            if not callable(target):
                if getattr(self.params, "debug_output", False):
                    self._debug("父类缺少 unsub_market_data，跳过退订")
                return
            # 退订在暂停/销毁时仍允许执行；兼容父类无参调用
            try:
                if exchange is None or instrument_id is None:
                    target(*args, **kwargs)
                else:
                    target(exchange=exchange, instrument_id=instrument_id)
            except Exception:
                if exchange is None or instrument_id is None:
                    target(*args, **kwargs)
                else:
                    target(exchange, instrument_id)
        except Exception as e:
            self.output(f"退订失败{exchange}.{instrument_id}: {e}")
    
    def calculate_all_option_widths(self) -> None:
        """计算所有主流商品期货的期权宽度（含调试输出）"""
        # 统一状态门控：任何暂停/停止/销毁态均跳过（并避免输出日志）
        if self._is_paused_or_stopped():
            self.output(f"[调试] calculate_all_option_widths 被跳过，因为处于暂停/停止态", force=True)
            return
        self.output(
            f"[调试] 进入 calculate_all_option_widths，my_is_running={self.my_is_running} "
            f"my_is_paused={self.my_is_paused} my_destroyed={self.my_destroyed}",
            force=True,
        )
        start_time = time.time()
        self.calculation_stats["total_calculations"] += 1
        try:
            # 汇总容器：计算覆盖度与缺失对照
            baseline_products_cfg = str(getattr(self.params, "future_products", "") or "")
            if baseline_products_cfg:
                baseline_products = set([p.strip().upper() for p in baseline_products_cfg.split(',') if p.strip()])
            else:
                baseline_products = set(["CU","RB","AL","ZN","AU","AG","M","Y","A","J","JM","I","CF","SR","MA","TA"])  # 商品基线
            computed_products: Set[str] = set()
            computed_items: List[str] = []
            skipped_items: List[str] = []
            skip_reasons: Dict[str, int] = {}
            with self.data_lock:
                # 打印当前 option_instruments 的分组键，确认商品/股指期权是否都已进入计算集合
                try:
                    group_keys = sorted(self.option_instruments.keys())
                    counts = {k: len(self.option_instruments.get(k, [])) for k in group_keys}
                    self.output(f"[调试] option_instruments 分组键 ({len(group_keys)}): {group_keys}", force=True)
                    self.output(f"[调试] option_instruments 分组数量: {counts}", force=True)
                except Exception:
                    pass

                # 调试：打印已加载的品种映射数量，确认指定月/指定下月映射已生效
                try:
                    mm = self._get_debug_month_mapping()
                    if mm:
                        sample_items = list(mm.items())[:5]
                        self.output(f"[调试] month_mapping 加载 {len(mm)} 条，示例: {sample_items}", force=True)
                except Exception:
                    pass

                # 覆盖所有已加载的品种（股指 + 商品），统一排序与信号生成
                # CORE_LOCK_START_BATCH_FILTER
                for future in self.future_instruments:
                    # 若暂停/停止/销毁，则提前中断循环，确保暂停生效
                    if self._is_paused_or_stopped():
                        self._debug("[调试] 计算中途检测到暂停/停止，立即中断循环")
                        return
                    exchange = future.get("ExchangeID", self.params.exchange)
                    future_id = future.get("InstrumentID", "")
                    if not exchange or not future_id:
                        skip_reasons["无效合约"] = skip_reasons.get("无效合约", 0) + 1
                        continue
                    prod = self._extract_product_code(future_id)
                    # 跳过没有对应期权分组的品种（如无商品期权的品种）
                    if prod and not self._has_option_for_product(prod):
                        self._debug(f"[调试] 跳过无期权分组品种 {exchange}.{future_id}")
                        skipped_items.append(f"{exchange}.{future_id}")
                        skip_reasons["无期权分组"] = skip_reasons.get("无期权分组", 0) + 1
                        try:
                            if future_id in self.option_width_results:
                                del self.option_width_results[future_id]
                        except Exception:
                            pass
                        self._cleanup_kline_cache_for_symbol(future_id)
                        continue
                    # 跳过非真实月份（如Main/Weighted/带下划线等综合合约）
                    if not self._is_real_month_contract(future_id):
                        self._debug(f"[调试] 跳过非真实月份合约 {exchange}.{future_id}")
                        skipped_items.append(f"{exchange}.{future_id}")
                        skip_reasons["非真实月份"] = skip_reasons.get("非真实月份", 0) + 1
                        continue
                    # 所有品种仅计算指定月合约的宽度；指定下月仅作为指定月的宽度组件，不单独计算
                    if not self._is_symbol_current(str(future_id).upper()):
                        self._debug(f"[调试] 跳过非指定月/指定下月商品合约: {exchange}.{future_id}")
                        # 清理可能的历史结果，避免显示已被规则排除的非指定月宽度
                        try:
                            if future_id in self.option_width_results:
                                del self.option_width_results[future_id]
                        except Exception:
                            pass
                        skipped_items.append(f"{exchange}.{future_id}")
                        skip_reasons["非指定月"] = skip_reasons.get("非指定月", 0) + 1
                        continue
                    self.output(f"[调试] 宽度计算循环: {exchange}.{future_id}")
                    self._debug(f"[宽度计算] 参与计算: {exchange}.{future_id}")
                    try:
                        self.calculate_option_width_optimized(exchange, future_id)
                        if prod:
                            computed_products.add(prod)
                        computed_items.append(f"{exchange}.{future_id}")
                    except Exception as e:
                        self._debug(f"计算期权宽度失败 {future_id}: {e}")
                # CORE_LOCK_END_BATCH_FILTER

                # 额外：让期权品种（IO/HO/MO/EO）自然参与排序
                option_groups = self._build_option_groups_by_option_prefix()
                inv_prefix_map = {"IF": "IO", "IH": "HO", "IC": "MO", "IM": "EO"}
                # CORE_LOCK_START_BATCH_GROUPS
                for group_id, opts in option_groups.items():
                    # 支持所有期权品种的分组（含商品期权），格式前缀+YY+M/M:
                    # 如 IO2601、M2505、SR509 等，统一自然参与排序
                    # 映射到对应期货ID用于价格与月份判断
                    underlying = self._extract_future_symbol(group_id)
                    if not underlying:
                        continue
                    # 指定月判断使用期货ID，确保与统一规则一致
                    if not self._is_symbol_current(underlying.upper()):
                        skipped_items.append(f"CFFEX.{group_id}")
                        skip_reasons["非指定月"] = skip_reasons.get("非指定月", 0) + 1
                        # 清理旧结果避免误显示
                        try:
                            if group_id in self.option_width_results:
                                del self.option_width_results[group_id]
                        except Exception:
                            pass
                        continue
                    # 选择一个期权的交易所作为分组的Exchange（优先期权自身交易所）
                    exch_guess = None
                    try:
                        if opts:
                            exch_guess = opts[0].get('ExchangeID', None)
                    except Exception:
                        pass
                    if not exch_guess:
                        exch_guess = self.future_symbol_to_exchange.get(underlying.upper(), 'CFFEX')
                    # 计算并写入以期权前缀为键的宽度结果
                    try:
                        self._calculate_option_width_for_option_group(exch_guess, group_id, underlying.upper(), opts, inv_prefix_map)
                        computed_items.append(f"{exch_guess}.{group_id}")
                    except Exception as e:
                        self._debug(f"计算期权分组宽度失败 {group_id}: {e}")
                # CORE_LOCK_END_BATCH_GROUPS

                # SHFE期权分组（商品期权，两位年格式）
                self._debug(f"[调试] 开始SHFE期权分组")
                shfe_option_groups = self._build_shfe_option_groups()
                self._debug(f"[调试] SHFE期权分组完成，共{len(shfe_option_groups)}个分组")
                inv_prefix_map_shfe = {}
                for group_id, opts in shfe_option_groups.items():
                    self._debug(f"[调试] 处理SHFE期权分组: {group_id}")
                    underlying = group_id
                    if not underlying:
                        continue
                    if not self._is_symbol_current(underlying.upper()):
                        skipped_items.append(f"SHFE.{group_id}")
                        skip_reasons["非指定月"] = skip_reasons.get("非指定月", 0) + 1
                        try:
                            if group_id in self.option_width_results:
                                del self.option_width_results[group_id]
                        except Exception:
                            pass
                        continue
                    try:
                        self._calculate_option_width_for_option_group("SHFE", group_id, underlying.upper(), opts, inv_prefix_map_shfe)
                        computed_items.append(f"SHFE.{group_id}")
                    except Exception as e:
                        self._debug(f"计算SHFE期权分组宽度失败 {group_id}: {e}")

                # DCE期权分组（商品期权，两位年格式）
                self._debug(f"[调试] 开始DCE期权分组")
                dce_option_groups = self._build_dce_option_groups()
                self._debug(f"[调试] DCE期权分组完成，共{len(dce_option_groups)}个分组")
                inv_prefix_map_dce = {}
                for group_id, opts in dce_option_groups.items():
                    self._debug(f"[调试] 处理DCE期权分组: {group_id}")
                    underlying = group_id
                    if not underlying:
                        continue
                    if not self._is_symbol_current(underlying.upper()):
                        skipped_items.append(f"DCE.{group_id}")
                        skip_reasons["非指定月"] = skip_reasons.get("非指定月", 0) + 1
                        try:
                            if group_id in self.option_width_results:
                                del self.option_width_results[group_id]
                        except Exception:
                            pass
                        continue
                    try:
                        self._calculate_option_width_for_option_group("DCE", group_id, underlying.upper(), opts, inv_prefix_map_dce)
                        computed_items.append(f"DCE.{group_id}")
                    except Exception as e:
                        self._debug(f"计算DCE期权分组宽度失败 {group_id}: {e}")

                # CZCE期权分组（商品期权，一位年格式）
                self._debug(f"[调试] 开始CZCE期权分组")
                czce_option_groups = self._build_czce_option_groups()
                self._debug(f"[调试] CZCE期权分组完成，共{len(czce_option_groups)}个分组")
                inv_prefix_map_czce = {}
                for group_id, opts in czce_option_groups.items():
                    self._debug(f"[调试] 处理CZCE期权分组: {group_id}")
                    underlying = group_id
                    if not underlying:
                        continue
                    if not self._is_symbol_current(underlying.upper()):
                        skipped_items.append(f"CZCE.{group_id}")
                        skip_reasons["非指定月"] = skip_reasons.get("非指定月", 0) + 1
                        try:
                            if group_id in self.option_width_results:
                                del self.option_width_results[group_id]
                        except Exception:
                            pass
                        continue
                    try:
                        self._calculate_option_width_for_option_group("CZCE", group_id, underlying.upper(), opts, inv_prefix_map_czce)
                        computed_items.append(f"CZCE.{group_id}")
                    except Exception as e:
                        self._debug(f"计算CZCE期权分组宽度失败 {group_id}: {e}")

            self.calculation_stats["successful_calculations"] += 1
        except Exception as e:
            self.calculation_stats["failed_calculations"] += 1
            self._debug(f"批量计算期权宽度失败: {e}")
        end_time = time.time()
        calculation_time = end_time - start_time
        self.calculation_stats["last_calculation_time"] = calculation_time
        old_avg = self.calculation_stats["average_calculation_time"]
        self.calculation_stats["average_calculation_time"] = (
            old_avg * 0.9 + calculation_time * 0.1
        )
        with self.data_lock:
            self._cleanup_caches()

        # 在本次批量计算结束后统一生成并输出信号，避免每个品种重复输出TOP3
        try:
            signals = self.generate_trading_signals_optimized()
            self.output_trading_signals_optimized(signals)
        except Exception as e:
            self._debug(f"统一输出交易信号失败: {e}")

        # 一次性汇总输出：覆盖度与缺失对照（仅基于商品基线集合）
        try:
            commodity_baseline = set(["CU","RB","AL","ZN","AU","AG","M","Y","A","J","JM","I","CF","SR","MA","TA"]) if not baseline_products_cfg else set([p for p in baseline_products if p in {"CU","RB","AL","ZN","AU","AG","M","Y","A","J","JM","I","CF","SR","MA","TA"}])
            covered = sorted(list(commodity_baseline.intersection(computed_products)))
            missing = sorted(list(commodity_baseline.difference(computed_products)))
            self.output("================ 覆盖度一次性汇总 ================", force=True)
            self.output(f"商品基线品种数: {len(commodity_baseline)} | 覆盖(参与指定月宽度计算)数: {len(covered)} | 缺失数: {len(missing)}", force=True)
            if covered:
                self.output("已覆盖品种: " + ",".join(covered), force=True)
            if missing:
                self.output("缺失品种: " + ",".join(missing), force=True)
            self.output(f"指定月参与计算合约共: {len(computed_items)}", force=True)
            if getattr(self.params, "debug_output", False):
                self._debug(f"参与计算合约样例: {computed_items[:8]}")
                self._debug(f"跳过合约计数: {sum(skip_reasons.values())} | 明细: {skip_reasons}")
            self.output("===============================================")
        except Exception as e:
            self._debug(f"覆盖度汇总输出失败: {e}")

    def calculate_option_width_optimized(self, exchange: str, future_id: str) -> None:
        """优化版期权宽度计算（BestVersion迁移）"""
        try:
            # 暂停/未运行/销毁或交易关闭时跳过计算
            if (not getattr(self, "is_running", False)) or getattr(self, "is_paused", False) or (getattr(self, "trading", True) is False) or getattr(self, "destroyed", False):
                return
            key = f"{exchange}_{future_id}"
            klines = self._get_kline_series(exchange, future_id)
            if len(klines) < 2:
                if key not in self.kline_insufficient_logged:
                    self._debug(f"K线不足 {future_id}, 当前{len(klines)}根")
                    self.kline_insufficient_logged.add(key)
                if future_id in self.option_width_results:
                    del self.option_width_results[future_id]
                return
            self.kline_insufficient_logged.discard(key)

            # K线新鲜度校验：超过阈值则视为过期，不参与计算
            try:
                max_age = int(getattr(self.params, "kline_max_age_sec", 0) or 0)
            except Exception:
                max_age = 0

            if max_age > 0:
                def _get_ts(bar: Any) -> Optional[datetime]:
                    candidates = ["datetime", "DateTime", "timestamp", "Timestamp", "time", "Time"]
                    for name in candidates:
                        try:
                            val = getattr(bar, name, None)
                        except Exception:
                            val = None
                        if val is None:
                            continue
                        if isinstance(val, datetime):
                            return val
                        if isinstance(val, (int, float)):
                            try:
                                return datetime.fromtimestamp(val)
                            except Exception:
                                continue
                        if isinstance(val, str):
                            for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y%m%d %H:%M:%S"):
                                try:
                                    return datetime.fromisoformat(val) if fmt is None else datetime.strptime(val, fmt)
                                except Exception:
                                    continue
                    return None

                last_ts = _get_ts(klines[-1])
                if last_ts:
                    age = (datetime.now() - last_ts).total_seconds()
                    if age > max_age:
                        try:
                            self.output(
                                f"K线过旧({age:.1f}s>{max_age}s)，跳过计算 {exchange}.{future_id}"
                            )
                        except Exception:
                            pass
                        if future_id in self.option_width_results:
                            del self.option_width_results[future_id]
                        return
            # 当前价取最新一根的 close
            current_price = getattr(klines[-1], 'close', 0)
            # 统一通过 helper 获取上一根的 close（符合开盘/非开盘规则）
            previous_price = self._previous_price_from_klines(klines)
            if current_price <= 0 or previous_price <= 0:
                if key not in self.zero_price_logged:
                    self._debug(f"价格无效: {future_id}, 当前价{current_price}, 前价:{previous_price}")
                    self.zero_price_logged.add(key)
                if future_id in self.option_width_results:
                    del self.option_width_results[future_id]
                return
            future_rising = current_price > previous_price
            future_id_upper = self._normalize_future_id(future_id)
            next_month_id = self._get_next_month_id(future_id)
            
            # 关键修复：确保期权键已归一化，防止因动态加载或键不一致导致匹配失败
            self._normalize_option_group_keys()
            
            if getattr(self.params, "debug_output", False):
                try:
                    self._debug(f"[调试] {exchange}.{future_id_upper} 采用相邻指定下月: {str(next_month_id).upper() if next_month_id else '无'}（严格不跨指定下月之后的月份）")
                except Exception:
                    pass
            current_options = self.option_instruments.get(future_id_upper, [])
            next_options = self.option_instruments.get(
                next_month_id.upper(), []
            ) if next_month_id else []

            # 调试：商品品种宽度输入快照
            if getattr(self.params, "debug_output", False):
                try:
                    if future_id_upper.startswith(("CU", "RB", "AL", "ZN", "AU", "AG")):
                        self._debug(
                            f"[调试] 宽度输入快照 {exchange}.{future_id_upper} K线={len(klines)} 指定月期权={len(current_options)} 指定下月期权={len(next_options)}"
                        )
                except Exception:
                    pass
            if not current_options and not next_options:
                self._debug(
                    f"期权数据为空: {future_id} (指定月:0, 指定下月:{len(next_options) if next_month_id else 0})"
                )
                if future_id in self.option_width_results:
                    del self.option_width_results[future_id]
                self._cleanup_kline_cache_for_symbol(future_id)
                return
            # CORE_LOCK_START_WIDTH_FUTURES
            target_om_current = []
            target_om_next = []
            missing_kline_current: Set[str] = set()
            missing_kline_next: Set[str] = set()
            for option in current_options:
                opt_exch = option.get("ExchangeID", exchange)
                opt_id = option["InstrumentID"]
                klines_opt = self._get_kline_series(opt_exch, opt_id)
                kl_len = len(klines_opt)
                if kl_len < 2:
                    # 缺少K线的行权价记为0贡献，记录诊断但不终止本轮行权价遍历
                    if getattr(self.params, "debug_output", False):
                        key = f"{opt_exch}.{opt_id}"
                        if key not in self._insufficient_option_kline_logged:
                            self._insufficient_option_kline_logged.add(key)
                            self._debug(f"[诊断] 期权K线不足: {key} 当前K线={kl_len}，需要>=2")
                    missing_kline_current.add(opt_id)
                if not self._is_out_of_money_optimized(
                    opt_id,
                    current_price,
                    option
                ):
                    continue
                option_type = self._get_option_type(
                    opt_id,
                    option,
                    opt_exch
                )
                # 期货上涨时选择看涨期权（C），期货下跌时选择看跌期权（P）
                if future_rising and option_type == "C":
                    target_om_current.append(option)
                elif not future_rising and option_type == "P":
                    target_om_current.append(option)
            for option in next_options:
                opt_exch = option.get("ExchangeID", exchange)
                opt_id = option["InstrumentID"]
                klines_opt = self._get_kline_series(opt_exch, opt_id)
                kl_len = len(klines_opt)
                if kl_len < 2:
                    if getattr(self.params, "debug_output", False):
                        key = f"{opt_exch}.{opt_id}"
                        if key not in self._insufficient_option_kline_logged:
                            self._insufficient_option_kline_logged.add(key)
                            self._debug(f"[诊断] 期权K线不足: {key} 当前K线={kl_len}，需要>=2")
                    missing_kline_next.add(opt_id)
                if not self._is_out_of_money_optimized(
                    opt_id,
                    current_price,
                    option
                ):
                    continue
                option_type = self._get_option_type(
                    opt_id,
                    option,
                    opt_exch
                )
                # 期货上涨时选择看涨期权（C），期货下跌时选择看跌期权（P）
                if future_rising and option_type == "C":
                    target_om_next.append(option)
                elif not future_rising and option_type == "P":
                    target_om_next.append(option)
            total_target_current = len(target_om_current)
            total_target_next = len(target_om_next)
            has_direction_options = total_target_current > 0 and total_target_next > 0
            # 若任一月份没有方向虚值，则跳过（核心策略要求指定月与指定下月同时有期权宽度）
            if total_target_current == 0 or total_target_next == 0:
                if self.params.debug_output:
                    sample = (current_options + next_options)[:10]
                    self.output(
                        f"无方向虚值期权 {future_id}, 方向={'上涨' if future_rising else '下跌'}, 期权总数={len(current_options)+len(next_options)}, "
                        f"指定月虚值目标{total_target_current}, 指定下月虚值目标{total_target_next}, 期货价{current_price}"
                    )
                    for idx, opt in enumerate(sample, start=1):
                        oid = opt.get('InstrumentID', '')
                        exch_opt = opt.get('ExchangeID', exchange)
                        otype = self._get_option_type(oid, opt, exch_opt)
                        strike = float(opt.get('StrikePrice', 0) or 0)
                        key = f"{exch_opt}_{oid}"
                        series = self.kline_data.get(key, {}).get('data', [])
                        if len(series) < 2:
                            # 即时兜底拉取最近M1，尽量避免误判缺数据
                            try:
                                fetched = self.get_recent_m1_kline(exch_opt, oid, count=self.params.max_kline)
                                if fetched:
                                    series = self.kline_data.get(key, {}).get('data', [])
                            except Exception:
                                pass
                        if len(series) < 2:
                            self.output(f"  [{idx}] {oid} 缺少K线")
                            continue
                        o_prev = getattr(series[-2], 'close', 0)
                        o_cur = getattr(series[-1], 'close', 0)
                        is_call = (str(otype).upper() == 'C')
                        is_otm = (current_price < strike) if is_call else (current_price > strike)
                        # 修改 dir_match 计算逻辑：当期货下跌时，如果选择了看涨期权（C），则 dir_match = True
                        dir_match = (is_call and future_rising) or ((not is_call) and (not future_rising))
                        # 如果期货下跌时选择了看涨期权（C），则 dir_match = True
                        # 重新计算 dir_match，确保逻辑正确
                        if not future_rising and is_call:
                            dir_match = True
                        up_move = o_cur > o_prev
                        self.output(
                            f"  [{idx}] {oid} 类型={otype} 行权={strike} prev={o_prev} cur={o_cur} "
                            f"otm={is_otm} dir_match={dir_match} up_move={up_move}"
                        )
                else:
                    self._debug(f"无方向虚值期权 {future_id}")
                if future_id in self.option_width_results:
                    del self.option_width_results[future_id]
                return
            current_count = 0
            for option in target_om_current:
                opt_exch = option.get("ExchangeID", exchange)
                if option.get("InstrumentID") in missing_kline_current:
                    continue
                if self._is_option_sync_rising_optimized(
                    option,
                    opt_exch,
                    future_rising,
                    current_price
                ):
                    current_count += 1
            next_count = 0
            for option in target_om_next:
                opt_exch = option.get("ExchangeID", exchange)
                if option.get("InstrumentID") in missing_kline_next:
                    continue
                if self._is_option_sync_rising_optimized(
                    option,
                    opt_exch,
                    future_rising,
                    current_price
                ):
                    next_count += 1
            # 宽度=指定月方向虚值同步 + 指定下月方向虚值同步（核心原始逻辑）
            option_width = current_count + next_count
            all_current_sync = (current_count == total_target_current) if total_target_current > 0 else False
            all_next_sync = (next_count == total_target_next) if total_target_next > 0 else False
            all_sync = all_current_sync and all_next_sync
            total_om_current = 0
            total_om_next = 0
            for option in current_options:
                opt_exch = option.get("ExchangeID", exchange)
                opt_id = option["InstrumentID"]
                if self._is_out_of_money_optimized(
                    opt_id,
                    current_price,
                    option
                ):
                    total_om_current += 1
            for option in next_options:
                opt_exch = option.get("ExchangeID", exchange)
                opt_id = option["InstrumentID"]
                if self._is_out_of_money_optimized(
                    opt_id,
                    current_price,
                    option
                ):
                    total_om_next += 1
            self.option_width_results[future_id] = {
                "exchange": exchange,
                "future_id": future_id,
                "option_width": option_width,
                "all_sync": all_sync,
                "current_month_count": current_count,
                "total_current_target": total_target_current,
                "next_month_count": next_count,
                "total_next_target": total_target_next,
                "total_all_om_current": total_om_current,
                "total_all_om_next": total_om_next,
                "current_price": current_price,
                "previous_price": previous_price,
                "future_rising": future_rising,
                "timestamp": datetime.now(),
                "has_both_months": bool(current_options) and bool(next_options),
                "has_direction_options": has_direction_options
            }
            self._debug(
                f"计算完成: {future_id}, 指定月期权宽度={option_width} (指定月{current_count}+指定下月{next_count}), "
                f"指定月同步={current_count}/{total_target_current}, 指定下月同步={next_count}/{total_target_next}, "
                f"指定月虚值={total_om_current}, 指定下月虚值={total_om_next}"
            )
            self.output(
                f"计算完成: {future_id} 指定月期权宽度={option_width} 指定月{current_count}/{total_target_current} 指定下月{next_count}/{total_target_next}"
            )
            # CORE_LOCK_END_WIDTH_FUTURES
            # 生成/输出信号改为批量计算结束后统一进行，避免重复输出
        except Exception as e:
            self._debug(f"计算期权宽度失败 {future_id}: {e}")

    def _calculate_option_width_for_option_group(self, exchange: str, group_id: str, underlying_future_id: str,
                                                 options: List[Dict[str, Any]], inv_prefix_map: Dict[str, str]) -> None:
        """基于期权分组（如 IO2601）计算宽度，使用对应期货ID的价格与月份。
        仅用于让期权品种自然参与排序，不改变生产逻辑。
        """
        # 暂停/未运行/销毁或交易关闭时跳过
        if (not getattr(self, "is_running", False)) or getattr(self, "is_paused", False) or (getattr(self, "trading", True) is False) or getattr(self, "destroyed", False):
            return
        key = f"{exchange}_{group_id}"
        # 获取期货K线
        klines = self._get_kline_series(exchange, underlying_future_id)
        if len(klines) < 2:
            if key not in self.kline_insufficient_logged:
                self._debug(f"K线不足(期货) {underlying_future_id}, 当前{len(klines)}根")
                self.kline_insufficient_logged.add(key)
            if group_id in self.option_width_results:
                del self.option_width_results[group_id]
            return
        self.kline_insufficient_logged.discard(key)

        # 期货价格与方向
        current_price = klines[-1].close
        previous_price = self._previous_price_from_klines(klines)
        if current_price <= 0 or previous_price <= 0:
            if key not in self.zero_price_logged:
                self._debug(f"价格无效(期货) {underlying_future_id}, 当前价{current_price}, 前价:{previous_price}")
                self.zero_price_logged.add(key)
            if group_id in self.option_width_results:
                del self.option_width_results[group_id]
            return
        future_rising = current_price > previous_price

        # 指定下月分组ID：由期货指定下月ID映射到期权前缀
        next_future_id = self._get_next_month_id(underlying_future_id)
        next_group_id = None
        if next_future_id:
            try:
                m = re.match(r"^([A-Z]+)(\d{2})(\d{1,2})$", next_future_id.upper())
                if m:
                    fut_prefix = m.group(1)
                    yy = m.group(2)
                    mm = m.group(3)
                    # 次月分组选项前缀：优先映射（股指），否则与期货前缀一致（商品期权）
                    opt_prefix = inv_prefix_map.get(fut_prefix, fut_prefix)
                    if opt_prefix:
                        next_group_id = f"{opt_prefix}{yy}{mm}"
            except Exception:
                pass

        # 取指定月/指定下月期权列表
        current_options = list(options or [])
        next_options = []
        if next_group_id:
            # 从所有期权再构建一次分组以获取指定下月同类
            try:
                groups_all = self._build_option_groups_by_option_prefix()
                next_options = list(groups_all.get(next_group_id.upper(), []))
            except Exception:
                next_options = []

        if not current_options and not next_options:
            if group_id not in self._no_option_group_logged:
                self._no_option_group_logged.add(group_id)
                self._debug(
                    f"期权数据为空(分组): {group_id} | 基础期货={underlying_future_id} | 交易所={exchange}"
                )
            if group_id in self.option_width_results:
                del self.option_width_results[group_id]
            return
        else:
            self._no_option_group_logged.discard(group_id)

        # CORE_LOCK_START_WIDTH_GROUP
        # 方向虚值筛选与同步判断
        target_om_current = []
        target_om_next = []
        missing_kline_current: Set[str] = set()
        missing_kline_next: Set[str] = set()
        for option in current_options:
            opt_exch = option.get("ExchangeID", exchange)
            opt_id = option.get("InstrumentID", "")
            klines_opt = self._get_kline_series(opt_exch, opt_id)
            kl_len = len(klines_opt)
            if kl_len < 2:
                if getattr(self.params, "debug_output", False):
                    key = f"{opt_exch}.{opt_id}"
                    if key not in self._insufficient_option_kline_logged:
                        self._insufficient_option_kline_logged.add(key)
                        self._debug(f"[诊断] 期权K线不足: {key} 当前K线={kl_len}，需要>=2")
                missing_kline_current.add(opt_id)
            if not self._is_out_of_money_optimized(opt_id, current_price, option):
                continue
            option_type = self._get_option_type(opt_id, option, opt_exch)
            if future_rising and option_type == "C":
                target_om_current.append(option)
            elif (not future_rising) and option_type == "P":
                target_om_current.append(option)
        for option in next_options:
            opt_exch = option.get("ExchangeID", exchange)
            opt_id = option.get("InstrumentID", "")
            klines_opt = self._get_kline_series(opt_exch, opt_id)
            kl_len = len(klines_opt)
            if kl_len < 2:
                if getattr(self.params, "debug_output", False):
                    key = f"{opt_exch}.{opt_id}"
                    if key not in self._insufficient_option_kline_logged:
                        self._insufficient_option_kline_logged.add(key)
                        self._debug(f"[诊断] 期权K线不足: {key} 当前K线={kl_len}，需要>=2")
                missing_kline_next.add(opt_id)
            if not self._is_out_of_money_optimized(opt_id, current_price, option):
                continue
            option_type = self._get_option_type(opt_id, option, opt_exch)
            if future_rising and option_type == "C":
                target_om_next.append(option)
            elif (not future_rising) and option_type == "P":
                target_om_next.append(option)

        total_target_current = len(target_om_current)
        total_target_next = len(target_om_next)
        has_direction_options = total_target_current > 0 and total_target_next > 0
        # 若任一月份没有方向虚值，则跳过（核心策略要求两个月同时有方向虚值期权）
        if total_target_current == 0 or total_target_next == 0:
            self._debug(f"无方向虚值期权(分组) {group_id}")
            if group_id in self.option_width_results:
                del self.option_width_results[group_id]
            return

        current_count = 0
        for option in target_om_current:
            opt_exch = option.get("ExchangeID", exchange)
            if option.get("InstrumentID") in missing_kline_current:
                continue
            if self._is_option_sync_rising_optimized(option, opt_exch, future_rising, current_price):
                current_count += 1
        next_count = 0
        for option in target_om_next:
            opt_exch = option.get("ExchangeID", exchange)
            if option.get("InstrumentID") in missing_kline_next:
                continue
            if self._is_option_sync_rising_optimized(option, opt_exch, future_rising, current_price):
                next_count += 1

        # 宽度=指定月方向虚值同步 + 指定下月方向虚值同步（核心逻辑）
        option_width = current_count + next_count
        all_current_sync = (current_count == total_target_current) if total_target_current > 0 else False
        all_next_sync = (next_count == total_target_next) if total_target_next > 0 else False
        all_sync = all_current_sync and all_next_sync

        # 统计全部虚值数量
        total_om_current = 0
        total_om_next = 0
        for option in current_options:
            opt_exch = option.get("ExchangeID", exchange)
            opt_id = option.get("InstrumentID", "")
            if self._is_out_of_money_optimized(opt_id, current_price, option):
                total_om_current += 1
        for option in next_options:
            opt_exch = option.get("ExchangeID", exchange)
            opt_id = option.get("InstrumentID", "")
            if self._is_out_of_money_optimized(opt_id, current_price, option):
                total_om_next += 1

        self.option_width_results[group_id] = {
            "exchange": exchange,
            "future_id": group_id,
            "option_width": option_width,
            "all_sync": all_sync,
            "current_month_count": current_count,
            "total_current_target": total_target_current,
            "next_month_count": next_count,
            "total_next_target": total_target_next,
            "total_all_om_current": total_om_current,
            "total_all_om_next": total_om_next,
            "current_price": current_price,
            "previous_price": previous_price,
            "future_rising": future_rising,
            "timestamp": datetime.now(),
            "has_both_months": bool(current_options) and bool(next_options),
            "has_direction_options": has_direction_options
        }
        # CORE_LOCK_END_WIDTH_GROUP
        self._debug(
            f"[分组] 计算完成: {group_id}, 宽度={option_width} 指定月{current_count}/{total_target_current} 指定下月{next_count}/{total_target_next}"
        )

    def _get_option_type(self, option_symbol: str, option_dict: Optional[Dict[str, Any]] = None, exchange: str = "") -> Optional[str]:
        """获取期权类型（C=看涨，P=看跌）带缓存，支持多种格式"""
        
        # 调试输出
        debug_output = getattr(self.params, "debug_output", False)
        if debug_output:
            self._debug(f"[调试] _get_option_type 开始: option_symbol={option_symbol}, exchange={exchange}")
        
        # 检查缓存
        if option_symbol in self.option_type_cache:
            result = self.option_type_cache[option_symbol]
            if debug_output:
                self._debug(f"[调试] _get_option_type 缓存命中: option_symbol={option_symbol}, type={result}")
            return result
        
        # 优先使用合约字典中的字段
        if option_dict and option_dict.get("OptionType"):
            option_type = str(option_dict["OptionType"]).upper()
            if debug_output:
                self._debug(f"[调试] _get_option_type 从字典获取: option_symbol={option_symbol}, OptionType={option_type}")
            if option_type in ("C", "P"):
                self.option_type_cache[option_symbol] = option_type
                if debug_output:
                    self._debug(f"[调试] _get_option_type 字典返回: option_symbol={option_symbol}, type={option_type}")
                return option_type
        
        option_symbol_upper = option_symbol.upper()
        result = None
        
        # 根据交易所选择合适的正则表达式
        if exchange == "CFFEX":
            # 中金所格式: IO2601-C-4350 或 IO2601C4350
            match = re.search(r"[A-Z]{2}\d{4}-([CP])-\d+", option_symbol_upper)
            if not match:
                match = re.search(r"[A-Z]{2}\d{4}([CP])\d+", option_symbol_upper)
            if match:
                result = match.group(1)
                if debug_output:
                    self._debug(f"[调试] _get_option_type 中金所匹配: option_symbol={option_symbol}, type={result}")
        elif exchange == "CZCE":
            # 郑商所格式: SR509C5000
            match = re.search(r"[A-Z]{2}\d{3}([CP])\d+", option_symbol_upper)
            if match:
                result = match.group(1)
                if debug_output:
                    self._debug(f"[调试] _get_option_type 郑商所匹配: option_symbol={option_symbol}, type={result}")
        elif exchange in ("SHFE", "DCE"):
            # 上期所/大商所格式: cu2601c7300
            match = re.search(r"[A-Z]{1}\d{4}([CP])\d+", option_symbol_upper)
            if match:
                result = match.group(1)
                if debug_output:
                    self._debug(f"[调试] _get_option_type {exchange}匹配: option_symbol={option_symbol}, type={result}")
        else:
            # 未知交易所，尝试所有格式
            # 格式1: IO2601-C-4350 (中金所标准格式)
            match = re.search(r"[A-Z]{2}\d{4}-([CP])-\d+", option_symbol_upper)
            if match:
                result = match.group(1)
                if debug_output:
                    self._debug(f"[调试] _get_option_type 未知交易所格式1匹配: option_symbol={option_symbol}, type={result}")
            else:
                # 格式2: IO2601C4350 (股指期权无分隔符格式)
                match = re.search(r"[A-Z]{2}\d{4}([CP])\d+", option_symbol_upper)
                if match:
                    result = match.group(1)
                    if debug_output:
                        self._debug(f"[调试] _get_option_type 未知交易所格式2匹配: option_symbol={option_symbol}, type={result}")
                else:
                    # 格式3: SR509C (郑商所格式)
                    match = re.search(r"[A-Z]{2}\d{3}([CP])\d+", option_symbol_upper)
                    if match:
                        result = match.group(1)
                        if debug_output:
                            self._debug(f"[调试] _get_option_type 未知交易所格式3匹配: option_symbol={option_symbol}, type={result}")
                    else:
                        # 格式0: cu2601c7300 (商品期权格式)
                        match = re.search(r"[A-Z]{1}\d{4}([CP])\d+", option_symbol_upper)
                        if match:
                            result = match.group(1)
                            if debug_output:
                                self._debug(f"[调试] _get_option_type 未知交易所格式0匹配: option_symbol={option_symbol}, type={result}")
                        else:
                            # 格式4: 字母+数字+C/P 格式 (通用格式)
                            match = re.search(r"([CP])\d+", option_symbol_upper)
                            if match:
                                result = match.group(1)
                                if debug_output:
                                    self._debug(f"[调试] _get_option_type 未知交易所格式4匹配: option_symbol={option_symbol}, type={result}")
                            else:
                                # 格式5: 数字+C/P 格式
                                match = re.search(r"\d+([CP])", option_symbol_upper)
                                if match:
                                    result = match.group(1)
                                    if debug_output:
                                        self._debug(f"[调试] _get_option_type 未知交易所格式5匹配: option_symbol={option_symbol}, type={result}")
                                else:
                                    # 格式6: 包含 C- 或 P- 的格式
                                    if "-C-" in option_symbol_upper or "_C_" in option_symbol_upper:
                                        result = "C"
                                        if debug_output:
                                            self._debug(f"[调试] _get_option_type 未知交易所格式6匹配: option_symbol={option_symbol}, type=C")
                                    elif "-P-" in option_symbol_upper or "_P_" in option_symbol_upper:
                                        result = "P"
                                        if debug_output:
                                            self._debug(f"[调试] _get_option_type 未知交易所格式6匹配: option_symbol={option_symbol}, type=P")
                                    else:
                                        # 格式7: 最后一位字母是C或P
                                        if option_symbol_upper.endswith("C"):
                                            result = "C"
                                            if debug_output:
                                                self._debug(f"[调试] _get_option_type 未知交易所格式7匹配: option_symbol={option_symbol}, type=C")
                                        elif option_symbol_upper.endswith("P"):
                                            result = "P"
                                            if debug_output:
                                                self._debug(f"[调试] _get_option_type 未知交易所格式7匹配: option_symbol={option_symbol}, type=P")
        
        # 缓存结果
        if result is not None:
            self.option_type_cache[option_symbol] = result
            if debug_output:
                self._debug(f"[调试] _get_option_type 返回: option_symbol={option_symbol}, type={result}")
        else:
            if debug_output:
                self._debug(f"[调试] _get_option_type 无匹配: option_symbol={option_symbol}, result=None")
            key = f"{exchange}.{option_symbol_upper}"
            if key not in self._option_type_failed_logged:
                self._option_type_failed_logged.add(key)
                self._debug(f"[诊断] 期权类型识别失败: {key}，请检查合约格式")
            self.option_type_cache[option_symbol] = result
        
        return result

    def _is_option_sync_rising_optimized(self, option: Dict[str, Any], exchange: str,
                                        future_rising: bool, current_price: float) -> bool:
        """判断期权是否同步移动（只采用连续2根K线收盘价判断）"""
        try:
            option_id = option.get("InstrumentID", "")
            klines = self._get_kline_series(exchange, option_id)
            if len(klines) < 2:
                if getattr(self.params, "debug_output", False):
                    key = f"{exchange}.{option_id}"
                    if key not in self._insufficient_option_kline_logged:
                        self._insufficient_option_kline_logged.add(key)
                        self._debug(f"[诊断] 期权K线不足: {key} 当前K线={len(klines)}，需要>=2")
                return False
            
            # 只采用连续2根K线收盘价判断（使用统一 helper 获取上一根）
            previous_option_price = self._previous_price_from_klines(klines)
            option_rising = klines[-1].close > previous_option_price
            
            # 获取期权类型
            option_type = self._get_option_type(option_id, option, option.get("ExchangeID", exchange))
            
            # 检查期货和期权是否同步移动
            # 看涨期权上涨和期货上涨；看跌期权上涨和期货下跌
            if option_type == "C":
                # 看涨期权：期权上涨 且 期货上涨
                return option_rising and future_rising
            elif option_type == "P":
                # 看跌期权：期权上涨 且 期货下跌
                return option_rising and not future_rising
            else:
                return False
        except Exception:
            return False

    def _is_out_of_money_optimized(self, option_symbol: str, future_price: float, 
                                  option_dict: Optional[Dict[str, Any]] = None) -> bool:
        """判断虚值期权（带缓存，BestVersion迁移）"""
        
        # 调试输出
        debug_output = getattr(self.params, "debug_output", False)
        if debug_output:
            self._debug(f"[调试] _is_out_of_money_optimized 开始: option_symbol={option_symbol}, future_price={future_price}")
        
        cache_key = f"{option_symbol}_{future_price:.2f}"
        if cache_key in self.out_of_money_cache:
            if debug_output:
                self._debug(f"[调试] _is_out_of_money_optimized 缓存命中: option_symbol={option_symbol}, future_price={future_price}")
            return self.out_of_money_cache[cache_key]
        try:
            # 优先使用合约字典中的信息
            if option_dict:
                strike_price = option_dict.get("StrikePrice")
                option_type = option_dict.get("OptionType")
                if debug_output:
                    self._debug(f"[调试] _is_out_of_money_optimized 从字典获取: option_symbol={option_symbol}, StrikePrice={strike_price}, OptionType={option_type}")
                if strike_price is not None and option_type:
                    option_type_upper = str(option_type).upper()
                    # 关键修复：正确判断虚值期权
                    if option_type_upper == "C":
                        # 看涨期权：行权价 > 期货价格 = 虚值
                        result = strike_price > future_price
                        if debug_output:
                            self._debug(f"[调试] _is_out_of_money_optimized 看涨期权: strike={strike_price}, future={future_price}, otm={result}")
                    elif option_type_upper == "P":
                        # 看跌期权：行权价 < 期货价格 = 虚值
                        result = strike_price < future_price
                        if debug_output:
                            self._debug(f"[调试] _is_out_of_money_optimized 看跌期权: strike={strike_price}, future={future_price}, otm={result}")
                    else:
                        result = False
                    self.out_of_money_cache[cache_key] = result
                    if debug_output:
                        self._debug(f"[调试] _is_out_of_money_optimized 字典返回: option_symbol={option_symbol}, result={result}")
                    return result
            # 如果合约字典中没有，尝试从期权代码中解析
            option_symbol_upper = option_symbol.upper()
            
            # 尝试解析行权价和期权类型
            strike_price = None
            option_type = None
            
            # 格式1: IO2601-C-4350 或 IO2601-P-4350 (中金所标准格式) - 优先匹配
            match1 = re.search(r"[A-Z]{2}\d{4}-([CP])-(\d+(?:\.\d+)?)", option_symbol_upper)
            if match1:
                option_type = match1.group(1)
                strike_price = match1.group(2)
            
            # 格式2: IO2601C4350 (股指期权无分隔符格式)
            if not match1:
                match2 = re.search(r"[A-Z]{2}\d{4}([CP])(\d+(?:\.\d+)?)", option_symbol_upper)
                if match2:
                    option_type = match2.group(1)
                    strike_price = match2.group(2)
            
            # 格式3: C1200 或 P1200
            if not match1 and not match2:
                match3 = re.search(r"([CP])(\d+(?:\.\d+)?)", option_symbol_upper)
                if match3:
                    option_type = match3.group(1)
                    strike_price = match3.group(2)
            
            # 格式0: cu2601c7300 (商品期权格式：品种+年月+类型+行权价) - 最后匹配，避免误匹配股指期权
            if not match1 and not match2 and not match3:
                match0 = re.search(r"[A-Z]{1}\d{4}([CP])(\d+(?:\.\d+)?)", option_symbol_upper)
                if match0:
                    option_type = match0.group(1)
                    strike_price = match0.group(2)
            
            # 格式4: 1200C 或 1200P
            if not match1 and not match2 and not match3 and not match0:
                match4 = re.search(r"(\d+(?:\.\d+)?)([CP])", option_symbol_upper)
                if match4:
                    strike_price = match4.group(1)
                    option_type = match4.group(2)
            
            if strike_price and option_type:
                try:
                    strike = float(strike_price)
                    
                    if option_type == "C":
                        result = strike > future_price
                    elif option_type == "P":
                        result = strike < future_price
                    else:
                        result = False
                    
                    self.out_of_money_cache[cache_key] = result
                    return result
                except Exception:
                    pass
            
            # 如果以上方法都无法解析，返回False
            result = False
            self.out_of_money_cache[cache_key] = result
            return result
        except Exception:
            result = False
            self.out_of_money_cache[cache_key] = result
            return result

    def _cleanup_caches(self) -> None:
        """清理缓存"""
        if len(self.out_of_money_cache) > self.cache_max_size:
            keys = list(self.out_of_money_cache.keys())
            keys_to_remove = keys[:len(keys)//2]
            for k in keys_to_remove:
                del self.out_of_money_cache[k]
        if len(self.option_type_cache) > self.cache_max_size:
            keys = list(self.option_type_cache.keys())
            keys_to_remove = keys[:len(keys)//2]
            for k in keys_to_remove:
                del self.option_type_cache[k]

    def generate_trading_signals_optimized(self) -> List[Dict[str, Any]]:
        """优化版信号生成（排序三原则）"""
        if not self.option_width_results:
            return []
        valid_results = []
        for future_id, result in self.option_width_results.items():
            if not self._is_real_month_contract(str(future_id)):
                continue
            if not self._is_symbol_current(str(future_id)):
                continue
            prod = self._extract_product_code(str(future_id))
            if prod and not self._has_option_for_product(prod):
                continue
            if (result.get("has_direction_options", False) and
                result["option_width"] > 0):
                valid_results.append((future_id, result))
        if not valid_results:
            return []
        
        # 划分全部同步和部分同步结果
        all_sync_results = []
        partial_sync_results = []
        for future_id, result in valid_results:
            if result["all_sync"]:
                all_sync_results.append((future_id, result))
            else:
                partial_sync_results.append((future_id, result))
        
        # 按期权宽度降序排序
        def sort_key(item: tuple) -> int:
            return item[1]["option_width"]
        all_sync_results.sort(key=sort_key, reverse=True)
        partial_sync_results.sort(key=sort_key, reverse=True)
        
        signals = []
        
        # 原则1：全部同步结果中取期权宽度最大者为最优信号
        if all_sync_results:
            max_all_sync_width = all_sync_results[0][1]["option_width"]
            for future_id, result in all_sync_results:
                if result["option_width"] == max_all_sync_width:
                    signals.append(
                        self._create_signal_dict(future_id, result, "最优信号")
                    )
        
        # 原则2：全部同步结果中较大宽度优于部分同步结果较小宽度
        # 遍历全部同步结果（除了已添加的最优信号）
        for future_id, result in all_sync_results[1:]:
            # 检查是否优于所有部分同步结果
            if partial_sync_results:
                max_partial_width = partial_sync_results[0][1]["option_width"]
                if result["option_width"] > max_partial_width:
                    signals.append(
                        self._create_signal_dict(future_id, result, "次优信号")
                    )
        
        # 原则3：如果没有全部同步结果，或者全部同步结果已经处理完毕
        # 从部分同步结果中取最大宽度者
        if not all_sync_results or (len(all_sync_results) == 1 and partial_sync_results):
            if partial_sync_results:
                max_partial_width = partial_sync_results[0][1]["option_width"]
                for future_id, result in partial_sync_results:
                    if result["option_width"] == max_partial_width:
                        signals.append(
                            self._create_signal_dict(future_id, result, "次优信号")
                        )
        
        # 如果没有全部同步结果，只有部分同步结果
        if not all_sync_results and partial_sync_results:
            max_partial_width = partial_sync_results[0][1]["option_width"]
            signals = []  # 重置信号列表
            for future_id, result in partial_sync_results:
                if result["option_width"] == max_partial_width:
                    signals.append(
                        self._create_signal_dict(future_id, result, "部分同步信号")
                    )

        # 全品种统一排序：优先级（最优>次优>部分同步），同优先级按宽度降序
        priority = {"最优信号": 0, "次优信号": 1, "部分同步信号": 2}
        signals.sort(
            key=lambda s: (
                priority.get(s.get("signal_type"), 3) if isinstance(s, dict) else 3,
                -(s.get("option_width", 0) if isinstance(s, dict) else 0),
                s.get("future_id", "") if isinstance(s, dict) else ""
            )
        )

        # 调试输出全品种排序结果，便于验证三原则排序
        if getattr(self.params, "debug_output", False):
            try:
                ranked = [
                    f"{sig.get('signal_type')} {sig.get('exchange')}.{sig.get('future_id')} 宽度={sig.get('option_width')}"
                    for sig in signals
                ]
                self._debug(f"信号排序结果: {ranked}")
            except Exception:
                pass

        return signals

    def _create_signal_dict(self, future_id: str, result: Dict[str, Any], signal_type: str) -> Dict[str, Any]:
        """创建信号字典（BestVersion迁移）"""
        if result["future_rising"]:
            action = "买入"
        else:
            action = "卖出"
        price_change_percent = 0
        if result["previous_price"] > 0:
            price_change_percent = (
                (result["current_price"] - result["previous_price"]) / 
                result["previous_price"] * 100
            )
        return {
            "future_id": future_id,
            "exchange": result["exchange"],
            "signal_type": signal_type,
            "option_width": result["option_width"],
            "all_sync": result["all_sync"],
            "action": action,
            "price": result["current_price"],
            "previous_price": result["previous_price"],
            "price_change_percent": price_change_percent,
            "timestamp": result["timestamp"],
            "current_month_count": result["current_month_count"],
            "next_month_count": result["next_month_count"],
            "total_current_target": result["total_current_target"],
            "total_next_target": result["total_next_target"],
            "total_all_om_current": result["total_all_om_current"],
            "total_all_om_next": result["total_all_om_next"]
        }

    def output_trading_signals_optimized(self, signals: List[Dict[str, Any]]) -> None:
        """优化版信号输出（BestVersion迁移）"""
        now = datetime.now()
        # 暂停/停止/销毁/关闭交易时直接跳过输出，确保暂停生效
        if self._is_paused_or_stopped():
            return
        mode = str(getattr(self.params, "output_mode", "debug")).lower()
        # 普通输出：记录当前输出模式，便于判断是否进入交易模式
        try:
            self.output(f"当前输出模式: {mode}")
        except Exception:
            pass
        if not signals:
            # 交易模式下不更新UI，保持上次有信号的状态；调试模式输出提示
            if mode == "trade":
                # 不输出任何内容、不更新冷却键，直接返回
                return
            else:
                self._debug("当前无有效交易信号")
                return
        # 调试模式下也执行与交易模式一致的下单流程（开盘门控 + 时效过滤 + TOP3签名去重，只对最高优先级尝试下单）
        if mode != "trade":
            # 开盘时间门控：调试模式也遵循开盘时间
            if not self.is_market_open():
                try:
                    self.output("当前非开盘时间，跳过自动下单（调试模式）")
                except Exception:
                    pass
            else:
                # 信号时间戳新鲜度检查，防止用到过期K线生成的信号
                try:
                    max_age = int(getattr(self.params, "signal_max_age_sec", 180) or 0)
                except Exception:
                    max_age = 0
                if max_age > 0:
                    fresh_signals: List[Dict[str, Any]] = []
                    for sig in signals:
                        ts = sig.get("timestamp")
                        if not isinstance(ts, datetime):
                            continue
                        age = (now - ts).total_seconds()
                        if age > max_age:
                            continue
                        fresh_signals.append(sig)
                    signals = fresh_signals

                if signals:
                    try:
                        top_n = int(getattr(self.params, "top3_rows", 3) or 3)
                    except Exception:
                        top_n = 3
                    top = signals[:top_n]
                    if top:
                        sig_items = [
                            f"{sig.get('exchange')}.{sig.get('future_id')}|{sig.get('signal_type')}|{sig.get('option_width')}"
                            for sig in top
                        ]
                        signature = "||".join(sig_items)
                    else:
                        signature = "EMPTY"
                    last_sig = getattr(self, "top3_last_signature", None)
                    # 调试输出签名变化，便于确认是否被去重拦截
                    try:
                        self.output(f"TOP3签名(调试模式下单): 当前={signature} 上次={last_sig}")
                    except Exception:
                        pass
                    if last_sig != signature:
                        # 更新去重状态并尝试下单最高优先级信号
                        self.top3_last_signature = signature
                        self.top3_last_emit_time = now
                        try:
                            try:
                                self.output(
                                    f"准备自动下单(调试): {top[0].get('exchange')}.{top[0].get('future_id')} "
                                    f"信号={str(top[0].get('signal_type') or '')} 宽度={top[0].get('option_width')}"
                                )
                            except Exception:
                                pass
                            self._try_execute_signal_order(top[0])
                        except Exception as e:
                            self.output(f"执行信号下单失败(调试): {e}")
        if mode == "trade":
            # 开盘时间门控：测试模式可绕过
            if not self.is_market_open():
                try:
                    self.output("当前非开盘时间，跳过交易模式信号输出与自动下单")
                except Exception:
                    pass
                return

            # 信号时间戳新鲜度检查，防止用到过期K线生成的信号
            try:
                max_age = int(getattr(self.params, "signal_max_age_sec", 180) or 0)
            except Exception:
                max_age = 0
            if max_age > 0:
                fresh_signals: List[Dict[str, Any]] = []
                for sig in signals:
                    ts = sig.get("timestamp")
                    if not isinstance(ts, datetime):
                        try:
                            self.output(
                                f"信号缺少时间戳，跳过 {sig.get('exchange')}.{sig.get('future_id')}"
                            )
                        except Exception:
                            pass
                        continue
                    age = (now - ts).total_seconds()
                    if age > max_age:
                        try:
                            self.output(
                                f"信号过旧({age:.1f}s>{max_age}s)，跳过 {sig.get('exchange')}.{sig.get('future_id')}"
                            )
                        except Exception:
                            pass
                        continue
                    fresh_signals.append(sig)
                signals = fresh_signals

            if not signals:
                try:
                    self.output("交易模式无满足开盘/时效条件的信号，跳过输出与自动下单")
                except Exception:
                    pass
                return

            # 交易模式：仅输出排序前N名（默认3）的品种与信号性质，且进行去重/节流（按优先级→宽度降序）
            try:
                top_n = int(getattr(self.params, "top3_rows", 3) or 3)
            except Exception:
                top_n = 3
            # 普通输出：记录本轮信号数量与TOP行数
            try:
                self.output(f"交易模式: 本轮信号数={len(signals)} TOP行数={top_n}")
            except Exception:
                pass
            top = signals[:top_n]
            # 构建签名用于去重
            if top:
                sig_items = [
                    f"{sig.get('exchange')}.{sig.get('future_id')}|{sig.get('signal_type')}|{sig.get('option_width')}"
                    for sig in top
                ]
                signature = "||".join(sig_items)
            else:
                signature = "EMPTY"
            # 仅当签名发生变化时才刷新输出；相同签名直接跳过（不受冷却时间影响）
            last_sig = getattr(self, "top3_last_signature", None)
            # 普通输出：记录签名对比，便于确认是否被去重拦截
            try:
                self.output(f"TOP3签名: 当前={signature} 上次={last_sig}")
            except Exception:
                pass
            if last_sig == signature:
                try:
                    self.output("TOP3签名未变化，跳过刷新与自动下单")
                except Exception:
                    pass
                return
            # 更新去重状态
            self.top3_last_signature = signature
            self.top3_last_emit_time = now

            # 新增：在交易模式下对最高优先级信号立即尝试下单（遵循签名去重，避免重复下单）
            if top:
                try:
                    try:
                        self.output(
                            f"准备自动下单: {top[0].get('exchange')}.{top[0].get('future_id')} "
                            f"信号={str(top[0].get('signal_type') or '')} 宽度={top[0].get('option_width')}"
                        )
                    except Exception:
                        pass
                    self._try_execute_signal_order(top[0])
                except Exception as e:
                    # 提升为普通输出，便于日志可见
                    self.output(f"执行信号下单失败: {e}")

            # 实际输出（表格形式），按优先级→宽度降序，表格完整封闭（顶线、表头、分隔、行、底线）；不足N个时补0行
            headers = ["#", "交易所", "品种", "优先级", "信号", "宽度", "时间"]
            rows: List[List[str]] = []
            display_list = []
            # 固定N行：已有信号按序，其余补零
            for idx in range(top_n):
                if idx < len(top):
                    display_list.append(top[idx])
                else:
                    display_list.append({
                        "exchange": "",
                        "future_id": "",
                        "signal_type": "",
                        "option_width": 0,
                        "timestamp": None,
                    })
            for idx, sig in enumerate(display_list, start=1):
                ts = sig.get('timestamp')
                try:
                    tstr = ts.strftime('%H:%M:%S') if isinstance(ts, datetime) else str(ts or '')
                except Exception:
                    tstr = str(ts or '')
                stype = str(sig.get('signal_type') or "")
                if "最优" in stype:
                    pri = "最优"
                elif "次优" in stype:
                    pri = "次优"
                elif "部分同步" in stype:
                    pri = "部分同步"
                else:
                    pri = stype
                rows.append([
                    str(idx),
                    str(sig.get('exchange') or ""),
                    str(sig.get('future_id') or ""),
                    pri,
                    stype,
                    str(sig.get('option_width') or 0),
                    tstr
                ])
            col_widths = []
            for i in range(len(headers)):
                max_cell = max([len(r[i]) for r in rows]) if rows else 0
                col_widths.append(max(len(headers[i]), max_cell))
            sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
            header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
            table_lines: List[str] = [
                "交易模式：TOP3 信号（优先级→宽度降序）",
                sep,
                header_line,
                sep,
            ]
            for r in rows:
                row_line = "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
                table_lines.append(row_line)
            table_lines.append(sep)
            for line in table_lines:
                self.output(line, trade=True)
            # 记录单信号冷却避免刷屏但不逐条详细输出
            for sig in top:
                key = f"{sig.get('exchange')}|{sig.get('future_id')}|{sig.get('signal_type')}"
                self.signal_last_emit[key] = now
            return

        # 调试模式：详细输出
        signal_groups = {}
        for signal in signals:
            signal_type = signal['signal_type']
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal)
        for signal_type, signal_list in signal_groups.items():
            for signal in signal_list:
                signal_key = f"{signal['exchange']}|{signal['future_id']}|{signal_type}"
                last_emit = self.signal_last_emit.get(signal_key)
                if last_emit and self.signal_cooldown_sec > 0:
                    time_diff = (now - last_emit).total_seconds()
                    if time_diff < self.signal_cooldown_sec:
                        continue
                self.signal_last_emit[signal_key] = now
                sync_status = "全部同步" if signal['all_sync'] else "部分同步"
                action_desc = f"{signal['action']}开仓"
                price_change = signal.get('price_change_percent', 0)
                direction = "看涨" if signal['action'] == "买入" else "看跌"
                total_om = (
                    signal.get('total_all_om_current', 0) + 
                    signal.get('total_all_om_next', 0)
                )
                total_target = (
                    signal.get('total_current_target', 0) + 
                    signal.get('total_next_target', 0)
                )
                total_sync = (
                    signal.get('current_month_count', 0) + 
                    signal.get('next_month_count', 0)
                )
                self.output(
                    f"交易信号 [{signal_type}]: "
                    f"{signal['exchange']}.{signal['future_id']} "
                    f"方向: {direction} "
                    f"期权宽度: {signal['option_width']} "
                    f"同步状态 {sync_status} "
                    f"价格: {signal['price']:.2f} "
                    f"涨跌: {price_change:+.2f}% "
                    f"操作: {action_desc} "
                    f"(所有虚值期权{total_om}个，"
                    f"方向虚值{total_target}个，"
                    f"同步:{total_sync}个)"
                )

    def _try_execute_signal_order(self, signal: Dict[str, Any]) -> None:
        """将最高优先级信号转换为下单请求（调试/交易模式均可触发）。"""
        # 仅在交易开启且未暂停/销毁时下单
        if self._is_paused_or_stopped() or (not getattr(self, "trading", True)):
            try:
                self.output("自动下单被暂停/停止或 trading=False，跳过")
            except Exception:
                pass
            return

        exchange = signal.get("exchange") or getattr(self.params, "exchange", "")
        instrument_id = signal.get("future_id")
        if not exchange or not instrument_id:
            return

        # 进入钩子时记录信号摘要
        try:
            self.output(
                f"进入自动下单: {exchange}.{instrument_id} "
                f"action={str(signal.get('action') or '')} 宽度={signal.get('option_width')} "
                f"同步={signal.get('all_sync')}"
            )
        except Exception:
            pass

        # 方向映射：买入→"0"，卖出→"1"
        action = str(signal.get("action", "")).strip()
        direction = "0" if action == "买入" else "1"
        offset_flag = "0"  # 信号默认开仓

        # 手数：使用配置下限，防止越界
        volume = 1
        try:
            volume = int(getattr(self.params, "lots_min", 1) or 1)
        except Exception:
            volume = 1
        if volume <= 0:
            volume = 1

        # 价格：优先用盘口价，其次最新价，最后信号里的价格
        price = None
        price_source = None
        # 支持不含/含交易所前缀的两种键名
        tick = self.latest_ticks.get(instrument_id) or self.latest_ticks.get(f"{exchange}.{instrument_id}")
        if tick:
            if direction == "0":
                price = getattr(tick, "ask", None) or getattr(tick, "AskPrice1", None)
                price_source = "ask/AskPrice1" if price is not None else None
                if price is None:
                    price = getattr(tick, "last", None) or getattr(tick, "last_price", None)
                    price_source = "last/last_price" if price is not None else price_source
            else:
                price = getattr(tick, "bid", None) or getattr(tick, "BidPrice1", None)
                price_source = "bid/BidPrice1" if price is not None else None
                if price is None:
                    price = getattr(tick, "last", None) or getattr(tick, "last_price", None)
                    price_source = "last/last_price" if price is not None else price_source
        else:
            try:
                self.output(f"缺少最新tick: {exchange}.{instrument_id}，将尝试使用信号价")
            except Exception:
                pass
        if price is None:
            try:
                price = float(signal.get("price", 0) or 0)
                price_source = "signal.price"
            except Exception:
                price = 0
        if price is None or price <= 0:
            self.output(f"信号下单缺少有效价格，跳过 {exchange}.{instrument_id}")
            return
        # 调试：记录最终价格来源
        try:
            self.output(f"自动下单价格来源: {price_source} 价={price}")
        except Exception:
            pass

        # 发单
        order_id = self.place_order(
            exchange=exchange,
            instrument_id=instrument_id,
            direction=direction,
            offset_flag=offset_flag,
            price=float(price),
            volume=volume,
            order_price_type="2",
        )

        if order_id:
            self.output(f"信号下单成功 {exchange}.{instrument_id} action={action} 价={price} 手={volume} 订单ID={order_id}")
        else:
            self.output(f"信号下单失败 {exchange}.{instrument_id} action={action} 价={price} 手={volume}")

    def set_option_buy_limit(
        self,
        account_id: str,
        limit_amount: float,
        valid_hours: int = 24,
        force_set: bool = False,
    ) -> Tuple[bool, str]:
        """设置期权买方开仓限额（委托前调用）。"""
        return self.option_buy_executor.set_position_limit(
            account_id=account_id,
            limit_amount=limit_amount,
            valid_hours=valid_hours,
            force_set=force_set,
        )

    def execute_option_buy_open(
        self,
        account_id: str,
        option_data: Dict[str, Any],
        lots: int = 1,
    ) -> Tuple[bool, str, Optional[Dict]]:
        """执行期权买入开仓，内部包含限额校验与委托构造。"""
        return self.option_buy_executor.execute_option_buy_open(
            account_id=account_id,
            option_data=option_data,
            lots=lots,
        )
    
    def load_historical_klines(self) -> None:
        """主动获取历史K线数据（解决模拟环境缺少实时K线的问题）"""
        try:
            self._debug("=== 开始获取历史K线数据 ===")
            
            # 先解析可用的接口，避免后面引用未定义
            mc_get_bars = getattr(self.market_center, "get_bars", None)
            infini_get_bars = getattr(infini, "get_bars", None)
            mc_get_kline = getattr(self.market_center, "get_kline_data", None)

            get_bars_fn = mc_get_bars if callable(mc_get_bars) else None
            get_bars_source = "MarketCenter"
            if get_bars_fn is None and callable(infini_get_bars):
                get_bars_fn = infini_get_bars
                get_bars_source = "infini"

            # 如果没有 get_bars，则尝试使用 MarketCenter.get_kline_data 作为回退
            if get_bars_fn is None and callable(mc_get_kline):
                get_bars_source = "MarketCenter.get_kline_data"

            if get_bars_fn is None and not callable(mc_get_kline):
                self.output("历史K线接口不可用：无 get_bars / get_kline_data，请检查行情源或接口支持")

            # 获取所有需要K线数据的合约
            all_instruments = []
            all_seen: Set[str] = set()
            
            # 添加期货合约
            for future in self.future_instruments:
                exchange = future.get("ExchangeID", "")
                instrument_id = future.get("InstrumentID", "")
                instrument_norm = self._normalize_future_id(instrument_id)
                # 仅拉取真实月份合约的历史K线，可选：只拉当前/下一月
                if exchange and instrument_id and self._is_real_month_contract(instrument_norm):
                    if self._resolve_subscribe_flag(
                        "subscribe_only_specified_month_futures",
                        "subscribe_only_current_next_futures",
                        False
                    ) and (not self._is_symbol_current_or_next(instrument_norm)):
                        continue
                    rec = {
                        "exchange": exchange,
                        "instrument_id": instrument_id,
                        "type": "future"
                    }
                    key = f"{exchange}_{str(instrument_id).upper()}"
                    if key not in all_seen:
                        all_seen.add(key)
                        all_instruments.append(rec)
            
            # 添加期权合约（可配置）。即使get_bars/get_kline 缺失，也尝试后续 get_recent_m1_kline 兜底，便于发现空数据
            if self.params.subscribe_options and getattr(self.params, "load_history_options", False):
                allow_only_current_next = self._resolve_subscribe_flag(
                    "subscribe_only_specified_month_options",
                    "subscribe_only_current_next_options",
                    False
                )
                allowed_future_symbols: Set[str] = set()
                if allow_only_current_next:
                    for fid in list(self.option_instruments.keys()):
                        fid_norm = self._normalize_future_id(fid)
                        if self._is_symbol_current_or_next(fid_norm.upper()):
                            allowed_future_symbols.add(fid_norm.upper())

                for future_symbol, options in self.option_instruments.items():
                    future_symbol_norm = self._normalize_future_id(future_symbol)
                    if allow_only_current_next and future_symbol_norm.upper() not in allowed_future_symbols:
                        continue
                    for option in options:
                        opt_exchange = option.get("ExchangeID", "")
                        opt_instrument = option.get("InstrumentID", "")
                        if opt_exchange and opt_instrument:
                            rec = {
                                "exchange": opt_exchange,
                                "instrument_id": opt_instrument,
                                "type": "option",
                                "_normalized_future": future_symbol_norm
                            }
                            key = f"{opt_exchange}_{str(opt_instrument).upper()}"
                            if key not in all_seen:
                                all_seen.add(key)
                                all_instruments.append(rec)
            
            fut_count = sum(1 for i in all_instruments if i.get("type") == "future")
            opt_count = sum(1 for i in all_instruments if i.get("type") == "option")
            self._debug(f"需要获取K线数据的合约总数: {len(all_instruments)} (期货 {fut_count}, 期权 {opt_count})，load_history_options={getattr(self.params, 'load_history_options', False)}")
            
            # 主动获取K线数据

            empty_keys: List[str] = []  # 收集空数据的合约，统一输出，避免刷屏
            non_empty_keys: List[str] = []  # 收集成功有数据的合约
            history_minutes = getattr(self.params, "history_minutes", 240) or 240
            # 多级回退窗口（分钟）：用户设置 -> 加倍 -> 1440（一天）
            fallback_windows = [history_minutes]
            if history_minutes < 720:
                fallback_windows.append(max(120, history_minutes * 2))
            if 1440 not in fallback_windows:
                fallback_windows.append(1440)
            for instrument in all_instruments:
                exchange = instrument["exchange"]
                instrument_id = instrument["instrument_id"]
                exch_upper = str(exchange).upper()
                inst_upper = str(instrument_id).upper()
                key = f"{exchange}_{instrument_id}"
                
                try:
                    bars = None
                    if callable(get_bars_fn):
                        try:
                            bars = get_bars_fn(
                                exchange=exchange,
                                instrument_id=instrument_id,
                                period=self.params.kline_style,
                                count=self.params.max_kline
                            )
                        except TypeError:
                            bars = get_bars_fn(
                                instrument_id=instrument_id,
                                period=self.params.kline_style,
                                count=self.params.max_kline
                            )
                    # 回退：使用MarketCenter.get_kline_data 按时间段获取（多级窗口重试）
                    if callable(mc_get_kline) and (not bars or len(bars) == 0):
                        for win in fallback_windows:
                            try:
                                end_dt = datetime.now()
                                start_dt = end_dt - timedelta(minutes=win)
                                bars = mc_get_kline(
                                    exchange=exchange,
                                    instrument_id=instrument_id,
                                    style=self.params.kline_style,
                                    start_time=start_dt,
                                    end_time=end_dt
                                )
                                if bars and len(bars) > 0:
                                    break
                            except Exception:
                                bars = []
                    # 第二层回退：如果按时间段仍为空，尝试使用count 负数取最近N根
                    if (not bars or len(bars) == 0) and callable(mc_get_kline):
                        try:
                            bars = mc_get_kline(
                                exchange=exchange,
                                instrument_id=instrument_id,
                                style=self.params.kline_style,
                                count=-(abs(self.params.max_kline))
                            )
                        except Exception:
                            bars = []
                    # 最终兜底：若上述均为空，尝试使用get_recent_m1_kline(count)
                    if (not bars or len(bars) == 0):
                        try:
                            recent = self.get_recent_m1_kline(exchange, instrument_id, count=self.params.max_kline)
                            if recent:
                                bars = recent
                                get_bars_source = get_bars_source or "get_recent_m1_kline"
                        except Exception:
                            pass

                    if bars and len(bars) > 0:
                        # 初始化K线数据
                        key_list = [
                            f"{exchange}_{instrument_id}",
                            f"{exchange}|{instrument_id}",
                            f"{exch_upper}_{inst_upper}",
                            f"{exch_upper}|{inst_upper}",
                        ]
                        key_list = list(dict.fromkeys(key_list))
                        for k in key_list:
                            if k not in self.kline_data:
                                self.kline_data[k] = {'generator': None, 'data': []}
                        
                        # 转换为轻量K线对象（避免构造KLineData 触发参数不兼容错误）
                        valid_cnt = 0
                        for bar in bars:
                            lk = self._to_light_kline(bar)
                            if getattr(lk, 'close', 0) and lk.close > 0:
                                for k in key_list:
                                    self.kline_data[k]['data'].append(lk)
                                valid_cnt += 1
                        
                        # 限制K线数量
                        for k in key_list:
                            if len(self.kline_data[k]['data']) > self.params.max_kline:
                                self.kline_data[k]['data'] = self.kline_data[k]['data'][-self.params.max_kline:]
                        
                        if valid_cnt > 0:
                            non_empty_keys.append(key)
                            self._debug(f"获取历史K线成功({get_bars_source}): {key}, 数量: {len(self.kline_data[key]['data'])}")
                        else:
                            empty_keys.append(key)
                            self._debug(f"历史K线全零/已过期 {key}")
                    else:
                        empty_keys.append(key)
                except Exception as e:
                    self._debug(f"获取历史K线失败{key}: {e}")
            
            # 汇总输出就绪与空数据情况，提升可见性
            ready_cnt = len(non_empty_keys)
            empty_cnt = len(empty_keys)
            if ready_cnt or empty_cnt:
                self.output(f"历史K线加载总结：就绪{ready_cnt} 个，空数据{empty_cnt} 个")
            if empty_keys:
                preview = ", ".join(empty_keys[:10])
                more = "" if len(empty_keys) <= 10 else f" 等共 {len(empty_keys)} 个"
                self.output(f"历史K线为空({get_bars_source}): {preview}{more}")
                # 最多重试两次：提升窗口到一天并再次调度获取
                if self.history_retry_count < 2:
                    try:
                        self.history_retry_done = True
                        self.history_retry_count += 1
                        self.params.history_minutes = max(history_minutes, 1440)
                        # 延迟2秒再次尝试，避免阻塞当前启动流程
                        self._safe_add_once_job(
                            job_id=f"retry_load_history_{self.history_retry_count}",
                            func=self.load_historical_klines,
                            delay_seconds=2
                        )
                        self.output(f"检测到历史K线为空，第{self.history_retry_count}次重试已安排（窗口>= 1440 分钟）")
                    except Exception:
                        pass
            # 不论成功与否，标记一次历史拉取已完成，避免反复刷屏
            self.history_loaded = True
            if ready_cnt > 0:
                self.output("=== 历史K线加载完成 ===")
            else:
                self.output("=== 历史K线加载失败 ===")
            self._debug(f"=== 历史K线数据获取完成 ===")
        except Exception as e:
            self._debug(f"获取历史K线数据失败 {e}\n{traceback.format_exc()}")
            self.output("=== 历史K线加载失败 ===")

    def print_kline_counts(self, limit: int = 10) -> None:
        """打印部分期货与期权的K线数量，便于快速核验"""
        try:
            # 确保分组键已规范化，避免统计漏计
            try:
                self._normalize_option_group_keys()
            except Exception:
                pass
            style = getattr(self.params, "kline_style", "M1")
            self.output("=== K线就绪情况快照 ===")
            # 汇总统计
            fut = []
            for f in self.future_instruments:
                ex = f.get("ExchangeID", "")
                fid = f.get("InstrumentID", "")
                if not ex or not fid:
                    continue
                fid_norm = self._normalize_future_id(fid)
                if self._resolve_subscribe_flag(
                    "subscribe_only_specified_month_futures",
                    "subscribe_only_current_next_futures",
                    False
                ):
                    if not self._is_symbol_current_or_next(fid_norm):
                        continue
                fut.append((ex, fid))
            fut_ready = sum(1 for exch, fid in fut if len(self._get_kline_series(exch, fid)) >= 2)
            fut_zero = [f"{ex}.{fid}" for ex, fid in fut if len(self._get_kline_series(ex, fid)) == 0][:5]
            self.output(f"期货合计 {len(fut)}个(>=2根) {fut_ready}，缺0根示例 {fut_zero}")

            # 期货明细（前 limit 个）
            for exch, fid in fut[:limit]:
                series = self._get_kline_series(exch, fid)
                self.output(f"期货 {exch}.{fid} K线数: {len(series)}")

            # 期权汇总与明细（仅指定月/指定下月分组键）
            keys = []
            seen_keys: Set[str] = set()
            for k in self.option_instruments.keys():
                kn = self._normalize_future_id(k)
                if self._is_symbol_current_or_next(kn.upper()) and kn not in seen_keys:
                    seen_keys.add(kn)
                    keys.append(kn)
            opt_items = []
            opt_seen: Set[str] = set()
            for k in keys:
                for opt in self.option_instruments.get(k, []):
                    ex = opt.get("ExchangeID", "")
                    oid = opt.get("InstrumentID", "")
                    key = f"{ex}_{self._normalize_future_id(oid)}"
                    if key in opt_seen:
                        continue
                    opt_seen.add(key)
                    opt_items.append((ex, oid))
            opt_ready = sum(1 for ex, oid in opt_items if len(self._get_kline_series(ex, oid)) >= 2)
            opt_zero = [f"{ex}.{oid}" for ex, oid in opt_items if len(self._get_kline_series(ex, oid)) == 0][:5]
            self.output(f"期权合计 {len(opt_items)}个(>=2根) {opt_ready}，缺0根示例 {opt_zero}")

            cnt = 0
            for ex, oid in opt_items:
                series = self._get_kline_series(ex, oid)
                self.output(f"期权 {ex}.{oid} K线数: {len(series)}")
                cnt += 1
                if cnt >= limit:
                    break
            self.output("=== K线快照结束 ===")
        except Exception as e:
            self.output(f"打印K线快照失败 {e}")

    def print_commodity_option_readiness(self, limit: int = 10) -> None:
        """打印商品期货指定月/指定下月期权的就绪摘要（>=2根统计），快速确认是否具备生成信号的前提"""
        try:
            try:
                self._normalize_option_group_keys()
            except Exception:
                pass
            main_ex = {"SHFE", "DCE", "CZCE"}
            self.output("=== 商品期权就绪摘要（指定月/指定下月） ===")
            rows = []
            seen_keys: Set[str] = set()
            seen_opts: Set[str] = set()
            for fut_symbol, options in self.option_instruments.items():
                # 仅指定月/指定下月
                fut_norm = self._normalize_future_id(str(fut_symbol))
                if fut_norm in seen_keys:
                    continue
                if not self._is_symbol_current_or_next(fut_norm.upper()):
                    continue
                seen_keys.add(fut_norm)
                # 聚合该键下的选项（仅主流商品交易所）
                total = 0
                ready = 0
                zero = 0
                sample_zero = []
                sample_ready = []
                for opt in options:
                    ex = opt.get("ExchangeID", "")
                    oid = opt.get("InstrumentID", "")
                    if not ex or not oid:
                        continue
                    if ex not in main_ex:
                        continue
                    opt_key = f"{ex}_{self._normalize_future_id(oid)}"
                    if opt_key in seen_opts:
                        continue
                    seen_opts.add(opt_key)
                    series = self._get_kline_series(ex, oid)
                    total += 1
                    if len(series) >= 2:
                        ready += 1
                        if len(sample_ready) < 2:
                            sample_ready.append(f"{ex}.{oid}")
                    elif len(series) == 0:
                        zero += 1
                        if len(sample_zero) < 2:
                            sample_zero.append(f"{ex}.{oid}")
                if total > 0:
                    rows.append((fut_symbol.upper(), total, ready, zero, sample_ready, sample_zero))
            # 输出摘要（限制行数）
            if not rows:
                self.output("无可摘要的商品期权（指定月/指定下月），请检查合约加载或过滤条件")
            else:
                rows.sort(key=lambda r: r[2], reverse=True)
                for fut_symbol, total, ready, zero, s_ready, s_zero in rows[:limit]:
                    extra = ""
                    if s_ready:
                        extra += f"，就绪示例 {s_ready}"
                    if s_zero:
                        extra += f"，缺0根示例 {s_zero}"
                    self.output(f"[{fut_symbol}] 期权合计 {total}个(>=2根) {ready}，缺0根 {zero}{extra}")
            self.output("=== 商品期权就绪摘要结束 ===")
        except Exception as e:
            self.output(f"打印商品期权就绪摘要失败: {e}")

# 兼容部分平台期望的类名
class Strategy(Strategy20260105_3):
    def print_filtered_readiness(self, symbol_prefix: str, exchanges: Optional[Set[str]] = None, limit: int = 20) -> None:
        """按品种前缀与交易所过滤，打印指定月/指定下月期货及其期权的K线就绪情况

        symbol_prefix: 品种前缀（例如"SR"）
        exchanges: 交易所集合（例如{"CZCE"}）；为空则不过滤交易所
        limit: 输出明细限制
        """
        try:
            sp = str(symbol_prefix or "").upper()
            exset = set(exchanges) if exchanges else None
            def _out(msg: str):
                # 仅在调试开关开启时输出/打印过滤摘要
                try:
                    if not bool(getattr(self.params, "debug_output", False)):
                        return
                except Exception:
                    return
                try:
                    self.output(msg)
                except Exception:
                    pass
                try:
                    print(msg)
                except Exception:
                    pass
            _out(f"=== 过滤就绪摘要 [{sp}] {sorted(exset) if exset else ''} ===")
            # 期货过滤与就绪
            futs = []
            fut_seen: Set[str] = set()
            for f in self.future_instruments:
                ex = f.get("ExchangeID", "")
                fid = f.get("InstrumentID", "")
                if not ex or not fid:
                    continue
                fid_norm = self._normalize_future_id(fid)
                if not fid_norm.startswith(sp):
                    continue
                if not self._is_symbol_current_or_next(fid_norm):
                    continue
                if exset and ex not in exset:
                    continue
                key = f"{ex}_{fid_norm}"
                if key in fut_seen:
                    continue
                fut_seen.add(key)
                series = self._get_kline_series(ex, fid)
                futs.append((ex, fid, len(series)))
            if futs:
                ready_cnt = sum(1 for _, _, n in futs if n >= 2)
                zero_cnt = sum(1 for _, _, n in futs if n == 0)
                _out(f"期货合计 {len(futs)}个(>=2根) {ready_cnt}，缺0根 {zero_cnt}")
                for ex, fid, n in futs[:limit]:
                    _out(f"期货 {ex}.{fid} K线数: {n}")
            else:
                _out("无匹配期货（指定月/指定下月）")

            # 期权过滤与就绪（按键）
            opt_total = 0
            opt_ready = 0
            opt_zero = 0
            opt_rows = []
            opt_group_seen: Set[str] = set()
            opt_inst_seen: Set[str] = set()
            for fut_symbol, options in self.option_instruments.items():
                fsu = self._normalize_future_id(str(fut_symbol))
                if not fsu.startswith(sp):
                    continue
                if not self._is_symbol_current_or_next(fsu):
                    continue
                if fsu in opt_group_seen:
                    continue
                opt_group_seen.add(fsu)
                total = 0
                ready = 0
                zero = 0
                detail = []
                for opt in options:
                    ex = opt.get("ExchangeID", "")
                    oid = opt.get("InstrumentID", "")
                    if not ex or not oid:
                        continue
                    if exset and ex not in exset:
                        continue
                    key = f"{ex}_{self._normalize_future_id(oid)}"
                    if key in opt_inst_seen:
                        continue
                    opt_inst_seen.add(key)
                    series = self._get_kline_series(ex, oid)
                    total += 1
                    if len(series) >= 2:
                        ready += 1
                    elif len(series) == 0:
                        zero += 1
                    dmsg = None
                    if len(detail) < limit:
                        dmsg = f"期权 {ex}.{oid} K线数: {len(series)}"
                    if dmsg:
                        detail.append(dmsg)
                if total > 0:
                    opt_total += total
                    opt_ready += ready
                    opt_zero += zero
                    opt_rows.append((fsu, total, ready, zero, detail))
            if opt_rows:
                self.output(f"期权合计 {opt_total}个(>=2根) {opt_ready}，缺0根 {opt_zero}")
                _out(f"期权合计 {opt_total}个(>=2根) {opt_ready}，缺0根 {opt_zero}")
                for fsu, total, ready, zero, detail in opt_rows[:limit]:
                    _out(f"[{fsu}] 期权 {total}个(>=2根) {ready}，缺0根 {zero}")
                    for d in detail[:limit]:
                        _out(d)
            else:
                _out("无匹配期权（指定月/指定下月）")
            _out("=== 过滤就绪摘要结束 ===")
        except Exception as e:
            try:
                self.output(f"打印过滤就绪摘要失败: {e}")
            except Exception:
                pass
            try:
                # 仅在调试开关开启时打印到控制台
                if bool(getattr(self.params, "debug_output", False)):
                    print(f"打印过滤就绪摘要失败: {e}")
            except Exception:
                pass
    
    def calculate_option_width(self, exchange: str, future_id: str) -> None:
        """计算期权宽度（已废弃，统一使用 calculate_option_width_optimized）"""
        self.calculate_option_width_optimized(exchange, future_id)
            
    def _is_option_sync_rising(self, option: Dict[str, Any], exchange: str, 
                              future_rising: bool, future_price: float) -> bool:
        """判断期权是否同步移动"""
        try:
            option_id = option["InstrumentID"]
            opt_id_upper = option_id.upper()
            
            # 检查是否是虚值期权
            if not self._is_out_of_money(option_id, future_price, option):
                self._debug(f"期权 {option_id} 不是虚值期权")
                return False
            
            # 检查是否有K线数据
            kline_list = self._get_kline_series(exchange, option_id)
            if len(kline_list) < 2:
                option_key = f"{exchange}_{option_id}"
                if option_key not in self.option_kline_insufficient_logged:
                    self.output(f"{exchange}.{option_id} K线不足，当前 {len(kline_list)} 根，需>=2 才能判定同步移动")
                    self.option_kline_insufficient_logged.add(option_key)
                return False
            
            # 判断期权是否上涨（允许平盘/微跌容忍，用于提高命中率）
            current_price = kline_list[-1].close
            previous_price = kline_list[-2].close
            option_change = current_price - previous_price
            sync_tolerance = float(getattr(self.params, "option_sync_tolerance", 0.5) or 0.0)
            allow_flat = bool(getattr(self.params, "option_sync_allow_flat", True))
            option_rising = (option_change > 0) or (allow_flat and option_change >= -sync_tolerance)
            
            # 判断是否同步移动
            is_call = "C" in opt_id_upper
            is_put = "P" in opt_id_upper
            
            self._debug(f"检查期权{option_id}: 期货{ '上涨' if future_rising else '下跌'}, 期权{ '上涨' if option_rising else '下跌'}, "
                      f"类型{ '看涨' if is_call else '看跌'}, "
                      f"价格变化: {previous_price:.2f} 到{current_price:.2f}")
            
            if future_rising and is_call:
                result = option_rising
                self._debug(f"  看涨期权在期货上涨时{ '同步移动' if result else '未同步移动'}")
                return result
            elif not future_rising and is_put:
                result = option_rising
                self._debug(f"  看跌期权在期货下跌时{ '同步移动' if result else '未同步移动'}")
                return result
            
            self._debug(f"  期权类型与期货趋势不匹配，不满足同步移动条件")
            return False
            
        except Exception as e:
            self.output(f"_is_option_sync_rising 错误: {e}\n{traceback.format_exc()}")
            return False

    def _is_out_of_money(self, option_symbol: str, future_price: float, option: Optional[Dict[str, Any]] = None) -> bool:
        """判断是否是虚值期权，修复逻辑错误"""
        
        # 诊断/测试开关
        if getattr(self.params, "ignore_otm_filter", False):
            if self.params.debug_output:
                self._debug(f"  忽略虚值筛选(ignore_otm_filter=True): {option_symbol}")
            return True
        
        if self.params.debug_output:
            self._debug(f"检查期权{option_symbol}，当前期货价格 {future_price:.2f}")
        
        # 优先使用结构化字段
        if isinstance(option, dict):
            sp = option.get("StrikePrice")
            ot = str(option.get("OptionType", "")).upper()
            
            if sp not in (None, ""):
                try:
                    strike_price = float(sp)
                    
                    # 关键修复：正确判断虚值
                    if ot in ("C", "CALL", "1"):
                        # 看涨期权：行权价 > 期货价格 = 虚值
                        is_otm = strike_price > future_price
                        if self.params.debug_output:
                            self._debug(f"  看涨期权(字段) - 行权价 {strike_price:.2f}, 期货价 {future_price:.2f}, 虚值判断 {is_otm}")
                        return is_otm
                    elif ot in ("P", "PUT", "2"):
                        # 看跌期权：行权价 < 期货价格 = 虚值
                        is_otm = strike_price < future_price
                        if self.params.debug_output:
                            self._debug(f"  看跌期权(字段) - 行权价 {strike_price:.2f}, 期货价 {future_price:.2f}, 虚值判断 {is_otm}")
                        return is_otm
                except Exception as e:
                    if self.params.debug_output:
                        self._debug(f"  解析行权价失败: {e}")
        
        # 从代码中解析
        option_symbol_upper = option_symbol.upper()
        
        # 尝试解析IO2601-C-4350格式
        match1 = re.search(r"[A-Z]{2}\d{4}-([CP])-(\d+(?:\.\d+)?)", option_symbol_upper)
        if match1:
            option_type = match1.group(1)
            strike_price = float(match1.group(2))
            
            if option_type == "C":
                is_otm = strike_price > future_price
                if self.params.debug_output:
                    self._debug(f"  看涨期权(IO格式) - 行权价 {strike_price:.2f}, 虚值判断 {is_otm}")
                return is_otm
            elif option_type == "P":
                is_otm = strike_price < future_price
                if self.params.debug_output:
                    self._debug(f"  看跌期权(IO格式) - 行权价 {strike_price:.2f}, 虚值判断 {is_otm}")
                return is_otm
        
        # 其他格式（保持原有逻辑，但修复判断）
        call_match = re.search(r"[Cc][-_]?(\d+(?:\.\d+)?)", option_symbol_upper)
        if call_match:
            strike_price = float(call_match.group(1))
            # 看涨期权：行权价 > 期货价格 = 虚值
            is_otm = strike_price > future_price
            if self.params.debug_output:
                self._debug(f"  看涨期权(C格式) - 行权价 {strike_price:.2f}, 虚值判断 {is_otm}")
            return is_otm
        
        put_match = re.search(r"[Pp][-_]?(\d+(?:\.\d+)?)", option_symbol_upper)
        if put_match:
            strike_price = float(put_match.group(1))
            # 看跌期权：行权价 < 期货价格 = 虚值
            is_otm = strike_price < future_price
            if self.params.debug_output:
                self._debug(f"  看跌期权(P格式) - 行权价 {strike_price:.2f}, 虚值判断 {is_otm}")
            return is_otm
        
        # 后缀格式：1200C 或 1200P
        alt_call = re.search(r"(\d+(?:\.\d+)?)[Cc]", option_symbol_upper)
        if alt_call:
            strike_price = float(alt_call.group(1))
            is_otm = strike_price > future_price
            if self.params.debug_output:
                self._debug(f"  看涨期权(后缀C) - 行权价 {strike_price:.2f}, 虚值判断 {is_otm}")
            return is_otm
        
        alt_put = re.search(r"(\d+(?:\.\d+)?)[Pp]", option_symbol_upper)
        if alt_put:
            strike_price = float(alt_put.group(1))
            is_otm = strike_price < future_price
            if self.params.debug_output:
                self._debug(f"  看跌期权(后缀P) - 行权价 {strike_price:.2f}, 虚值判断 {is_otm}")
            return is_otm
        
        if self.params.debug_output:
            self._debug(f"  未识别期权类型/行权价 {option_symbol}")
        return False
            
    def generate_trading_signals(self) -> List[Dict[str, Any]]:
        """
        生成并返回交易信号列表（不直接打印）
        """
        if not self.option_width_results:
            self._debug("无option_width 结果，暂不生成信号")
            return []
        min_width = int(getattr(self.params, "min_option_width", 1) or 1)
        allow_minimal = bool(getattr(self.params, "allow_minimal_signal", False))

        self._debug(f"当前 option_width_results: {len(self.option_width_results)} 条记录")
        for future_id, result in self.option_width_results.items():
            self._debug(f"  {future_id}: 宽度={result['option_width']}, 全部同步={result['all_sync']}, 趋势={result['future_rising']}")
        
        # 分组：全部同步移动和部分同步移动
        all_sync_results = []  # 全部同步移动
        partial_sync_results = []  # 部分同步移动
        
        for future_id, result in self.option_width_results.items():
            if not self._is_real_month_contract(str(future_id)):
                continue
            if not self._is_symbol_current(str(future_id)):
                continue
            prod = self._extract_product_code(str(future_id))
            if prod and not self._has_option_for_product(prod):
                continue
            if result["option_width"] < min_width:
                # 没有同步移动的虚值期权，不参与信号生成
                continue
            elif result["all_sync"]:
                # 全部同步移动
                all_sync_results.append((future_id, result))
            else:
                # 部分同步移动（有同步移动但不是全部）
                partial_sync_results.append((future_id, result))
        
        # 在各组内部按期权宽度排序
        all_sync_results.sort(key=lambda x: x[1]["option_width"], reverse=True)
        partial_sync_results.sort(key=lambda x: x[1]["option_width"], reverse=True)
        
        # 生成信号
        signals = []

        # 诊断输出：候选数量和详细信息
        self._debug(
            f"信号候选:全同步 {len(all_sync_results)} 部分同步: {len(partial_sync_results)}"
        )
        
        if all_sync_results:
            self._debug(f"全同步排序前3名")
            for i, (fut_id, res) in enumerate(all_sync_results[:3]):
                self._debug(f"  {i+1}. {fut_id}: 宽度={res['option_width']}, 价格={res['current_price']:.2f}, 趋势={res['future_rising']}")
        
        if partial_sync_results:
            self._debug(f"部分同步排序前3名")
            for i, (fut_id, res) in enumerate(partial_sync_results[:3]):
                self._debug(f"  {i+1}. {fut_id}: 宽度={res['option_width']}, 价格={res['current_price']:.2f}, 趋势={res['future_rising']}")
        
        # 原则A：全部同步移动中，取期权宽度最大者
        if all_sync_results:
            best_id, best_result = all_sync_results[0]
            signals.append({
                "future_id": best_id,
                "exchange": best_result["exchange"],
                "signal_type": "最优信号",
                "option_width": best_result["option_width"],
                "all_sync": True,
                "action": "买入" if best_result["future_rising"] else "卖出",
                "price": best_result["current_price"],
                "timestamp": best_result["timestamp"]
            })
            self._debug(f"生成最优信号 {best_id}, 期权宽度={best_result['option_width']}")
        
        # 原则B：次优原则
        if len(all_sync_results) > 1:
            # 情况1：有2个或更多全部同步移动品种
            # 最优信号：all_sync_results[0]（全部同步中宽度最大）
            # 情况2：次优信号：唯一的全部同步移动品种
            second_best_id, second_best_result = all_sync_results[1]
            signals.append({
                "future_id": second_best_id,
                "exchange": second_best_result["exchange"],
                "signal_type": "次优信号",
                "option_width": second_best_result["option_width"],
                "all_sync": True,
                "action": "买入" if second_best_result["future_rising"] else "卖出",
                "price": second_best_result["current_price"],
                "timestamp": second_best_result["timestamp"]
            })
            self._debug(f"生成次优信号(全同步): {second_best_id}, 期权宽度={second_best_result['option_width']}")
        elif len(all_sync_results) == 1 and partial_sync_results:
            # 只有1个全部同步移动品种且有部分同步品种时，不生成次优信号
            # 次优信号应该指向这个唯一的全部同步移动品种
            pass
        
        # 原则C：都不是全部同步移动，则取期权宽度值最大者
        if not all_sync_results and partial_sync_results:
            best_id, best_result = partial_sync_results[0]
            signals.append({
                "future_id": best_id,
                "exchange": best_result["exchange"],
                "signal_type": "部分同步信号",
                "option_width": best_result["option_width"],
                "all_sync": False,
                "action": "买入" if best_result["future_rising"] else "卖出",
                "price": best_result["current_price"],
                "timestamp": best_result["timestamp"]
            })
            self._debug(f"生成部分同步信号: {best_id}, 期权宽度={best_result['option_width']}")
        # 兜底：允许最小信号（仅根据期货趋势），用于验证链路与便捷观察
        if not signals and allow_minimal:
            try:
                # 选择价格变动绝对值最大者
                sorted_by_move = sorted(
                    self.option_width_results.items(),
                    key=lambda kv: abs(kv[1]["current_price"] - kv[1]["previous_price"]),
                    reverse=True
                )
                if sorted_by_move:
                    fut_id, res = sorted_by_move[0]
                    signals.append({
                        "future_id": fut_id,
                        "exchange": res["exchange"],
                        "signal_type": "最小信号",
                        "option_width": res["option_width"],
                        "all_sync": res.get("all_sync", False),
                        "action": "买入" if res["future_rising"] else "卖出",
                        "price": res["current_price"],
                        "timestamp": res["timestamp"]
                    })
                    self._debug(f"兜底生成最小信号 {fut_id}, 趋势={'上涨' if res['future_rising'] else '下跌'}")
            except Exception as e:
                self._debug(f"兜底最小信号生成失败 {e}")

        return signals

    def output_trading_signals(self, signals: List[Dict[str, Any]]) -> None:
        """格式化输出信号（由generate_trading_signals 返回）"""
        if signals:
            now = datetime.now()
            for signal in signals:
                key = f"{signal.get('exchange','')}|{signal.get('future_id','')}|{signal.get('signal_type','')}|{signal.get('action','')}"
                last_emit = self.signal_last_emit.get(key)
                if self.signal_cooldown_sec > 0 and last_emit:
                    if (now - last_emit).total_seconds() < self.signal_cooldown_sec:
                        self._debug(f"信号去重: {key} 在冷却时间内被忽略")
                        continue
                self.signal_last_emit[key] = now

                sync_status = "全部同步" if signal['all_sync'] else "部分同步"
                self.output(
                    f"交易信号 [{signal['signal_type']}]: "
                    f"{signal['exchange']}.{signal['future_id']} "
                    f"期权宽度: {signal['option_width']} "
                    f"同步状态 {sync_status} "
                    f"价格: {signal['price']} "
                    f"操作: {signal['action']}"
                )
        else:
            self.output("当前无交易信号")

    def get_current_signals(self) -> List[Dict[str, Any]]:
        """获取当前信号（返回信号列表，不输出）"""
        return self.generate_trading_signals()

    def get_position_manager_status(self) -> Optional[Dict[str, Any]]:
        """获取平仓管理器状态（查询接口）"""
        try:
            if not hasattr(self, 'position_manager') or not self.position_manager:
                return None
            return self.position_manager.get_manager_status()
        except Exception as e:
            self._debug(f"获取平仓管理器状态失败: {e}")
            return None

    def get_position_info(self) -> Optional[List[Dict[str, Any]]]:
        """获取仓位信息（查询接口）"""
        try:
            if not hasattr(self, 'position_manager') or not self.position_manager:
                return None
            return self.position_manager.get_position_info()
        except Exception as e:
            self._debug(f"获取仓位信息失败: {e}")
            return None




# 使用示例
if __name__ == "__main__":
    print("期权宽度交易信号生成器")
    print("严格遵循核心策略逻辑")
    print("1. 同步移动判断")
    print("2. 期权宽度计算（指定月 + 指定下月）")
    print("3. 交易信号判定三原则")
    print()
    print("使用示例：")
    print("1. 在交易平台中加载此策略")
    print("2. 配置参数（如API密钥、K线周期等）")
    print("3. 启动策略，策略会自动：")
    print("   - 订阅期货和期权行情")
    print("   - 计算期权宽度")
    print("   - 生成交易信号")
    print("4. 查看日志输出，获取交易信号")
    print()
    print("注意：此策略需要在交易平台环境中运行")
    print("直接运行此文件仅显示使用说明")