"""
简化版期权宽度交易信号生成器 - Strategy20260105_3.py
最后修改时间: 2026-01-09 23:55:00
版本: v2026-01-09-2355
"""

from datetime import datetime, timedelta
import os
import json
import traceback
import types
import re
from typing import Dict, List, Any, Optional

from pythongo import infini
from pythongo.base import BaseStrategy, BaseParams, Field

try:
    class Params(BaseParams):
        debug_output: bool = Field(bool, default=True, title="是否输出调试信息")
        min_option_width: int = Field(int, default=1, title="最小期权宽度")
        allow_minimal_signal: bool = Field(bool, default=False, title="允许最小信号")
        option_sync_tolerance: float = Field(float, default=0.5, title="期权同步容忍度")
        option_sync_allow_flat: bool = Field(bool, default=True, title="允许平走")
        ignore_otm_filter: bool = Field(bool, default=False, title="忽略虚值筛选")
except Exception:
    from dataclasses import dataclass
    @dataclass
    class Params:
        debug_output: bool = True
        min_option_width: int = 1
        allow_minimal_signal: bool = False
        option_sync_tolerance: float = 0.5
        option_sync_allow_flat: bool = True
        ignore_otm_filter: bool = False

class Strategy20260105_3(BaseStrategy):
    """简化版策略 - 用于测试平台加载"""
    
    def __init__(self):
        super().__init__()
        self.params = Params()
        self.trading = False
        self.started = False
        self.is_running = False
        
        self.option_width_results = {}
        self.signal_last_emit = {}
        self.signal_cooldown_sec = 60
        self.option_kline_insufficient_logged = set()
        
        self.output("=== 简化版策略初始化 ===")
        self.output(f"策略名称: Strategy20260105_3")
        self.output(f"初始化时间: {datetime.now()}")
    
    def output(self, message: str) -> None:
        """输出日志"""
        try:
            super().output(message)
        except Exception:
            pass
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def _debug(self, message: str) -> None:
        """调试输出"""
        if self.params.debug_output:
            self.output(f"[DEBUG] {message}")
    
    def on_start(self) -> None:
        """策略启动方法"""
        self.output("=== 2026-01-09 23:55 简化版策略已加载 ===")
        self.output("=== on_start 被调用 ===")
        
        try:
            self.trading = True
            self.started = True
            self.is_running = True
            
            self.output("=== 简化版策略启动成功 ===")
            self.output(f"trading = {self.trading}")
            self.output(f"started = {self.started}")
            self.output(f"is_running = {self.is_running}")
            
        except Exception as e:
            self.output(f"on_start 异常: {e}")
            self.output(traceback.format_exc())
    
    def on_stop(self) -> None:
        """策略停止方法"""
        self.output("=== on_stop 被调用 ===")
        self.trading = False
        self.is_running = False
        self.output("=== 简化版策略已停止 ===")
    
    def on_tick(self, tick_data: Dict[str, Any]) -> None:
        """Tick数据回调"""
        if not self.is_running:
            return
        
        if self.params.debug_output:
            self._debug(f"收到Tick: {tick_data.get('InstrumentID', 'Unknown')}")
    
    def on_bar(self, bar_data: Dict[str, Any]) -> None:
        """K线数据回调"""
        if not self.is_running:
            return
        
        if self.params.debug_output:
            self._debug(f"收到K线: {bar_data.get('InstrumentID', 'Unknown')}")
    
    def generate_trading_signals(self) -> List[Dict[str, Any]]:
        """生成交易信号（简化版）"""
        signals = []
        
        if not self.option_width_results:
            self._debug("无 option_width 结果，暂不生成信号")
            return signals
        
        for future_id, result in self.option_width_results.items():
            if result["option_width"] >= self.params.min_option_width:
                signals.append({
                    "future_id": future_id,
                    "exchange": result.get("exchange", ""),
                    "signal_type": "简化版信号",
                    "option_width": result["option_width"],
                    "all_sync": result.get("all_sync", False),
                    "action": "买入" if result.get("future_rising", True) else "卖出",
                    "price": result.get("current_price", 0),
                    "timestamp": result.get("timestamp", datetime.now())
                })
        
        return signals
    
    def output_trading_signals(self, signals: List[Dict[str, Any]]) -> None:
        """输出交易信号"""
        if signals:
            for signal in signals:
                sync_status = "全部同步" if signal['all_sync'] else "部分同步"
                self.output(
                    f"交易信号 [{signal['signal_type']}]: "
                    f"{signal['exchange']}.{signal['future_id']} "
                    f"期权宽度: {signal['option_width']} "
                    f"同步状态: {sync_status} "
                    f"价格: {signal['price']} "
                    f"操作: {signal['action']}"
                )
        else:
            self.output("当前无交易信号")


if __name__ == "__main__":
    print("简化版期权宽度交易信号生成器")
    print("策略名称: Strategy20260105_3")
    print("用于测试平台加载功能")
