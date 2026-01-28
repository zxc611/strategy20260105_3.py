"""本地轻量占位 `pythongo.infini`。

行为：
- 优先尝试代理到系统范围内的真实 `infini` SDK（如果可用），将外部调用转发给真实 SDK；
- 如果真实 SDK 不存在，则保持兼容的占位实现，但在调用时输出明确的错误信息（便于诊断），不再悄悄返回空列表。

保留本地 `pythongo` 的同时，尽量使用平台 SDK。此文件读取环境变量 `INFINI_API_KEY` / `INFINI_ACCESS_KEY` / `INFINI_ACCESS_SECRET`。
"""

from typing import List, Any, Optional
import os
import importlib
import traceback
import sys
from datetime import datetime, timedelta

try:
    from pythongo.core import MarketCenter
except Exception:
    MarketCenter = None

# 导出凭证变量（会反映当前环境变量）
api_key = os.environ.get("INFINI_API_KEY")
API_KEY = api_key
access_key = os.environ.get("INFINI_ACCESS_KEY") or os.environ.get("ACCESS_KEY")
ACCESS_KEY = access_key
access_secret = os.environ.get("INFINI_ACCESS_SECRET") or os.environ.get("ACCESS_SECRET")
ACCESS_SECRET = access_secret

# 尝试导入系统范围的真实 SDK（模块名 'infini'）
_real_infini = None
_is_proxy = False
_fallback_market_center: Optional[Any] = None
try:
    # 先尝试常见模块名，再尝试其它候选名，打印导入信息
    candidates = ["infini", "infini_sdk", "infini_api", "infinimarket", "infinipy"]
    for name in candidates:
        try:
            _real_infini = importlib.import_module(name)
            
            # Prevent self-proxying (infinite recursion)
            if hasattr(_real_infini, '__file__') and _real_infini.__file__ and os.path.abspath(_real_infini.__file__) == os.path.abspath(__file__):
                _real_infini = None
                continue

            _is_proxy = True
            print(f"[pythongo.infini] proxied to real SDK module: {name} -> {_real_infini.__file__}")
            break
        except Exception:
            _real_infini = None
            _is_proxy = False
    if not _is_proxy:
        # 若当前虚拟环境下未安装 SDK，尝试从系统 Python 的 site-packages 回退导入
        try:
            import sysconfig
            possible_sites = []
            # 常见的系统 site-packages 路径候选（Windows）
            base_prefix = getattr(sys, 'base_prefix', None) or getattr(sys, 'real_prefix', None)
            if base_prefix:
                possible_sites.append(os.path.join(base_prefix, 'Lib', 'site-packages'))
            # 运行时前缀
            exec_prefix = getattr(sys, 'exec_prefix', None)
            if exec_prefix and exec_prefix != base_prefix:
                possible_sites.append(os.path.join(exec_prefix, 'Lib', 'site-packages'))
            # 还尝试 sysconfig 获取路径
            try:
                sitep = sysconfig.get_paths().get('purelib')
                if sitep:
                    possible_sites.append(sitep)
            except Exception:
                pass

            # 将可用的系统 site-packages 插入 sys.path 并重试导入候选模块
            for sitep in possible_sites:
                try:
                    if sitep and os.path.isdir(sitep) and sitep not in sys.path:
                        sys.path.insert(0, sitep)
                        for name in candidates:
                            try:
                                _real_infini = importlib.import_module(name)
                                _is_proxy = True
                                print(f"[pythongo.infini] proxied to real SDK module (from system site-packages): {name} -> {_real_infini.__file__}")
                                break
                            except Exception:
                                _real_infini = None
                                _is_proxy = False
                        if _is_proxy:
                            break
                except Exception:
                    continue

            if not _is_proxy:
                print("[pythongo.infini] no real SDK found among candidates; remaining as placeholder")
        except Exception:
            _real_infini = None
            _is_proxy = False
            try:
                print("[pythongo.infini] proxy detection encountered an error; falling back to placeholder")
            except Exception:
                pass
except Exception:
    _real_infini = None
    _is_proxy = False
    try:
        print("[pythongo.infini] proxy detection failed; placeholder active")
    except Exception:
        pass

def _update_creds_from_env():
    global api_key, API_KEY, access_key, ACCESS_KEY, access_secret, ACCESS_SECRET
    api_key = os.environ.get("INFINI_API_KEY")
    API_KEY = api_key
    access_key = os.environ.get("INFINI_ACCESS_KEY") or os.environ.get("ACCESS_KEY")
    ACCESS_KEY = access_key
    access_secret = os.environ.get("INFINI_ACCESS_SECRET") or os.environ.get("ACCESS_SECRET")
    ACCESS_SECRET = access_secret


def _log_placeholder(msg: str) -> None:
    try:
        print(f"[pythongo.infini placeholder] {msg}")
    except Exception:
        pass


def _ensure_market_center() -> Optional[Any]:
    global _fallback_market_center
    if MarketCenter is None:
        return None
    if _fallback_market_center is None:
        try:
            _fallback_market_center = MarketCenter()
        except Exception:
            _fallback_market_center = None
    return _fallback_market_center


# 在模块导入时打印当前可见的凭证与 local_secrets.json（若存在），便于策略启动日志记录诊断信息
try:
    _update_creds_from_env()
    print("[pythongo.infini] imported. env API_KEY:", API_KEY, "ACCESS_KEY set:", bool(ACCESS_KEY))
except Exception:
    pass

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # local_secrets.json 在 demo 目录上一级
    local_path = os.path.normpath(os.path.join(base_dir, "..", "local_secrets.json"))
    if os.path.exists(local_path):
        try:
            import json
            with open(local_path, "r", encoding="utf-8") as _f:
                _ls = json.load(_f)
            print("[pythongo.infini] local_secrets.json read; keys present:", {k: bool(_ls.get(k)) for k in ("infini_api_key", "access_key", "access_secret")})
        except Exception:
            pass
except Exception:
    pass


def get_instruments(exchange: Optional[str] = None) -> List[Any]:
    """返回交易所下的合约列表，代理到真实 SDK（若可用）。"""
    _update_creds_from_env()
    if _is_proxy and _real_infini is not None:
        try:
            # 兼容多种真实 SDK 命名：优先使用 get_instruments，其次尝试常见替代名
            for fn in ('get_instruments', 'list_instruments', 'instruments', 'get_all_instruments'):
                f = getattr(_real_infini, fn, None)
                if callable(f):
                    try:
                        # 尝试以关键字参数调用，回退到位置参数
                        try:
                            return f(exchange=exchange)
                        except TypeError:
                            return f(exchange,)
                    except Exception:
                        # 若调用失败，继续尝试下一个候选
                        continue
            # 找不到可调用的接口
            raise AttributeError('no compatible get_instruments function found on proxied SDK')
        except Exception:
            _log_placeholder("代理到真实 infini.get_instruments 时出错:\n" + traceback.format_exc())
            return []
    # 未找到真实 SDK：打印诊断信息（便于排查），并返回空列表以保持兼容
    _log_placeholder("真实 infini SDK 未找到，get_instruments 返回空。请在运行环境中安装/提供平台 SDK。")
    return []


def get_instruments_by_product(exchange: str, product_id: str) -> List[Any]:
    _update_creds_from_env()
    if _is_proxy and _real_infini is not None:
        try:
            # 兼容多种真实 SDK 命名
            for fn in ('get_instruments_by_product', 'get_instruments_by_prod', 'list_by_product', 'instruments_by_product'):
                f = getattr(_real_infini, fn, None)
                if callable(f):
                    try:
                        return f(exchange, product_id)
                    except Exception:
                        continue
            raise AttributeError('no compatible get_instruments_by_product function found on proxied SDK')
        except Exception:
            _log_placeholder("代理到真实 infini.get_instruments_by_product 时出错:\n" + traceback.format_exc())
            return []
    mc = _ensure_market_center()
    if mc is not None:
        try:
            return mc.get_instruments_by_product(exchange=exchange, product_id=product_id)
        except Exception as exc:
            _log_placeholder(f"MarketCenter.get_instruments_by_product 兜底失败: {exc}")
    _log_placeholder("真实 infini SDK 未找到，get_instruments_by_product 返回空。")
    return []


def get_bars(exchange: str = None, instrument_id: str = None, period: str = "M1", count: int = 100) -> List[Any]:
    _update_creds_from_env()
    if _is_proxy and _real_infini is not None:
        try:
            # 兼容多种真实 SDK 命名
            for fn in ('get_bars', 'get_klines', 'fetch_bars', 'bars'):
                f = getattr(_real_infini, fn, None)
                if callable(f):
                    try:
                        res = None
                        try:
                            res = f(exchange=exchange, instrument_id=instrument_id, period=period, count=count)
                        except TypeError:
                            # 位置参数尝试
                            res = f(exchange, instrument_id, period, count)
                        
                        # 如果获取到有效数据，直接返回
                        if res and len(res) > 0:
                            return res
                        # 如果返回空数据，中断尝试，允许 fallback 到 Mock
                        break
                    except Exception:
                        continue
            # 若循环结束仍未返回（报错或空），则自然进入下方 fallback
        except Exception:
            _log_placeholder("代理到真实 infini.get_bars 时出错:\n" + traceback.format_exc())
            pass
    mc = _ensure_market_center()
    if mc is not None:
        try:
            # 计算时间窗口以适配 MarketCenter 接口
            end_dt = datetime.now()
            delta_mins = 1
            if period == 'M1': delta_mins = 1
            elif period == 'M3': delta_mins = 3
            elif period == 'M5': delta_mins = 5
            elif period == 'M15': delta_mins = 15
            elif period == 'M30': delta_mins = 30
            elif period == 'H1': delta_mins = 60
            elif period == 'D1': delta_mins = 1440
            
            # 为了确保有足够的数据，多给一点缓冲
            total_minutes = delta_mins * (count + 5)
            start_dt = end_dt - timedelta(minutes=total_minutes)
            
            return mc.get_kline_data(
                exchange=exchange,
                instrument_id=instrument_id,
                style=period,
                start_time=start_dt,
                end_time=end_dt
            )
        except Exception as exc:
            _log_placeholder(f"MarketCenter.get_kline_data 兜底失败: {exc}")
    _log_placeholder("真实 infini SDK 未找到，get_bars 返回空。")
    return []


def is_proxy() -> bool:
    """返回是否成功代理到真实 SDK。"""
    return _is_proxy
