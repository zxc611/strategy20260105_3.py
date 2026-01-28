"""本地轻量占位 `pythongo.infini` 的备份副本（已重命名为 infini_placeholder）。

此文件为原占位实现的完整备份，便于在需要时恢复或比较差异。
"""

from typing import List, Any, Optional
import os
import importlib
import traceback

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
try:
    # 先尝试常见模块名，再尝试其它候选名，打印导入信息
    candidates = ["infini", "infini_sdk", "infini_api", "infinimarket", "infinipy"]
    for name in candidates:
        try:
            _real_infini = importlib.import_module(name)
            _is_proxy = True
            print(f"[pythongo.infini] proxied to real SDK module: {name} -> {_real_infini.__file__}")
            break
        except Exception:
            _real_infini = None
            _is_proxy = False
    if not _is_proxy:
        print("[pythongo.infini] no real SDK found among candidates; remaining as placeholder")


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
            return _real_infini.get_instruments(exchange=exchange)
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
            return _real_infini.get_instruments_by_product(exchange, product_id)
        except Exception:
            _log_placeholder("代理到真实 infini.get_instruments_by_product 时出错:\n" + traceback.format_exc())
            return []
    _log_placeholder("真实 infini SDK 未找到，get_instruments_by_product 返回空。")
    return []


def get_bars(exchange: str = None, instrument_id: str = None, period: str = "M1", count: int = 100) -> List[Any]:
    _update_creds_from_env()
    if _is_proxy and _real_infini is not None:
        try:
            return _real_infini.get_bars(exchange=exchange, instrument_id=instrument_id, period=period, count=count)
        except Exception:
            _log_placeholder("代理到真实 infini.get_bars 时出错:\n" + traceback.format_exc())
            return []
    _log_placeholder("真实 infini SDK 未找到，get_bars 返回空。")
    return []


def is_proxy() -> bool:
    """返回是否成功代理到真实 SDK。"""
    return _is_proxy
