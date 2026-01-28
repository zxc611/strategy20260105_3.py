#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟 pythongo.utils 模块 - 用于独立调试环境
无限易平台专有模块的临时替代品，用于在普通 Python 环境中运行和测试策略
"""

import threading
import time

class KLineGenerator:
    """K线生成器类"""
    def __init__(self, callback=None, real_time_callback=None, exchange=None, instrument_id=None, style=None):
        print(f"KLineGenerator 初始化 - 虚拟实现: 合约 {instrument_id}, 周期 {style}")
        self.callback = callback
        self.real_time_callback = real_time_callback
        self.exchange = exchange
        self.instrument_id = instrument_id
        self.style = style
    
    def push_history_data(self):
        """推送历史数据"""
        print("KLineGenerator.push_history_data 被调用")
    
    def tick_to_kline(self, tick):
        """将Tick数据转换为K线"""
        print("KLineGenerator.tick_to_kline 被调用")

class Scheduler:
    """简易的任务调度实现 (Threaded)，支持 interval 触发"""
    def __init__(self, name: str = "PythonGO"):
        self._jobs = {}
        self._name = name
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # print(f"[{self._name}] Local Scheduler started.")

    def _run_loop(self):
        while True:
            with self._lock:
                if not self._running:
                    break
                now = time.time()
                current_jobs = list(self._jobs.items())

            for job_id, job in current_jobs:
                # 初始化下次运行时间
                if "next_run" not in job:
                    job["next_run"] = now + job.get("seconds", 1)
                
                # 检查是否到期
                if now >= job["next_run"]:
                    try:
                        func = job["func"]
                        kwargs = job["kwargs"] or {}
                        # 在独立线程执行任务，避免阻塞调度循环
                        threading.Thread(target=func, kwargs=kwargs, daemon=True).start()
                    except Exception as e:
                        print(f"Scheduler error running job {job_id}: {e}")
                    
                    # 更新下次运行时间 (简单间隔)
                    with self._lock:
                        if job_id in self._jobs:
                            self._jobs[job_id]["next_run"] = time.time() + self._jobs[job_id].get("seconds", 1)
            
            time.sleep(0.5)

    def add_job(self, func, trigger: str = "interval", id: str = None, seconds: int = 1, run_date=None, kwargs=None):
        if id is None:
            id = f"job_{len(self._jobs)+1}"
        
        with self._lock:
            self._jobs[id] = {
                "func": func,
                "trigger": trigger,
                "seconds": max(1, seconds),
                "run_date": run_date,
                "kwargs": kwargs or {},
                "next_run": time.time() + max(1, seconds) 
            }
            # print(f"Job added: {id}, interval={seconds}s")
        return id

    def remove_job(self, id: str):
        with self._lock:
            self._jobs.pop(id, None)

    def remove_all_jobs(self):
        with self._lock:
            self._jobs.clear()

    def shutdown(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        # print("Scheduler stopped.")

    # 兼容别名
    def stop(self):
        self.shutdown()
    def clear(self):
        self.remove_all_jobs()
    def cancel_all(self):
        self.remove_all_jobs()

# 导出所有类
__all__ = ['KLineGenerator', 'Scheduler']
