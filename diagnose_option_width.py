"""
期权宽度计算诊断脚本
用于诊断期权宽度计算失败的原因
"""

import re
from typing import Dict, List, Any, Optional

def analyze_option_width_issue(future_price: float, 
                               current_options: List[Dict[str, Any]], 
                               next_options: List[Dict[str, Any]],
                               future_rising: bool) -> Dict[str, Any]:
    """
    分析期权宽度计算失败的原因
    
    Args:
        future_price: 期货价格
        current_options: 当月期权列表
        next_options: 下月期权列表
        future_rising: 期货是否上涨
    
    Returns:
        诊断结果字典
    """
    result = {
        "future_price": future_price,
        "future_rising": future_rising,
        "current_options_count": len(current_options),
        "next_options_count": len(next_options),
        "current_otm_count": 0,
        "next_otm_count": 0,
        "current_direction_otm_count": 0,
        "next_direction_otm_count": 0,
        "current_options_detail": [],
        "next_options_detail": [],
        "issues": []
    }
    
    # 分析当月期权
    for option in current_options:
        option_id = option.get("InstrumentID", "")
        strike_price = float(option.get("StrikePrice", 0) or 0)
        option_type = option.get("OptionType", "")
        
        # 从期权代码中解析类型（如果字典中没有）
        if not option_type:
            match = re.search(r"[A-Z]{2}\d{4}-([CP])-", option_id.upper())
            if match:
                option_type = match.group(1)
        
        # 判断是否为虚值
        is_otm = False
        if option_type.upper() == "C":
            # 看涨期权：行权价 > 期货价格 = 虚值
            is_otm = strike_price > future_price
        elif option_type.upper() == "P":
            # 看跌期权：行权价 < 期货价格 = 虚值
            is_otm = strike_price < future_price
        
        if is_otm:
            result["current_otm_count"] += 1
        
        # 判断是否为方向虚值
        is_direction_otm = False
        if future_rising and option_type.upper() == "C":
            is_direction_otm = is_otm
        elif not future_rising and option_type.upper() == "P":
            is_direction_otm = is_otm
        
        if is_direction_otm:
            result["current_direction_otm_count"] += 1
        
        result["current_options_detail"].append({
            "option_id": option_id,
            "strike_price": strike_price,
            "option_type": option_type,
            "is_otm": is_otm,
            "is_direction_otm": is_direction_otm
        })
    
    # 分析下月期权
    for option in next_options:
        option_id = option.get("InstrumentID", "")
        strike_price = float(option.get("StrikePrice", 0) or 0)
        option_type = option.get("OptionType", "")
        
        # 从期权代码中解析类型（如果字典中没有）
        if not option_type:
            match = re.search(r"[A-Z]{2}\d{4}-([CP])-", option_id.upper())
            if match:
                option_type = match.group(1)
        
        # 判断是否为虚值
        is_otm = False
        if option_type.upper() == "C":
            # 看涨期权：行权价 > 期货价格 = 虚值
            is_otm = strike_price > future_price
        elif option_type.upper() == "P":
            # 看跌期权：行权价 < 期货价格 = 虚值
            is_otm = strike_price < future_price
        
        if is_otm:
            result["next_otm_count"] += 1
        
        # 判断是否为方向虚值
        is_direction_otm = False
        if future_rising and option_type.upper() == "C":
            is_direction_otm = is_otm
        elif not future_rising and option_type.upper() == "P":
            is_direction_otm = is_otm
        
        if is_direction_otm:
            result["next_direction_otm_count"] += 1
        
        result["next_options_detail"].append({
            "option_id": option_id,
            "strike_price": strike_price,
            "option_type": option_type,
            "is_otm": is_otm,
            "is_direction_otm": is_direction_otm
        })
    
    # 诊断问题
    if result["current_direction_otm_count"] == 0:
        result["issues"].append("当月没有方向虚值期权")
        if result["current_otm_count"] == 0:
            result["issues"].append("当月没有虚值期权")
        else:
            result["issues"].append(f"当月有{result['current_otm_count']}个虚值期权，但不符合方向要求")
    
    if result["next_direction_otm_count"] == 0:
        result["issues"].append("下月没有方向虚值期权")
        if result["next_otm_count"] == 0:
            result["issues"].append("下月没有虚值期权")
        else:
            result["issues"].append(f"下月有{result['next_otm_count']}个虚值期权，但不符合方向要求")
    
    if result["current_options_count"] == 0:
        result["issues"].append("当月期权数据为空")
    
    if result["next_options_count"] == 0:
        result["issues"].append("下月期权数据为空")
    
    return result


def print_diagnosis_result(result: Dict[str, Any]) -> None:
    """打印诊断结果"""
    print("=" * 80)
    print("期权宽度计算诊断结果")
    print("=" * 80)
    print(f"期货价格: {result['future_price']}")
    print(f"期货方向: {'上涨' if result['future_rising'] else '下跌'}")
    print()
    print("当月期权:")
    print(f"  总数: {result['current_options_count']}")
    print(f"  虚值期权数: {result['current_otm_count']}")
    print(f"  方向虚值期权数: {result['current_direction_otm_count']}")
    print()
    print("下月期权:")
    print(f"  总数: {result['next_options_count']}")
    print(f"  虚值期权数: {result['next_otm_count']}")
    print(f"  方向虚值期权数: {result['next_direction_otm_count']}")
    print()
    print("问题诊断:")
    if result["issues"]:
        for issue in result["issues"]:
            print(f"  - {issue}")
    else:
        print("  未发现问题")
    print()
    
    # 打印详细信息
    if result["current_options_detail"]:
        print("当月期权详细信息:")
        for i, opt in enumerate(result["current_options_detail"][:10], start=1):
            print(f"  [{i}] {opt['option_id']} 类型={opt['option_type']} 行权价={opt['strike_price']} "
                  f"虚值={opt['is_otm']} 方向虚值={opt['is_direction_otm']}")
        if len(result["current_options_detail"]) > 10:
            print(f"  ... 还有 {len(result['current_options_detail']) - 10} 个期权")
        print()
    
    if result["next_options_detail"]:
        print("下月期权详细信息:")
        for i, opt in enumerate(result["next_options_detail"][:10], start=1):
            print(f"  [{i}] {opt['option_id']} 类型={opt['option_type']} 行权价={opt['strike_price']} "
                  f"虚值={opt['is_otm']} 方向虚值={opt['is_direction_otm']}")
        if len(result["next_options_detail"]) > 10:
            print(f"  ... 还有 {len(result['next_options_detail']) - 10} 个期权")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    # 示例：模拟期货价格1205.00，上涨的情况
    future_price = 1205.00
    future_rising = True
    
    # 模拟当月期权数据
    current_options = [
        {"InstrumentID": "MO2601-C-1200", "StrikePrice": 1200.0, "OptionType": "C"},
        {"InstrumentID": "MO2601-C-1250", "StrikePrice": 1250.0, "OptionType": "C"},
        {"InstrumentID": "MO2601-P-1150", "StrikePrice": 1150.0, "OptionType": "P"},
        {"InstrumentID": "MO2601-P-1200", "StrikePrice": 1200.0, "OptionType": "P"},
    ]
    
    # 模拟下月期权数据
    next_options = [
        {"InstrumentID": "MO2602-C-1200", "StrikePrice": 1200.0, "OptionType": "C"},
        {"InstrumentID": "MO2602-C-1250", "StrikePrice": 1250.0, "OptionType": "C"},
        {"InstrumentID": "MO2602-P-1150", "StrikePrice": 1150.0, "OptionType": "P"},
        {"InstrumentID": "MO2602-P-1200", "StrikePrice": 1200.0, "OptionType": "P"},
    ]
    
    result = analyze_option_width_issue(future_price, current_options, next_options, future_rising)
    print_diagnosis_result(result)
