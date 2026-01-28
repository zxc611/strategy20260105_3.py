诊断 IH/IM/商品期权宽度计算问题
import re

log_file = r"c:\Users\xu\AppData\Roaming\InfiniTrader_SimulationX64\pyStrategy\demo\PythonGoLog__2026_01_12-11_46_44.txt"

try:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print("=" * 80)
    print("期权宽度计算问题诊断")
    print("=" * 80)
    
    # 统计各种类型的日志
    is_out_of_money_count = 0
    option_empty_count = 0
    calc_complete_count = 0
    calc_failed_count = 0
    
    # 提取期权代码
    option_codes = set()
    future_codes = set()
    
    for line in lines:
        if '_is_out_of_money_optimized' in line:
            is_out_of_money_count += 1
            # 提取期权代码
            match = re.search(r'option_symbol=([A-Z]{2,}\d{4}-[CP]-\d+)', line)
            if match:
                option_codes.add(match.group(1))
        
        if '期权数据为空' in line:
            option_empty_count += 1
        
        if '计算完成:' in line:
            calc_complete_count += 1
            # 提取期货代码
            match = re.search(r'计算完成:\s+(\w{2,}\d{4})', line)
            if match:
                future_codes.add(match.group(1))
        
        if '计算期权宽度失败' in line:
            calc_failed_count += 1
    
    print(f"\n1. 日志统计:")
    print(f"   _is_out_of_money_optimized 调用次数: {is_out_of_money_count}")
    print(f"   期权数据为空次数: {option_empty_count}")
    print(f"   计算完成次数: {calc_complete_count}")
    print(f"   计算失败次数: {calc_failed_count}")
    
    print(f"\n2. 期权代码统计:")
    print(f"   发现的期权代码: {sorted(option_codes)}")
    
    print(f"\n3. 期货代码统计:")
    print(f"   计算完成的期货代码: {sorted(future_codes)}")
    
    # 分析期权代码的品种
    print(f"\n4. 期权品种分析:")
    option_products = {}
    for code in option_codes:
        match = re.match(r'([A-Z]{2,})\d{4}', code)
        if match:
            product = match.group(1)
            option_products[product] = option_products.get(product, 0) + 1
    
    for product, count in sorted(option_products.items()):
        print(f"   {product}: {count} 个期权")
    
    # 分析期货代码的品种
    print(f"\n5. 期货品种分析:")
    future_products = {}
    for code in future_codes:
        match = re.match(r'([A-Z]{2,})\d{4}', code)
        if match:
            product = match.group(1)
            future_products[product] = future_products.get(product, 0) + 1
    
    for product, count in sorted(future_products.items()):
        print(f"   {product}: {count} 个期货")
    
    # 检查映射关系
    print(f"\n6. 映射关系检查:")
    print("   期权品种 -> 期货品种（根据平台实盘实际情况）:")
    print("   HO -> IH")
    print("   IO -> IF")
    print("   MO -> IM")
    print("   商品期权（CU、M、Y、A、RB、AU、AG、CF、SR、TA等）-> 相同品种")
    
    # 诊断结论
    print(f"\n7. 诊断结论:")
    if option_empty_count > 0:
        print("   ❌ 存在期权数据为空的情况，可能是期权键归一化失败")
    elif calc_complete_count == 0:
        print("   ❌ 没有计算完成的记录，可能是宽度计算被异常中断")
    elif is_out_of_money_count > 0 and calc_complete_count > 0:
        print("   ✅ 期权虚值判断和宽度计算都在进行")
    else:
        print("   ⚠️  无法确定问题原因")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
    
except Exception as e:
    print(f"诊断失败: {e}")
    import traceback
    traceback.print_exc()
