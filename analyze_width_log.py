分析期权宽度计算日志
import re

log_file = r"c:\Users\xu\AppData\Roaming\InfiniTrader_SimulationX64\pyStrategy\demo\PythonGoLog__2026_01_12-11_46_44.txt"

try:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 查找期权数据为空的记录
    empty_pattern = r'期权数据为空.*?'
    empty_matches = re.findall(empty_pattern, content)
    
    # 查找计算完成的记录
    complete_pattern = r'计算完成.*?'
    complete_matches = re.findall(complete_pattern, content)
    
    # 查找期权宽度的记录
    width_pattern = r'期权宽度.*?'
    width_matches = re.findall(width_pattern, content)
    
    # 查找 IH、IM、商品期货的记录
    ih_pattern = r'IH\d{4}'
    im_pattern = r'IM\d{4}'
    cu_pattern = r'CU\d{4}'
    rb_pattern = r'RB\d{4}'
    m_pattern = r'M\d{4}'
    
    ih_matches = re.findall(ih_pattern, content)
    im_matches = re.findall(im_pattern, content)
    cu_matches = re.findall(cu_pattern, content)
    rb_matches = re.findall(rb_pattern, content)
    m_matches = re.findall(m_pattern, content)
    
    print("=" * 80)
    print("期权宽度计算日志分析")
    print("=" * 80)
    print(f"\n1. 期权数据为空的记录: {len(empty_matches)} 条")
    for i, match in enumerate(empty_matches[:10], 1):
        print(f"   {i}. {match}")
    if len(empty_matches) > 10:
        print(f"   ... 还有 {len(empty_matches) - 10} 条")
    
    print(f"\n2. 计算完成的记录: {len(complete_matches)} 条")
    for i, match in enumerate(complete_matches[:10], 1):
        print(f"   {i}. {match}")
    if len(complete_matches) > 10:
        print(f"   ... 还有 {len(complete_matches) - 10} 条")
    
    print(f"\n3. 期权宽度的记录: {len(width_matches)} 条")
    for i, match in enumerate(width_matches[:10], 1):
        print(f"   {i}. {match}")
    if len(width_matches) > 10:
        print(f"   ... 还有 {len(width_matches) - 10} 条")
    
    print(f"\n4. IH 期货记录: {len(ih_matches)} 条")
    print(f"   IH 合约: {set(ih_matches)}")
    
    print(f"\n5. IM 期货记录: {len(im_matches)} 条")
    print(f"   IM 合约: {set(im_matches)}")
    
    print(f"\n6. CU 期货记录: {len(cu_matches)} 条")
    print(f"   CU 合约: {set(cu_matches)}")
    
    print(f"\n7. RB 期货记录: {len(rb_matches)} 条")
    print(f"   RB 合约: {set(rb_matches)}")
    
    print(f"\n8. M 期货记录: {len(m_matches)} 条")
    print(f"   M 合约: {set(m_matches)}")
    
    # 查找期权合约代码
    option_pattern = r'[A-Z]{2,}\d{4}-[CP]-\d+'
    option_matches = re.findall(option_pattern, content)
    print(f"\n9. 期权合约记录: {len(set(option_matches))} 个唯一合约")
    print(f"   前20个期权合约: {sorted(set(option_matches))[:20]}")
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)
    
except Exception as e:
    print(f"分析失败: {e}")
    import traceback
    traceback.print_exc()
