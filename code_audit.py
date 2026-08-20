#!/usr/bin/env python3
"""
代码审计：分析 Background Gap 与 Signed Background Effect 的数学一致性

目标：通过代码分析找出两个函数不一致的原因
"""

def analyze_code_differences():
    """分析两个函数的代码差异"""
    print("="*80)
    print("🔍 代码审计：分析 Background Gap 与 Signed Background Effect 的数学一致性")
    print("="*80)
    
    print("\n📋 分析两个函数的实现差异")
    print("="*80)
    
    # 分析 compute_background_gap_real 函数
    print("\n1️⃣ 旧版函数：compute_background_gap_real()")
    print("-"*50)
    print("""
def compute_background_gap_real(model, group_loaders, device='cpu'):
    # 水鸟：真实类别=0，比较水背景 vs 陆地背景
    probs_ww, _, _ = collect_probs_preds(model, group_loaders['waterbird_water'], device)
    probs_wl, _, _ = collect_probs_preds(model, group_loaders['waterbird_land'], device)
    
    if len(probs_ww) > 0 and len(probs_wl) > 0:
        p_waterbird_given_water_bg = probs_ww[:, 0].mean().item()
        p_waterbird_given_land_bg = probs_wl[:, 0].mean().item()
        bg_gap_waterbird = abs(p_waterbird_given_water_bg - p_waterbird_given_land_bg)
    
    # 陆鸟：真实类别=1，比较陆地背景 vs 水背景
    probs_ll, _, _ = collect_probs_preds(model, group_loaders['landbird_land'], device)
    probs_lw, _, _ = collect_probs_preds(model, group_loaders['landbird_water'], device)
    
    if len(probs_ll) > 0 and len(probs_lw) > 0:
        p_landbird_given_land_bg = probs_ll[:, 1].mean().item()
        p_landbird_given_water_bg = probs_lw[:, 1].mean().item()
        bg_gap_landbird = abs(p_landbird_given_land_bg - p_landbird_given_water_bg)
    
    background_gap = (bg_gap_waterbird + bg_gap_landbird) / 2
    return background_gap
""")
    
    # 分析 compute_signed_background_effect 函数
    print("\n2️⃣ 新版函数：compute_signed_background_effect()")
    print("-"*50)
    print("""
def compute_signed_background_effect(model, group_loaders, device='cpu'):
    # 水鸟：真实类别=0，比较水背景 vs 陆地背景
    probs_ww, _, _ = collect_probs_preds(model, group_loaders['waterbird_water'], device)
    probs_wl, _, _ = collect_probs_preds(model, group_loaders['waterbird_land'], device)
    
    if len(probs_ww) > 0 and len(probs_wl) > 0:
        p_waterbird_given_water_bg = probs_ww[:, 0].mean().item()
        p_waterbird_given_land_bg = probs_wl[:, 0].mean().item()
        delta_waterbird = p_waterbird_given_water_bg - p_waterbird_given_land_bg
    
    # 陆鸟：真实类别=1，比较陆地背景 vs 水背景
    probs_ll, _, _ = collect_probs_preds(model, group_loaders['landbird_land'], device)
    probs_lw, _, _ = collect_probs_preds(model, group_loaders['landbird_water'], device)
    
    if len(probs_ll) > 0 and len(probs_lw) > 0:
        p_landbird_given_land_bg = probs_ll[:, 1].mean().item()
        p_landbird_given_water_bg = probs_lw[:, 1].mean().item()
        delta_landbird = p_landbird_given_land_bg - p_landbird_given_water_bg
    
    return delta_waterbird, delta_landbird
""")
    
    # 分析数学关系
    print("\n3️⃣ 数学关系分析")
    print("-"*50)
    print("""
根据函数定义：
- 旧版 Background Gap = (|Δ_waterbird| + |Δ_landbird|) / 2
- 新版返回 = (Δ_waterbird, Δ_landbird)

其中：
- Δ_waterbird = P(水鸟|水背景) - P(水鸟|陆地背景)
- Δ_landbird = P(陆鸟|陆地背景) - P(陆鸟|水背景)

理论上应该满足：
Background_gap = (|Δ_waterbird| + |Δ_landbird|) / 2
""")
    
    # 潜在问题分析
    print("\n4️⃣ 潜在问题分析")
    print("-"*50)
    
    print("🔍 可能的差异来源：")
    print("1. 数据加载不一致:")
    print("   - 两个函数是否使用相同的 group loaders")
    print("   - 数据 shuffle 是否不同")
    print("   - batch size 是否不同")
    print("   - 数据增强是否不同")
    
    print("\n2. 概率计算方式不同:")
    print("   - 是否都使用了 F.softmax")
    print("   - 是否都使用了相同的维度")
    print("   - 是否都使用了 .mean()")
    
    print("\n3. 样本配对方式不同:")
    print("   - 旧版函数是否使用了配对样本")
    print("   - 新版函数是否使用了独立样本")
    print("   - 样本数量是否一致")
    
    print("\n4. 数值精度问题:")
    print("   - 浮点数精度差异")
    print("   - 数值稳定性问题")

def check_group_loader_consistency():
    """检查 group loader 的一致性"""
    print("\n5️⃣ Group Loader 一致性检查")
    print("-"*50)
    
    print("📁 Group 定义检查:")
    print("- waterbird_water: 水鸟主体 + 水背景")
    print("- waterbird_land: 水鸟主体 + 陆地背景")  
    print("- landbird_land: 陆鸟主体 + 陆地背景")
    print("- landbird_water: 陆鸟主体 + 水背景")
    
    print("\n🔍 潜在问题:")
    print("1. 类别映射: 0=水鸟, 1=陆鸟")
    print("2. 背景映射: 0=水背景, 1=陆地背景")
    print("3. 是否存在索引错误")
    print("4. 是否存在样本数量不一致")

def create_unit_test():
    """创建一致性测试"""
    print("\n6️⃣ 一致性单元测试")
    print("-"*50)
    
    test_code = '''
def test_background_gap_consistency(model, group_loaders, device='cpu'):
    """
    测试 Background Gap 与 Signed Background Effect 的一致性
    """
    # 调用两个函数
    bg_gap = compute_background_gap_real(model, group_loaders, device)
    delta_wb, delta_lb = compute_signed_background_effect(model, group_loaders, device)
    
    # 手工计算
    expected_bg_gap = (abs(delta_wb) + abs(delta_lb)) / 2
    
    # 验证一致性
    error = abs(bg_gap - expected_bg_gap)
    tolerance = 1e-5
    
    assert error < tolerance, f"Inconsistency detected: error={error}, tolerance={tolerance}"
    
    print(f"✅ 一致性验证通过: bg_gap={bg_gap:.6f}, expected={expected_bg_gap:.6f}")
    
    return True
'''
    
    print("建议的测试代码:")
    print(test_code)

def audit_conclusions():
    """审计结论"""
    print("\n7️⃣ 审计结论")
    print("-"*50)
    
    print("📊 当前状态:")
    print("- 旧版 Background Gap: ~0.005 (C组)")
    print("- 新版 Signed Effect: Δ_waterbird≈+0.108, Δ_landbird≈+0.192")
    print("- 理论 Background Gap: ~0.150")
    print("- 实际差异: 0.005 vs 0.150 (30倍差异)")
    
    print("\n🔍 最可能的原因:")
    print("1. 旧版函数可能存在 bug")
    print("2. 数据加载方式可能不同")
    print("3. 概率计算方式可能不同")
    print("4. 样本选择可能不同")
    
    print("\n🎯 建议的审计步骤:")
    print("1. 在相同的 checkpoint 和数据上运行两个函数")
    print("2. 打印四个底层概率值")
    print("3. 验证数学关系")
    print("4. 检查数据加载一致性")
    print("5. 添加一致性测试")

if __name__ == "__main__":
    analyze_code_differences()
    check_group_loader_consistency()
    create_unit_test()
    audit_conclusions()