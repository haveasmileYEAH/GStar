# scripts/run_gstar.py
"""
运行GStar增强版CDMem的脚本
"""
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "CDMem"))  # 添加CDMem路径

# 导入GStar集成模块
from GStar.integration.cdmem_hook import gstar_enhance, get_gstar, enable_gstar


def integrate_cdmem():
    """集成CDMem和GStar"""

    # 方法1: 猴子补丁（Monkey Patching）
    # 如果CDMem有明确的选择函数，可以直接替换
    try:
        # 假设CDMem的结构
        from CDMem.main import CDMemFramework  # 根据实际CDMem结构调整

        # 保存原始方法
        original_select = CDMemFramework.select_memories if hasattr(CDMemFramework, 'select_memories') else None

        if original_select:
            @gstar_enhance("selection")
            def enhanced_select_memories(self, candidates, context=None, **kwargs):
                return original_select(self, candidates, context, **kwargs)

            # 替换方法
            CDMemFramework.select_memories = enhanced_select_memories
            print("[Integration] 成功集成GStar到CDMem.select_memories")
        else:
            print("[Integration] 未找到CDMem.select_memories方法，请手动添加装饰器")

    except ImportError as e:
        print(f"[Integration] CDMem导入失败: {e}")
        return False

    return True


def manual_integration_guide():
    """手动集成指南"""
    guide = """
    # 手动集成GStar到CDMem

    在CDMem的主要选择函数上添加装饰器:

    ```python
    # 1. 在CDMem文件顶部添加导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from GStar.integration.cdmem_hook import gstar_enhance

    # 2. 在选择函数上添加装饰器
    @gstar_enhance("selection")
    def your_selection_function(self, candidates, context=None):
        # 原有CDMem逻辑完全不变
        return original_cdmem_logic(candidates, context)
    ```

    常见的CDMem选择函数名:
    - select_memories()
    - retrieve_memories() 
    - get_candidates()
    - memory_selection()
    """
    print(guide)


def run_cdmem_with_gstar():
    """运行GStar增强版CDMem"""
    print("=== 启动GStar增强版CDMem ===")

    # 1. 启用GStar
    config_path = project_root / "configs" / "gstar_config.yaml"
    enable_gstar(str(config_path))

    # 2. 尝试自动集成
    if integrate_cdmem():
        print("[Success] GStar集成成功")
    else:
        print("[Manual] 需要手动集成，请参考以下指南:")
        manual_integration_guide()
        return

    # 3. 运行CDMem主程序
    try:
        # 这里根据CDMem的实际启动方式调整
        from CDMem.main import main as cdmem_main  # 根据实际调整

        print("[CDMem] 启动CDMem主程序...")
        cdmem_main()

    except ImportError:
        print("[CDMem] 请根据CDMem的实际结构调整启动代码")
        print("或者直接运行CDMem，GStar会自动生效")

    # 4. 打印统计信息
    gstar = get_gstar()
    print("\n=== GStar运行统计 ===")
    stats = gstar.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_cdmem_with_gstar()