# GStar/integration/cdmem_hook.py
"""
CDMem集成钩子 - 无侵入式集成GStar功能
"""
import os
import sys
import functools
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

# 添加GStar到Python路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from .config import GStarConfig
from ..selector.value_aware_astar import ValueAwareAStarSelector, Candidate, Budget
from ..selector.adapter_cdmem import cdmem_to_candidate
from ..graphs.reuse_index import ReuseIndex
from ..graphs.perm_store import PermStore


class GStar:
    """GStar主控制器"""

    def __init__(self, config_path: str = None):
        self.config = GStarConfig.load(config_path)
        self.selector = None
        self.reuse_index = None
        self.perm_store = None
        self.stats = {
            "total_calls": 0,
            "enhanced_calls": 0,
            "avg_improvement": 0.0
        }
        self._init_components()

    def _init_components(self):
        """初始化GStar组件"""
        if not self.config.enabled:
            return

        try:
            # 初始化选择器
            self.selector = ValueAwareAStarSelector(
                top_k=self.config.selector.top_k,
                lambda_rel=self.config.selector.lambda_rel,
                lambda_cov=self.config.selector.lambda_cov,
                lambda_red=self.config.selector.lambda_red,
                lambda_cost=self.config.selector.lambda_cost
            )

            # 初始化永久存储
            if self.config.storage.enabled:
                self.perm_store = PermStore(self.config.storage.path)

                # 初始化复用索引
                all_nodes = self.perm_store.load_all()
                self.reuse_index = ReuseIndex(
                    all_nodes,
                    min_cov=self.config.reuse_index.min_cov,
                    max_age_days=self.config.reuse_index.max_age_days
                )

            print(
                f"[GStar] 组件初始化完成 - 选择器: ✓, 存储: {self.perm_store is not None}, 复用索引: {self.reuse_index is not None}")

        except Exception as e:
            print(f"[GStar] 组件初始化失败: {e}")
            self.config.enabled = False

    def enhance_selection(self, candidates: List[Dict], context: Dict = None) -> List[Dict]:
        """增强候选选择"""
        self.stats["total_calls"] += 1

        if not self.config.enabled or not self.selector:
            return candidates

        try:
            # 1. 格式转换
            gstar_candidates = [cdmem_to_candidate(c) for c in candidates if c]
            if not gstar_candidates:
                return candidates

            # 2. 复用索引查询（如果启用）
            if self.reuse_index and context:
                query_cov = set(context.get('query', '').lower().split())
                admissible = [c.text for c in gstar_candidates]
                reuse_hits = self.reuse_index.find_superset_actions(query_cov, admissible)

                # 将复用结果转换为Candidate格式
                for hit in reuse_hits:
                    reuse_candidate = Candidate(
                        id=f"reuse_{hash(hit['text'])}",
                        text=hit['text'],
                        rel=hit['rel'],
                        cov=hit['cov'],
                        cost_est=hit.get('cost_est', 10.0),
                        meta={'source': 'reuse_index', 'ts': hit.get('ts')}
                    )
                    gstar_candidates.append(reuse_candidate)

            # 3. GStar选择
            budget = Budget(
                max_steps=self.config.budget.max_steps,
                max_tokens=self.config.budget.max_tokens,
                wall_clock_s=self.config.budget.wall_clock_s
            )

            selected, logs, early_stop = self.selector.select(gstar_candidates, budget)

            # 4. 格式转换回CDMem
            result = []
            for candidate in selected:
                cdmem_item = {
                    'id': candidate.id,
                    'text': candidate.text,
                    'action': candidate.text,  # CDMem可能需要action字段
                    'score': candidate.rel,
                    'rel': candidate.rel,
                    'cost_est': candidate.cost_est,
                    'cov': list(candidate.cov),
                    'meta': candidate.meta
                }
                result.append(cdmem_item)

            # 5. 保存到永久存储
            if self.perm_store and result:
                self.perm_store.upsert([{
                    'text': item['text'],
                    'rel': item['rel'],
                    'cov': item['cov'],
                    'context': context or {}
                } for item in result])

            # 6. 统计
            self.stats["enhanced_calls"] += 1
            improvement = len(result) / max(1, len(candidates))
            self.stats["avg_improvement"] = (
                                                    self.stats["avg_improvement"] * (
                                                        self.stats["enhanced_calls"] - 1) + improvement
                                            ) / self.stats["enhanced_calls"]

            print(f"[GStar] 选择增强: {len(candidates)} -> {len(result)} (改善率: {improvement:.2f})")
            if logs.get('best_f'):
                print(f"[GStar] 最佳分数: {logs['best_f']:.3f}, 早停: {early_stop}")

            return result

        except Exception as e:
            print(f"[GStar] 增强选择失败，回退到原版: {e}")
            import traceback
            traceback.print_exc()
            return candidates

    def get_stats(self) -> Dict:
        """获取运行统计"""
        return {
            **self.stats,
            "enhancement_rate": self.stats["enhanced_calls"] / max(1, self.stats["total_calls"]),
            "config_enabled": self.config.enabled,
            "components": {
                "selector": self.selector is not None,
                "reuse_index": self.reuse_index is not None,
                "perm_store": self.perm_store is not None
            }
        }


# 全局GStar实例
_gstar = None


def get_gstar(config_path: str = None) -> GStar:
    """获取全局GStar实例"""
    global _gstar
    if _gstar is None:
        _gstar = GStar(config_path)
    return _gstar


def gstar_enhance(hook_name: str = "selection"):
    """
    GStar增强装饰器

    用法:
    @gstar_enhance("selection")
    def cdmem_select_function(candidates, context=None):
        return original_cdmem_logic(candidates)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 执行原始CDMem逻辑
            result = func(*args, **kwargs)

            # 如果结果是候选列表，尝试GStar增强
            if (hook_name == "selection" and
                    isinstance(result, list) and
                    len(result) > 0 and
                    isinstance(result[0], dict)):
                gstar = get_gstar()
                context = kwargs.get('context') or (args[1] if len(args) > 1 and isinstance(args[1], dict) else {})
                enhanced_result = gstar.enhance_selection(result, context)
                return enhanced_result

            return result

        return wrapper

    return decorator


# 便捷函数
def enable_gstar(config_path: str = None):
    """启用GStar增强"""
    gstar = get_gstar(config_path)
    gstar.config.enabled = True
    gstar._init_components()
    print("[GStar] 已启用GStar增强功能")


def disable_gstar():
    """禁用GStar增强"""
    gstar = get_gstar()
    gstar.config.enabled = False
    print("[GStar] 已禁用GStar增强功能")


def gstar_stats():
    """打印GStar统计信息"""
    gstar = get_gstar()
    stats = gstar.get_stats()
    print("\n=== GStar运行统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")