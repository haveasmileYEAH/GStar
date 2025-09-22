# GStar/integration/config.py
"""
GStar配置管理
"""
import yaml
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class SelectorConfig:
    top_k: int = 8
    lambda_rel: float = 1.0
    lambda_cov: float = 0.5
    lambda_red: float = 0.1
    lambda_cost: float = 0.1


@dataclass
class BudgetConfig:
    max_steps: Optional[int] = 50
    max_tokens: Optional[int] = 4000
    wall_clock_s: Optional[float] = None


@dataclass
class StorageConfig:
    enabled: bool = True
    path: str = "runs/gstar_perm.jsonl"


@dataclass
class ReuseIndexConfig:
    min_cov: int = 3
    max_age_days: Optional[int] = 30


@dataclass
class GStarConfig:
    enabled: bool = False
    selector: SelectorConfig = None
    budget: BudgetConfig = None
    storage: StorageConfig = None
    reuse_index: ReuseIndexConfig = None

    def __post_init__(self):
        if self.selector is None:
            self.selector = SelectorConfig()
        if self.budget is None:
            self.budget = BudgetConfig()
        if self.storage is None:
            self.storage = StorageConfig()
        if self.reuse_index is None:
            self.reuse_index = ReuseIndexConfig()

    @classmethod
    def load(cls, config_path: str = None) -> 'GStarConfig':
        """从配置文件加载GStar配置"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "gstar_config.yaml"

        config_path = Path(config_path)

        if not config_path.exists():
            print(f"[GStar] 配置文件不存在，创建默认配置: {config_path}")
            default_config = cls()
            default_config.save(config_path)
            return default_config

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            return cls._from_dict(data)
        except Exception as e:
            print(f"[GStar] 配置文件加载失败: {e}, 使用默认配置")
            return cls()

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'GStarConfig':
        """从字典创建配置对象"""
        config = cls()
        config.enabled = data.get('enabled', False)

        if 'selector' in data:
            config.selector = SelectorConfig(**data['selector'])

        if 'budget' in data:
            config.budget = BudgetConfig(**data['budget'])

        if 'storage' in data:
            config.storage = StorageConfig(**data['storage'])

        if 'reuse_index' in data:
            config.reuse_index = ReuseIndexConfig(**data['reuse_index'])

        return config

    def save(self, config_path: Path):
        """保存配置到文件"""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'enabled': self.enabled,
            'selector': asdict(self.selector),
            'budget': asdict(self.budget),
            'storage': asdict(self.storage),
            'reuse_index': asdict(self.reuse_index)
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[GStar] 配置已保存到: {config_path}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'enabled': self.enabled,
            'selector': asdict(self.selector),
            'budget': asdict(self.budget),
            'storage': asdict(self.storage),
            'reuse_index': asdict(self.reuse_index)
        }