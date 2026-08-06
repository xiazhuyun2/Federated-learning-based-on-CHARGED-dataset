"""
实验追踪模块 — 唯一输出目录、完整配置保存、可复现性保证

每次运行自动生成独立目录:
  outputs/{city}/{method}/seed_{seed}/{timestamp}_{git_commit[:7]}/
  ├── config.json       # 完整配置 (含命令行参数和默认值)
  ├── history.json      # 训练损失和验证指标
  ├── metrics.json      # 最终测试指标
  ├── best_model.pt     # 验证集最优 checkpoint
  └── predictions.npz   # 所有客户端的预测值和真实值
"""
import os
import json
import subprocess
import dataclasses
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import torch


def get_git_commit(project_root: str = None) -> str:
    """获取当前 Git commit 短哈希, 用于可复现性追踪"""
    try:
        cwd = project_root or os.getcwd()
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL
        )
        return result.decode().strip()
    except Exception:
        return "nogit"


class ExperimentTracker:
    """
    实验追踪器: 管理输出目录、保存配置/历史/指标/模型/预测值
    """

    def __init__(self, base_dir: str, city: str, method: str, seed: int,
                 project_root: str = None):
        self.base_dir = base_dir
        self.city = city
        self.method = method
        self.seed = seed
        self.project_root = project_root or os.getcwd()

        # 生成唯一运行目录
        git_commit = get_git_commit(self.project_root)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{git_commit}"
        self.run_dir = os.path.join(
            base_dir, city, method, f"seed_{seed}", self.run_id
        )
        os.makedirs(self.run_dir, exist_ok=True)

    def save_config(self, config, cli_args: Optional[dict] = None):
        """保存完整配置 (dataclass -> JSON)"""
        config_dict = _dataclass_to_dict(config)
        if cli_args:
            config_dict["_cli_args"] = cli_args
        config_dict["_git_commit"] = get_git_commit(self.project_root)
        config_dict["_timestamp"] = datetime.now().isoformat()

        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        return path

    def save_history(self, history: Dict):
        """保存训练历史 (loss, val_metrics)"""
        path = os.path.join(self.run_dir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        return path

    def save_metrics(self, test_results: Dict):
        """保存最终测试指标"""
        path = os.path.join(self.run_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, default=str)
        return path

    def save_best_model(self, model: torch.nn.Module, val_rmse: float):
        """保存验证集最优模型 checkpoint"""
        path = os.path.join(self.run_dir, "best_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "val_rmse": val_rmse,
            "run_id": self.run_id,
        }, path)
        return path

    def save_predictions(self, predictions: Dict[str, Dict]):
        """
        保存所有客户端预测值和真实值
        predictions: {client_id: {"pred": np.ndarray, "target": np.ndarray}}
        """
        path = os.path.join(self.run_dir, "predictions.npz")
        npz_dict = {}
        for client_id, data in predictions.items():
            npz_dict[f"{client_id}_pred"] = np.asarray(data["pred"])
            npz_dict[f"{client_id}_target"] = np.asarray(data["target"])
        np.savez_compressed(path, **npz_dict)
        return path

    @staticmethod
    def find_runs(base_dir: str, city: str = None, method: str = None,
                  seed: int = None) -> list:
        """扫描 outputs/ 目录, 返回所有匹配的运行目录路径"""
        import glob
        pattern = os.path.join(base_dir, city or "*", method or "*",
                               f"seed_{seed}" if seed is not None else "seed_*",
                               "run_*")
        return sorted(glob.glob(pattern))


def _dataclass_to_dict(obj) -> dict:
    """递归将 dataclass 转换为可 JSON 序列化的字典"""
    if dataclasses.is_dataclass(obj):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
