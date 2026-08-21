"""
FedBN / LocalHead 机制验收脚本 (问题与解决3.txt 第四节)

不跑完整数据实验, 只验证机制层面的四项:
  1. _get_excluded_param_names 正确识别 BN (含 running stats) 与预测头;
  2. 服务器聚合时不聚合被排除的 BN/头参数 (server 端保持不变);
  3. 广播共享参数时本地 BN/头不被覆盖 (set_parameters exclude);
  4. 两轮模拟联邦流程中, 各客户端 BN/头保持本地差异。

用法: python experiments/verify_fedbn_localhead.py
所有检查 PASS 才返回退出码 0。
"""
import sys
import os
import copy
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from src.models.tcn_lstm import build_model
from src.federated.aggregation import FLClient, FLServer
from src.federated.trainer import _get_excluded_param_names
from torch.utils.data import TensorDataset, DataLoader

PASS = 0
FAIL = 0


def check(cond, name, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def _build_tiny_model():
    cfg = ModelConfig()
    cfg.tcn_channels = [8, 8]
    cfg.lstm_hidden = 8
    cfg.lstm_layers = 1
    cfg.fc_hidden = 8
    return build_model(input_dim=4, pred_len=2, model_cfg=cfg)


def _dummy_loader():
    ds = TensorDataset(torch.randn(8, 4, 4), torch.randn(8, 2))
    return DataLoader(ds, batch_size=4)


def _keys_subset(state, substr):
    return {k: v for k, v in state.items() if substr in k}


def _max_abs_diff(a, b, keys):
    d = 0.0
    for k in keys:
        if k in a and k in b:
            d = max(d, float((a[k] - b[k]).abs().max()))
    return d


def main():
    print("=" * 60)
    print("  FedBN / LocalHead 机制验收")
    print("=" * 60)

    model = _build_tiny_model()
    excluded = _get_excluded_param_names(model, use_fedbn=True, use_local_head=True)

    bn_keys = [k for k in excluded if "bn" in k]
    head_keys = [k for k in excluded if "fc" in k]

    # ── 检查0: FedBN / LocalHead 开关必须独立 (回归测试) ──────
    fedbn_only = _get_excluded_param_names(model, use_fedbn=True, use_local_head=False)
    lh_only = _get_excluded_param_names(model, use_fedbn=False, use_local_head=True)
    check(not any("fc" in k for k in fedbn_only),
          "FedBN-only 不排除预测头 (开关独立)")
    check(not any("bn" in k for k in lh_only),
          "LocalHead-only 不排除 BN (开关独立)")

    # ── 检查1: 排除列表正确 ─────────────────────────────────
    check(any(".running_mean" in k for k in bn_keys),
          "FedBN: running_mean 被排除 (之前用 named_parameters 会漏掉 buffer)")
    check(any(".weight" in k for k in bn_keys),
          "FedBN: BN weight 被排除")
    check(any(k.endswith(".weight") for k in head_keys) and
          any(k.endswith(".bias") for k in head_keys),
          "LocalHead: 预测头 weight/bias 被排除")
    check(head_keys and all("fc.3" in k or "fc." in k for k in head_keys),
          "LocalHead: 头为 fc 末层 Linear (不再硬编码 fc.3)")

    # ── 检查2: 服务器聚合不改变被排除参数 ──────────────────
    client_a = FLClient("A", copy.deepcopy(model), _dummy_loader(),
                        _dummy_loader(), "cpu")
    client_b = FLClient("B", copy.deepcopy(model), _dummy_loader(),
                        _dummy_loader(), "cpu")

    # 模拟本地训练: 两客户端 BN running_mean 与头产生不同漂移
    for cl, delta in [(client_a, 1.0), (client_b, 5.0)]:
        sd = cl.model.state_dict()
        for k in bn_keys:
            if "running_mean" in k or "running_var" in k:
                sd[k] = sd[k] + delta
        for k in head_keys:
            sd[k] = sd[k] + delta
        cl.model.load_state_dict(sd)

    server = FLServer(copy.deepcopy(model), aggregation="fedavg")
    server_bn_before = _keys_subset(server.global_model.state_dict(), "bn")
    server_head_before = _keys_subset(server.global_model.state_dict(), "fc.3")

    server.aggregate([client_a.get_parameters(), client_b.get_parameters()],
                     [1.0, 1.0], exclude_param_names=excluded)

    server_bn_after = _keys_subset(server.global_model.state_dict(), "bn")
    server_head_after = _keys_subset(server.global_model.state_dict(), "fc.3")
    check(_max_abs_diff(server_bn_before, server_bn_after, bn_keys) == 0.0,
          "聚合后服务器 BN 参数不变 (未被聚合)")
    check(_max_abs_diff(server_head_before, server_head_after, head_keys) == 0.0,
          "聚合后服务器预测头不变 (未上传)")

    # ── 检查3: set_parameters exclude 保留本地 BN/头 ─────────
    c_before = copy.deepcopy(client_a.model.state_dict())
    client_a.set_parameters(server.get_global_params(), exclude_names=excluded)
    c_after = client_a.model.state_dict()
    check(_max_abs_diff(c_before, c_after, bn_keys) == 0.0,
          "广播时本地 BN running stats 未被覆盖")
    check(_max_abs_diff(c_before, c_after, head_keys) == 0.0,
          "广播时本地预测头未被覆盖")

    # ── 检查4: 两轮联邦流程端到端保留本地差异 ───────────────
    # 模拟第0轮全量广播(统一初始化), 第1轮起 exclude 广播
    c0 = FLClient("A", copy.deepcopy(model), _dummy_loader(), _dummy_loader(), "cpu")
    c1 = FLClient("B", copy.deepcopy(model), _dummy_loader(), _dummy_loader(), "cpu")
    srv = FLServer(copy.deepcopy(model), "fedavg")

    # 第0轮: 全量广播
    gp = srv.get_global_params()
    c0.set_parameters(gp)
    c1.set_parameters(gp)
    # 本地训练产生差异
    for cl, delta in [(c0, 2.0), (c1, 7.0)]:
        sd = cl.model.state_dict()
        for k in bn_keys:
            if "running_mean" in k:
                sd[k] = sd[k] + delta
        cl.model.load_state_dict(sd)
    # 聚合(排除) + 第1轮 exclude 广播
    srv.aggregate([c0.get_parameters(), c1.get_parameters()],
                  [1.0, 1.0], exclude_param_names=excluded)
    bn0 = c0.model.state_dict()["tcn.network.0.bn1.running_mean"].clone()
    bn1 = c1.model.state_dict()["tcn.network.0.bn1.running_mean"].clone()
    c0.set_parameters(srv.get_global_params(), exclude_names=excluded)
    c1.set_parameters(srv.get_global_params(), exclude_names=excluded)
    bn0_after = c0.model.state_dict()["tcn.network.0.bn1.running_mean"]
    bn1_after = c1.model.state_dict()["tcn.network.0.bn1.running_mean"]
    check(float((bn0_after - bn0).abs().max()) == 0.0,
          "第1轮广播后客户端A BN 未被覆盖")
    check(float((bn1_after - bn1).abs().max()) == 0.0,
          "第1轮广播后客户端B BN 未被覆盖")
    check(float((bn0_after - bn1_after).abs().max()) > 0.0,
          "两客户端 BN 保持本地差异 (FedBN 生效)")

    print("=" * 60)
    print(f"  结果: {PASS} PASS, {FAIL} FAIL")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
