#!/usr/bin/env python3
"""
uli_harness.py — 即插即用引导注入器(ULI) 快速迭代压测框架 (继续 第35+轮)

双层策略：
  - FAST  模式：T=25600, seeds=[42,7,99] 快速扫描多配置（缓存 baseline CI）
  - CONFIRM模式：对 fast 最优配置用大 T(T≥204800) 复核最终 avg

每配置 round 完成后立即写 JSON（防超时丢数据）。
满足约束：基座冻结；只注入可学习低秩权重 W_out（引导向量）；单一插件参数；
训练仅用锚定基座(standard)，零样本迁移到 3 族 —— 即插即用、非每方案重训。
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, GaitPhaseBlueprint
from plugins import BasePlugin
from learnable_bypass import ULI, run_sim

T_FAST = 25600
SEEDS_FAST = [42, 7, 99]
FAMS = ["standard", "transformer", "p2p"]
CACHE = "uli_baseline_cache.json"
OUT = "uli_harness.json"


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


# ---------------- baseline cache ----------------
def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def save_cache(c):
    json.dump(c, open(CACHE, "w"), indent=1)


def ci_base(family, T, seed):
    c = load_cache()
    key = f"{family}|{T}|{seed}"
    if key not in c:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        tb = run_sim(base, [], T, seed)
        c[key] = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], tb["dt"]),
                                   dt=tb["dt"])["coherence_index"]
        save_cache(c)
    return c[key]


def eval_plugin(family, plugin, T, seed):
    base = BasePolicy(n_joints=6, family=family, seed=seed)
    tp = run_sim(base, [plugin], T, seed)
    ci_p = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], tp["dt"]),
                            dt=tp["dt"])["coherence_index"]
    return opt(ci_p, ci_base(family, T, seed))


# ---------------- train & evaluate one config ----------------
def train_weval(cfg, T=T_FAST, seeds=SEEDS_FAST, anchor="standard", anchor_seed=42):
    plug = ULI(strength=cfg.get("strength", 1.0), W=cfg.get("W", 40),
               lam=cfg.get("lam", 5e-3), HID=cfg.get("HID", 48),
               j_lo=cfg.get("j_lo", 0.03), j_peak=cfg.get("j_peak", 0.2),
               j_hi=cfg.get("j_hi", 0.8))
    # anchor: 若提供共享语料则复用，否则现采
    if cfg.get("data"):
        d = cfg["data"]
    else:
        anchor_base = BasePolicy(n_joints=6, family=anchor, seed=anchor_seed)
        d = plug.collect_trajectory(anchor_base, seed=anchor_seed,
                                    t_len=cfg.get("t_train", 60000))
    plug.fit_from(d)
    per = {fam: [] for fam in FAMS}
    for fam in FAMS:
        for s in seeds:
            plug.reset()
            per[fam].append(eval_plugin(fam, plug, T, s))
        plug.reset()
    avgs = [(per["standard"][i] + per["transformer"][i]) / 2 for i in range(len(seeds))]
    uni = all(float(x) > 0 for fam in FAMS for x in per[fam])
    return {
        "name": cfg["name"], "T": T, "seeds": seeds, "anchor": anchor,
        "avg": float(np.mean(avgs)), "avg_std": float(np.std(avgs)),
        "families": {fam: float(np.mean(v)) for fam, v in per.items()},
        "universal": uni,
    }


def main():
    all_r = json.load(open(OUT)) if os.path.exists(OUT) else {}
    # 共享锚定数据集（一次采集，供多配置复用，减少重复仿真）
    print("采集锚定语料 (standard, seed=42, 120000) ...", flush=True)
    anchor_base = BasePolicy(n_joints=6, family="standard", seed=42)
    probe = ULI()
    shared = probe.collect_trajectory(anchor_base, seed=42, t_len=120000)
    print("完成。", flush=True)

    configs = [
        {"name": "uli_r1_lam3e3_H48_W40", "lam": 3e-3, "HID": 48, "W": 40},
        {"name": "uli_r1_lam1e2_H48_W40", "lam": 1e-2, "HID": 48, "W": 40},
        {"name": "uli_r1_lam5e3_H24_W40", "lam": 5e-3, "HID": 24, "W": 40},
        {"name": "uli_r1_lam5e3_H96_W40", "lam": 5e-3, "HID": 96, "W": 40},
        {"name": "uli_r1_lam5e3_H48_W20", "lam": 5e-3, "HID": 48, "W": 20},
        {"name": "uli_r1_strength0.6", "strength": 0.6},
        {"name": "uli_r1_strength1.4", "strength": 1.4},
        {"name": "uli_r1_lam5e3_H48_W40", "lam": 5e-3, "HID": 48, "W": 40},
    ]
    # 提供共享语料，避免每个配置重复采集
    for cfg in configs:
        cfg["data"] = shared

    best = None
    for i, cfg in enumerate(configs):
        if cfg["name"] in all_r:
            continue
        t0 = time.time()
        r = train_weval(cfg)
        r["time_s"] = round(time.time() - t0)
        all_r[f"R{i}_{cfg['name']}"] = r
        json.dump(all_r, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
        if best is None or r["avg"] > best["avg"]:
            best = r
        print(f"  R{i} {cfg['name']:26s} avg={r['avg']:+.4f} std=({r['families']['standard']:+.3f}) "
              f"trans=({r['families']['transformer']:+.3f}) p2p=({r['families']['p2p']:+.3f}) "
              f"uni={'Y' if r['universal'] else 'N'}  {r['time_s']}s", flush=True)
    print("\n  [round1 best]", best["name"], f"avg={best['avg']:+.4f}", flush=True)
    print(f"结果 -> {OUT}")


if __name__ == "__main__":
    main()