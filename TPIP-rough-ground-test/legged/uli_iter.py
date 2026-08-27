#!/usr/bin/env python3
"""
uli_iter.py — 可学引导注入器迭代 (第36+轮)：可学权重 ≠ 替代底座，而是做底座之上的自适应精修

方向修正（源自 learnable_bypass 结论：裸 ULI 直迁失败 trans=-0.58）：
  - 保留 BestCombo 的 jerk 门自适应平滑底座（跨族稳健）；
  - 在其上叠加可学习引导向量 ULI，做小强度 multipart 修正；
  - 用"结合后是否 ≥ 底座 && Universal 更稳"做增量判断（带可学权重且不劣化 = 合格；
    avg 更高 = 更优）。

FAST 环：T=25600, seeds=[42,7,99]，缓存 baseline CI；每配置完成即写 JSON。
"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy
from learnable_bypass import ULI, run_sim
from verify_architecture import BestCombo
from plugins import BasePlugin

T_FAST = 25600
SEEDS = [42, 7, 99]
FAMS = ["standard", "transformer", "p2p"]
CACHE = "uli_iter_base_cache.json"
OUT = "uli_iter.json"


def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)


def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def ci_base(family, T, seed):
    c = load_cache()
    k = f"{family}|{T}|{seed}"
    if k not in c:
        tb = run_sim(BasePolicy(n_joints=6, family=family, seed=seed), [], T, seed)
        c[k] = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], tb["dt"]),
                                 dt=tb["dt"])["coherence_index"]
        json.dump(c, open(CACHE, "w"), indent=1)
    return c[k]


def est(fam, T, seed, factory):
    """factory() -> 新插件实例（带已训练权重）。"""
    p = factory()
    if hasattr(p, "reset"):
        p.reset()
    tp = run_sim(BasePolicy(n_joints=6, family=fam, seed=seed), [p], T, seed)
    cpp = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], tp["dt"]),
                            dt=tp["dt"])["coherence_index"]
    return opt(cpp, ci_base(fam, T, seed))


class Chain(BasePlugin):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def inject(self, a, **kw):
        aa = self.a.inject(a, **kw)
        return self.b.inject(aa, **kw)


def train_uli(lam=5e-3, W=40, tlen=80000):
    plug = ULI(strength=1.0, W=W, lam=lam)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42),
                                   seed=42, t_len=tlen)
    plug.fit_from(data)
    return plug


_CLONES = {}


def build_bare(plug):
    def f():
        return copy.deepcopy(plug)
    return f


def build_chain(plug, base_order="b_u", strength=0.5):
    def f():
        base = BestCombo()
        u = copy.deepcopy(plug)
        u.strength = strength
        u._amp = min(u._amp, strength)
        return Chain(base, u) if base_order == "b_u" else Chain(u, base)
    return f


def run_configs(configs, plug, tlen_time):
    all_r = json.load(open(OUT)) if os.path.exists(OUT) else {}
    best = None
    for i, cfg in enumerate(configs):
        key = f"R{i}_{cfg['name']}"
        if key in all_r:
            continue
        t0 = time.time()
        fac = cfg["factory"]
        per = {fam: [est(fam, T_FAST, s, fac) for s in SEEDS] for fam in FAMS}
        avgs = [(per["standard"][i] + per["transformer"][i]) / 2 for i in range(len(SEEDS))]
        univ = all(float(x) > 0 for fam in FAMS for x in per[fam])
        r = {"name": cfg["name"], "T": T_FAST, "avg": float(np.mean(avgs)),
             "avg_std": float(np.std(avgs)),
             "families": {fam: float(np.mean(v)) for fam, v in per.items()},
             "universal": univ, "time_s": round(time.time() - t0) + round(tlen_time)}
        all_r[key] = r
        json.dump(all_r, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
        if best is None or r["avg"] > best["avg"]:
            best = r
        print(f"  R{i} {cfg['name']:34s} avg={r['avg']:+.4f} "
              f"s=({r['families']['standard']:+.3f}) t=({r['families']['transformer']:+.3f}) "
              f"p=({r['families']['p2p']:+.3f}) uni={'Y' if univ else 'N'}  {r['time_s']}s", flush=True)
    return best


def main():
    print("训练锚定可学 ULI ...", flush=True)
    plug = train_uli()
    print("  W_out", plug.W_out.shape, "amp", round(plug._amp, 3), flush=True)

    configs = [
        {"name": "ref_BestCombo",                "factory": (lambda: BestCombo())},
        {"name": "uli_bare",
         "factory": build_bare(plug)},
        {"name": "BestCombo_uli_s0.5",
         "factory": build_chain(plug, "b_u", 0.5)},
        {"name": "BestCombo_uli_s0.25",
         "factory": build_chain(plug, "b_u", 0.25)},
    ]
    best = run_configs(configs, plug, 8.0)
    print("\n  [best]", best["name"], f"avg={best['avg']:+.4f} uni={best['universal']}", flush=True)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()