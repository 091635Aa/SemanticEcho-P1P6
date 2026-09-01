#!/usr/bin/env python3
"""v12m.py — 校验"非谐波残差比"能否在跨seed上分开关节的去抖需求。
原理: clean/trans 的残差是全谐波(15Hz纹波=1Hz的整数倍) → 非谐波残差≈0 → 无需去抖;
      standard 含 0.3Hz漂移(非1Hz整数倍)+噪声 → 非谐波残差>0 → 需去抖;
      mild/strong-trans 被噪声破相 → 非谐波残差增大 → 需去抖。
若 clean/trans 稳定≈0 而其它(candidat)稳定>0, 则可作鲁棒门。"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy, LeggedMicroSim
import cdiag

T = 6000
DT = cdiag.DT
SEEDS = [99, 7, 42]
KH = 10  # 谐波阶数上限


def nh_ratio(base, noise, seed):
    s = LeggedMicroSim(base, plugins=[], T=T, dt=DT, seed=seed, **noise)
    tr = s.run()
    a = tr["a"]; t = tr["t"]; n = a.shape[1]
    phi = 2 * np.pi * t[:, None]                        # (T,1) 全局相位
    cols = [np.ones((T, 1))]
    for k in range(1, KH + 1):
        cols.append(np.sin(k * phi)); cols.append(np.cos(k * phi))
    B = np.hstack(cols)                                  # (T, 1+2K)
    c = np.linalg.pinv(B) @ a                            # (1+2K, n)
    fit = B @ c
    res = a - fit
    vr = res.var(axis=0); vt = a.var(axis=0) + 1e-9
    return float(np.mean(vr / vt)), float(np.mean((vt - vr) / vt))  # 非谐波比 / 谐波解释比


def main():
    print("非谐波残差比(mean)(谐波解释比) 跨seed:", flush=True)
    out = {}
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        out[scen] = {}
        for fam in ["standard", "transformer", "p2p"]:
            rows = []
            for seed in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=seed)
                nh, expl = nh_ratio(base, noise, seed)
                rows.append(nh)
            m = float(np.mean(rows))
            out[scen][fam] = m
            print(f"  [{scen}/{fam:11s}] 非谐波比={m:.4f}  (min={min(rows):.3f} max={max(rows):.3f})", flush=True)
    json.dump(out, open("v12m_nh.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()