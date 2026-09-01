#!/usr/bin/env python3
"""v12j.py — 聚焦 transformer 族: 扫"(blend?)×(lam)×(β cap)", 找到 clean 零损 且 mild/strong 大增益 的配置。
原理: clean/trans 基座已饱和(C0.599), 去抖目标只要稍微贴合其"自洽形状"即可不损;
      mild/strong 因噪声失相, 需更强的去抖。目标是让 clean 趋向恒等, 但轻量 λ/目标形变。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
cdiag.T = T


def mk(**kw):
    defaults = dict(alpha=0.05, blend=True, r2b_lo=0.45, r2b_hi=0.85, min_cyc=2)
    defaults.update(kw)
    return lambda: PhaseRecomb(**defaults)


def main():
    print(f"V12j transformer聚焦 [T={T}, seed={SEED}]", flush=True)
    cfgs = {
        "base":  (lambda: None),
        "l85-bl":     mk(lam=0.85),
        "l85-noB":    mk(lam=0.85, blend=False),
        "l40-bl":     mk(lam=0.40),
        "l60-bl":     mk(lam=0.60),
        "l85-bb0.3":  mk(lam=0.85, r2b_hi=9.0, blend_pow=4.0),   # β 被压低(几乎恒=模板)
        "l60-bb0.4":  mk(lam=0.60, r2b_lo=0.9, r2b_hi=0.98),     # 只有极高R²才偏BB
        "l85-bthr0.5":mk(lam=0.85, r2b_lo=0.97, r2b_hi=0.995),   # β 基本=模板(保谐波)强λ
    }
    order = ["l85-bl", "l85-noB", "l40-bl", "l60-bl", "l85-bb0.3", "l60-bb0.4", "l85-bthr0.5"]
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        base = BasePolicy(n_joints=6, family="transformer", seed=SEED)
        row = {}
        for name, make in cfgs.items():
            mb, mp = cdiag.run_metrics(base, SEED, noise, make)
            row[name] = {"base": mb, "plug": mp}
        line = f"[{scen}/transformer]"
        mb = row["base"]["base"]
        line += f" base C{mb['C']:.3f} R{mb['R']:.3f} CI{mb['CI']:.3f}"
        for cfg in order:
            p = row[cfg]["plug"]
            dC = p['C'] - mb['C']; dR = p['R'] - mb['R']; dCI = p['CI'] - mb['CI']
            line += f"  {cfg:11s} dC{dC:+.3f} dR{dR:+.3f} dCI{dCI:+.3f}"
        print(line, flush=True)
    json.dump({"T": T, "seed": SEED},
              open(f"v12j_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()