#!/usr/bin/env python3
"""
cdiag.py — 一致性(C_cons) / 联动性(L_link) / 参照性(R_ref) 三维诊断。

前几轮的 CI(连贯性指数) 只取"方差最大单关节通道"，衡量单通道平滑度与单通道
周而复始重合度。用户要求提升的 一致性/联动性/参照性 是多关节的协同属性，
不被单通道 CI 完全捕获 —— 这正是"正交增益"的空间：

  [1] C_cons 一致性   : 以步态蓝图频率(1.0Hz,已知)为锚，对全部 6 关节分别做
                        周而复始的相位散布，再求关节均值。>更高=每关节逐周期
                        一致，整体更连贯。
  [2] L_link 联动性   : 6 关节两两绝对相关的均值 |corr(q_i,q_j)|。所有关节
                        都锁定在共享步态相位上时 → 强(反)相关 → 高。
  [3] R_ref 参照性    : 逐时刻在 6 关节上算 q[t] 与蓝图动作 bp[t] 的 Pearson
                        相关，再取时间均值。越高=代理的关节姿态越贴合全局步态蓝图。

用法: python3 cdiag.py [T] [seed]
输出: 每族×每场景 baseline / champ(V5-g0) / v6g3s5 的三维值 + 相对增量。
"""
import sys, os, json, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, GaitPhaseBlueprint, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV5, RobustComboV6

T = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
FAMS = ["standard", "transformer", "p2p"]
FREQ = 1.0
DT = 0.01

SCEN = {
    "clean":  {"obs_noise":0.0,"act_noise":0.0,"obs_spike_p":0.0,"obs_spike_amp":0.0,"act_spike_p":0.0,"act_spike_amp":0.0},
    "mild":   {"obs_noise":0.02,"act_noise":0.01,"obs_spike_p":0.0,"obs_spike_amp":0.0,"act_spike_p":0.0,"act_spike_amp":0.0},
    "strong": {"obs_noise":0.05,"act_noise":0.02,"obs_spike_p":0.002,"obs_spike_amp":0.4,"act_spike_p":0.002,"act_spike_amp":0.4},
}


def c_cons(traj, n_bins=32, fe_scale=0.18):
    """一致性：蓝图锚定的全关节周而复始相位散布的反向测度。"""
    q = traj["q"]; t = traj["t"]; n = q.shape[1]
    phi = 2 * np.pi * FREQ * t                     # 已知步态相位
    cycle = np.floor(phi / (2 * np.pi)).astype(int)
    starts = np.where(np.diff(cycle, prepend=cycle[0]) != 0)[0]
    starts = np.concatenate([starts, [len(q)]])
    if len(starts) < 3:
        return 0.0
    ph = np.linspace(0.0, 1.0 - 1e-9, n_bins)
    cycle_std_sum = 0.0; cnt = 0
    for j in range(n):
        qj = q[:, j]; qn = (qj - qj.mean()) / (qj.std() + 1e-9)
        cyc = []
        for a, b in zip(starts[:-1], starts[1:]):
            if b - a < n_bins:
                continue
            seg = np.linspace(0.0, 1.0, b - a)
            cyc.append(np.interp(ph, seg, qn[a:b]))
        if len(cyc) < 2:
            continue
        C = np.stack(cyc, axis=0)
        gs = float(np.std(C)) + 1e-9
        disp = float(np.mean(np.linalg.norm(C - C.mean(axis=0), axis=-1)))
        cycle_std_sum += disp / gs; cnt += 1
    if cnt == 0:
        return 0.0
    D = cycle_std_sum / cnt
    return float(np.clip(1.0 / (1.0 + D / fe_scale), 0.0, 1.0))


def l_link(traj):
    """联动性：6 关节两两绝对相关的均值（越高越联动）。"""
    q = traj["q"]
    qc = q - q.mean(axis=0)
    s = qc.std(axis=0) + 1e-9
    qn = qc / s
    R = np.corrcoef(qn.T)
    n = q.shape[1]
    vals = [abs(R[i, j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else 0.0


def r_ref(traj):
    """参照性：逐时刻 q[t] 与蓝图 bp[t] 的关节间 Pearson 相关，再取时间均值。"""
    q = traj["q"]; t = traj["t"]; n = q.shape[1]
    phi = 2 * np.pi * FREQ * t
    offs = np.linspace(0, np.pi, n)
    bp = 0.25 * np.sin(phi[:, None] + offs[None, :])          # (T,n) 向量化蓝图
    bpc = bp - bp.mean(axis=1, keepdims=True)
    bp_s = bpc.std(axis=1, keepdims=True) + 1e-9
    bpn = bpc / bp_s
    qc = q - q.mean(axis=1, keepdims=True)
    q_s = qc.std(axis=1, keepdims=True) + 1e-9
    qn = qc / q_s
    r = np.mean(qn * bpn, axis=1)               # 逐时刻 6 关节相关
    return float(np.mean(r))


def run_metrics(base, seed, noise, make):
    s = LeggedMicroSim(base, T=T, dt=DT, seed=seed, **noise)
    tb = s.run()
    mb = {"C": c_cons(tb), "L": l_link(tb), "R": r_ref(tb),
          "CI": compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], DT), dt=DT)["coherence_index"]}
    p = make()
    if p is not None:
        if hasattr(p, "reset"): p.reset()
    ts = LeggedMicroSim(base, plugins=([p] if p is not None else []), T=T, dt=DT, seed=seed, **noise)
    tp = ts.run()
    mp = {"C": c_cons(tp), "L": l_link(tp), "R": r_ref(tp),
          "CI": compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], DT), dt=DT)["coherence_index"]}
    return mb, mp


def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42), seed=42, t_len=60000)
    plug.fit_from(data); plug.strength = 0.5; plug._amp = min(plug._amp, 0.5)
    return plug


def main():
    print(f"三维诊断 [T={T}, seed={SEED}]", flush=True)
    p0 = train_uli()
    cfgs = {
        "baseline": (lambda: None),
        "champ":    (lambda: RobustComboV5(BestCombo(), copy.deepcopy(p0), g_max=0.0)),
        "v6g3s5":   (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)),
    }
    out = {}
    for scen, noise in SCEN.items():
        out[scen] = {}
        for fam in FAMS:
            base = BasePolicy(n_joints=6, family=fam, seed=SEED)
            row = {}
            for name, make in cfgs.items():
                mb, mp = run_metrics(base, SEED, noise, make)
                row[name] = {"base": mb, "plug": mp}
            out[scen][fam] = row
            def lr(row, k, cfg): return row[cfg]["plug"][k] - row[cfg]["base"][k]
            line = f"[{scen}/{fam:11s}]"
            for cfg in ["champ", "v6g3s5"]:
                line += (f"  {cfg:6s} dC={lr(row,'C',cfg):+.3f} dL={lr(row,'L',cfg):+.3f} "
                         f"dR={lr(row,'R',cfg):+.3f} dCI={lr(row,'CI',cfg):+.3f}")
            print(line, flush=True)
    json.dump({"T": T, "seed": SEED, "results": out},
              open(f"cdiag_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()