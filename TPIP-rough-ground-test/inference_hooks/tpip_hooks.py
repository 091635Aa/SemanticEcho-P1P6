# tpip_hooks.py
# ============================================================================
# TPIP（Three-Propagation Injection Protocol, 三元传播注入协议）推理期挂载钩子
#
# 目标：在【不修改/不重训】基座模型 π_θ 任何一层权重的前提下，仅改变推理时
#       的"初始状态偏置"，提升双足人形控制器动作的连贯性(Coherence)，消除
#       点对点(P2P)僵硬控制。
#
# 三条并行电路（推理期，前向只读基座）：
#   电路A 主线·上下文传播   : 接收历史状态(上一帧关节角/IMU)，保持物理连续性与惯性
#   电路B 反向·全局目标传播 : 从终点目标(如 3m 外落点)倒推，生成全局步态相位蓝图
#   电路C 中间·瞬时感知传播 : 接收当前外部刺激(深度点云/足底压力突变)
#
# 注入时机 P1.5：在主模型正式输出第一个动作/token 的【前一刻】，将电路 B、C 的
#   高维隐藏态压缩为紧凑向量，与电路 A 的初始隐状态做门控融合(Gated Fusion)，
#   融合结果作为主模型"初始残差(Initial Residual)"注入；此后完全回归自回归生成，
#   不再有任何外部干预。
#
# 铁律：插件网络(编码器/压缩器/门控)的权重可训练；基座 π_θ 权重被 stop_gradient
#       完全冻结，禁止梯度回传/重训/LoRA。仅训练插件侧参数。
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 基座模型最小接口(占位, 代表任一无权修改的现成动作策略 / transformer 主干)    #
# --------------------------------------------------------------------------- #
class BasePolicy(nn.Module):
    """只读基座。TPIP 绝不写入/训练其参数。"""
    def __init__(self, init_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.init_proj = nn.Linear(init_dim, hidden_dim)  # 初始状态->隐状态 h0
        self.generative = nn.GRUCell(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, act_dim)
        # 冻结：铁律2——不做任何梯度和重训
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, obs0):
        h = self.init_proj(obs0)          # 基座自己算出的初始隐状态 (即 h_A)
        return h


# --------------------------------------------------------------------------- #
# 电路B：全局目标传导（反向传播整条轨迹意图）                                   #
# --------------------------------------------------------------------------- #
class GlobalGoalCircuit(nn.Module):
    """把终点目标倒推成"全局步态相位蓝图"的压缩分支(仅此分支参数可训)。"""
    def __init__(self, goal_dim: int, hidden_dim: int, latent_dim: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.compress = nn.Linear(hidden_dim, latent_dim)  # -> z_B (紧凑向量)

    def forward(self, goal):
        return self.compress(self.enc(goal))


# --------------------------------------------------------------------------- #
# 电路C：瞬时感知传导（当前环境刺激）                                           #
# --------------------------------------------------------------------------- #
class PerceptionCircuit(nn.Module):
    """把当前外部刺激(点云/足压)压缩为瞬时残差向量(仅此分支参数可训)。"""
    def __init__(self, stim_dim: int, hidden_dim: int, latent_dim: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(stim_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.compress = nn.Linear(hidden_dim, latent_dim)  # -> z_C

    def forward(self, stim):
        return self.compress(self.enc(stim))


# --------------------------------------------------------------------------- #
# TPIP 挂载器：拼接三路、门控融合、在 P1.5 注入初始残差                          #
# --------------------------------------------------------------------------- #
class TPIPInjectionHook(nn.Module):
    """
    非侵入式挂载：
      - 不触碰 base 任何权重（hook 挂在外层 forward）。
      - 仅为推理期前向计算增加并行分支与一次性的初始残差注入。
      - 插件自身参数 trainable=True（用于离策略训插件）, base 全冻结。

    Config
    ------
    latent_dim : 压缩向量维度 z, 建议扫描 [64, 128, 256, 512]
    gate       : 门控融合比率 α:β:γ = 主线保留重 / 电路B贡献 / 电路C贡献
    init_dim   : 基座初始观测维度
    """
    def __init__(self, base: nn.Module, goal_dim: int, stim_dim: int,
                 latent_dim: int = 256, gate=(1.0, 0.5, 0.5)):
        super().__init__()
        self.base = base                       # 只读基座, 冻结
        self.gate = gate                       # (α, β, γ)
        self.circuit_b = GlobalGoalCircuit(goal_dim, base.init_proj.out_features,
                                           latent_dim)
        self.circuit_c = PerceptionCircuit(stim_dim, base.init_proj.out_features,
                                           latent_dim)
        # 融合投影: 把 [z_B, z_C] 对齐到基座隐空间 h 的维度
        self.fuse = nn.Linear(2 * latent_dim, base.init_proj.out_features)

    def forward(self, obs0, goal, stim):
        """
        P1.5 注入流程(与基座自回归解耦, 仅在第一步前介入一次)：
          1) h_A = base(obs0)         电路A = 基座既有历史上下文隐状态
          2) z_B = circuit_b(goal)    全局目标蓝图压缩
          3) z_C = circuit_c(stim)    瞬时感知压缩
          4) 门控融合 -> 初始残差 Δ
          5) h0 = h_A + Δ  作为基座"初始残差", 之后完全自回归
        """
        with torch.no_grad():                 # 铁律2: 冻结基座梯度
            h_A = self.base(obs0)             # 电路A(主线上下文)

        z_B = self.circuit_b(goal)            # 电路B
        z_C = self.circuit_c(stim)            # 电路C

        alpha, beta, gamma = self.gate
        merged = self.fuse(torch.cat([z_B, z_C], dim=-1))   # 对齐到隐空间
        residual = (alpha * h_A.detach()      # 主线保留
                    + (beta / (beta + gamma + 1e-6)) * merged)
        h0 = h_A + residual                    # 融合后的"初始残差"注入点

        # 此后回归基座自带的自回归解码：(仅演示, 真实接入用基座 decode(h0))
        # actions = base.decode_autoregressive(h0)
        return h0

    def parameter_groups(self):
        """只返回插件侧可训参数, 基座梯度始终关闭。"""
        blocks = [self.circuit_b, self.circuit_c, self.fuse]
        return [p for blk in blocks for p in blk.parameters() if p.requires_grad]


# --------------------------------------------------------------------------- #
# 挂载工厂：三分钟把任意现成策略包成"TPIP 扩展策略"，不改一行基座                #
# --------------------------------------------------------------------------- #
def wrap_with_tpip(base: nn.Module, goal_dim: int, stim_dim: int,
                   latent_dim: int = 256, gate=(1.0, 0.5, 0.5)) -> TPIPInjectionHook:
    """非侵入式拼接：返回一个不改基座权重、仅在 P1.5 注入初始残差的扩展模型。"""
    return TPIPInjectionHook(base, goal_dim, stim_dim, latent_dim, gate)


# --------------------------------------------------------------------------- #
# 训练插件（离策略热身阶段）——为论证"无需重训基座"                           #
# --------------------------------------------------------------------------- #
def train_plugin(hook: TPIPInjectionHook, data_loader, lr: float = 1e-3,
                 epochs: int = 1):
    """
    只优化插件参数, 基座恒为 eval + no_grad。
    把"连贯性指数 CI"(见 metrics/coherence_index.py)作为奖励/正则,
    在离策略批次上验证注入对轨迹平滑度的影响。
    """
    opt = torch.optim.Adam(hook.parameter_groups(), lr=lr)
    hook.train()
    for _ in range(epochs):
        for obs0, goal, stim, target in data_loader:
            opt.zero_grad()
            h0 = hook(obs0, goal, stim)   # 注入初始残差
            # 这里是"插件侧"唯一梯度来源, 基座全部被 detach/no_grad 隔离
            loss = F.mse_loss(h0, target)
            loss.backward()
            opt.step()