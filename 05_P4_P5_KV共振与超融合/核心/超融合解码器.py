# -*- coding: utf-8 -*-
"""超融合解码器（Ultra-Fusion Decoder, UFD）—— P1×P2×P3×P4 四架构底层融合

现有「混合锚点器」把四架构做的是**三通道加性叠加**（A 锚点偏置 + B 回响偏置 +
C 稀疏潮汐偏置，各加各的 logits）。本模块做的不是叠加，而是**机制级融合**，
让四个架构的底层能力耦合进同一条注入链路：

① DSA 动态自参照锚点（P1×P4 融合）
   P1（语义回响）的情感来源是「模型自身 hidden_state」——模型内心状态；
   P4（锚点回响）的方向来源是「外部 VAD → v_target」，每轮一次、静态。
   融合：每步捕获最后一层 hidden_state h_t（轻量只读 hook），用 P4 的锚点质心
   E 打分 v_dyn[k] = cos(h_t, e_k)，再与静态 v_target 融合：
       v_eff = normalize((1-γ)·v_target + γ·v_dyn)
   效果：锚点方向从「外部指令」升级为「外部指令 + 模型此刻内心」，即模型真的
   "先感受到、再表达"。这是 P1 的自我回响思想在 P4 嵌入空间接口上的实现，
   且不需要 P1 的 933MB 随机投影矩阵（只用 K×d 锚点矩阵，K=6,d=1536）。

② DMR 稠密乘性重加权（P3×P4 融合）
   P3（情感潮汐）的引导分布 q_emo 来自**稀疏情感词表**（千级 token 覆盖），
   OOV/网络流行语无法引导；P4 有**稠密打分表** S ∈ R^{V×K}（全词表覆盖）。
   融合：用 S 构造稠密引导分布：
       q_emo(w) = softmax(S[w]·v_eff / T_emo)
   再以 P3 的乘性重加权形式（对数空间插值，数学上有界不坍缩）：
       logits'(w) = (1-α)·logits(w) + α·log q_emo(w)
   α∈[0,1] 时 p' 介于 p 与 q_emo 之间——**不可能坍缩**；
   且 q_emo 稠密覆盖全词表——**任何 token 都能被引导**。两者单独都不具备
   "既稠密又有界"的能力。

③ P2 贯穿：α/β 参数走 目标决策器 的自适应输出（扫描表/公式兜底/架构族因子）。

注入公式（全部作用于 logits 加法，与 锚点解码器/混合锚点器 同范式）：
   logits' = (1-α)·logits + α·log_softmax(S@v_eff/T_emo)      # ② DMR（主通道）
            + β·tanh(S@v_eff/T_anchor)                          # ④ 可选锚点加性偏置
   v_eff   = normalize((1-γ)·v_target + γ·v_dyn)                # ① DSA（动态方向）

接口签名对齐 锚点解码器.生成()/注入偏置()，评测脚本可无痛接入。
"""
import math
import os
import re
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 计算熵, 计算重复率


# ══════════════════════════════════════════════════
# 最后一层定位（轻量只读 hook，捕获模型内心状态 h_t）
# ══════════════════════════════════════════════════

def _定位最后一层(model):
    """返回模型最后一层 Transformer 层模块（Qwen/LLaMA/Mistral/GPT2/OPT/BLOOM）"""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[-1]
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h[-1]
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') \
            and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers[-1]
    raise ValueError(f"无法定位模型 {type(model).__name__} 的最后一层")


class 超融合解码器:
    """P1×P2×P3×P4 机制级融合解码器：DSA 动态方向 + DMR 稠密乘性 + 可选锚点加性"""

    def __init__(
        self,
        model,
        tokenizer,
        锚点库: 锚点库,
        目标决策器: 目标决策器,
        开启DSA: bool = True,
        开启DMR: bool = True,
        开启锚点偏置: bool = False,
        γ: float = 0.3,              # DSA 融合权重：v_eff = (1-γ)·v_target + γ·v_dyn
        α基: float = 0.15,           # DMR 乘性引导强度基础（可被决策器 α 覆盖）
        α倍率: float = 1.0,          # DMR 强度倍率（α = α基×倍率）
        α上限: float = 0.6,          # DMR 强度封顶（防过引导）
        T_emo: float = 0.5,          # q_emo 软max温度（越小越锐化）
        β: Optional[float] = 0.8,    # 可选锚点加性偏置强度（None → 自动适配）
        T_anchor: float = 0.3,
        退化窗口: int = 40,
        兜底阈值: float = 0.6,
        句分隔符: str = r"[。！？!?；;\n～…~]",
        密度基: float = 0.06,
        密度增益: float = 0.10,
        密度上限: float = 0.25,
        目标长度: int = 34,
        最短字数: int = 12,
        最大字数: int = 90,
        最长句数: int = 2,
        句子停止: bool = True,
        最小长度: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.锚点库 = 锚点库
        self.目标决策器 = 目标决策器
        self.开启DSA = bool(开启DSA)
        self.开启DMR = bool(开启DMR)
        self.开启锚点偏置 = bool(开启锚点偏置)
        self.γ = γ
        self.α基 = α基
        self.α倍率 = α倍率
        self.α上限 = α上限
        self.T_emo = T_emo
        self.β = β
        self.β基 = β
        self.T_anchor = T_anchor
        self.退化窗口 = 退化窗口
        self.兜底阈值 = 兜底阈值
        self.句分隔符 = re.compile(句分隔符)
        self.密度基 = 密度基
        self.密度增益 = 密度增益
        self.密度上限 = 密度上限
        self.目标长度 = 目标长度
        self.最短字数 = 最短字数
        self.最大字数 = 最大字数
        self.最长句数 = 最长句数
        self.句子停止 = 句子停止
        self.最小长度 = 最小长度

        self.device = model.device
        self.vocab_size = int(model.config.vocab_size)

        # ── P4 载体：锚点质心 E（K,d，L2 归一化）+ 稠密打分表 S（V,K fp16）──
        self.锚点矩阵 = 锚点库.构建()                       # (K,d) 每行单位向量
        self.打分表 = 锚点库.预计算打分表()                 # (V,K) fp16
        self.K = self.锚点矩阵.shape[0]

        # ── P1 内心状态捕获（轻量只读 hook，仅 DSA 开启时注册）──
        self._钩子列表: List = []
        self.当前hidden_state: Optional[torch.Tensor] = None
        if self.开启DSA:
            self._注册钩子()

        # ── 目标状态 ──
        self.v_target: Optional[np.ndarray] = None
        self.v_eff: Optional[torch.Tensor] = None
        self.密度目标 = self.密度基

        # ── 句状态 / 兜底 ──
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表: List[int] = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0

        # ── 情感命中率词集 ──
        self._情感token集: Set[int] = self._构建情感token集()

    # ──────────────────────────────────────────────
    # P1 内心状态捕获（只读 hook）
    # ──────────────────────────────────────────────

    def _注册钩子(self) -> None:
        try:
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    hs = output[0][0, -1, :]
                else:
                    hs = output[0, -1, :]
                self.当前hidden_state = hs.detach()
            handle = _定位最后一层(self.model).register_forward_hook(hook)
            self._钩子列表.append(handle)
        except Exception as e:  # noqa: BLE001 —— hook 失败则降级为静态方向
            print(f"[超融合解码器] DSA hook 注册失败，降级为静态方向：{e}")
            self.开启DSA = False

    def _移除钩子(self) -> None:
        for handle in self._钩子列表:
            handle.remove()
        self._钩子列表.clear()

    def __del__(self):
        self._移除钩子()

    # ──────────────────────────────────────────────
    # DSA：hidden_state → 锚点空间动态方向
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _动态方向(self, v_target_np: np.ndarray) -> torch.Tensor:
        """v_dyn[k] = cos(h_t, e_k)；v_eff = normalize((1-γ)·v_target + γ·v_dyn)。

        h_t 是模型"此刻内心"的隐藏状态（含已生成文本的语境），对锚点质心打分
        即"模型此刻情感倾向在锚点空间的方向"。与外部目标 v_target 融合。
        返回 (K,) fp32 单位向量。
        """
        if self.当前hidden_state is None:
            return torch.as_tensor(v_target_np, dtype=torch.float32,
                                   device=self.device)
        h = self.当前hidden_state.float()
        hn = h / (h.norm() + 1e-9)
        A = self.锚点矩阵.float() / (self.锚点矩阵.float().norm(dim=-1, keepdim=True) + 1e-9)
        v_dyn = hn @ A.T                                   # (K,) 余弦 ∈ [-1,1]
        v_t = torch.as_tensor(v_target_np, dtype=torch.float32, device=self.device)
        v_eff = (1.0 - self.γ) * v_t + self.γ * v_dyn
        范数 = v_eff.norm() + 1e-9
        return (v_eff / 范数).float()

    # ──────────────────────────────────────────────
    # DMR：稠密乘性重加权（对数空间插值，有界）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _稠密乘性重加权(self, logits: torch.Tensor, v_eff: torch.Tensor,
                       α: float) -> torch.Tensor:
        """logits'(w) = (1-α)·logits(w) + α·log q_emo(w)

        q_emo(w) = softmax(S[w]·v_eff / T_emo)：P4 稠密打分表构造的引导分布；
        α∈[0,1] 保证 p' 介于模型分布与引导分布之间——数学上有界、不可能坍缩。
        """
        v = v_eff.to(self.打分表.dtype)
        a = self.打分表 @ v                                  # (V,) 稠密极性内积
        log_q = F.log_softmax(a / self.T_emo, dim=-1)        # (V,) log q_emo
        return (1.0 - α) * logits + α * log_q.unsqueeze(0)

    # ──────────────────────────────────────────────
    # 注入偏置（接口对齐 锚点解码器.注入偏置）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 注入偏置(self, logits: torch.Tensor, α: float) -> torch.Tensor:
        """主通道 DMR + 可选锚点加性偏置，共用同一个 v_eff 动态方向"""
        if self.v_target is None:
            return logits
        v_t = np.asarray(self.v_target, dtype=np.float32)
        self.v_eff = self._动态方向(v_t) if self.开启DSA else torch.as_tensor(
            v_t, dtype=torch.float32, device=self.device)
        if self.开启DMR and α > 0:
            logits = self._稠密乘性重加权(logits, self.v_eff, α)
        if self.开启锚点偏置 and self.β > 0:
            密度系数 = self._密度系数()
            v = self.v_eff.to(self.打分表.dtype)
            a = self.打分表 @ v
            logits = logits + (self.β * 密度系数) * torch.tanh(a / self.T_anchor).unsqueeze(0)
        return logits

    def _密度系数(self) -> float:
        """句内情感词密度超限 → 注入幅度 ×0.3（防情感词堆砌）"""
        if self._句情感词数 <= 0:
            return 1.0
        L = max(len(self._当前句文本), 1)
        if self._句情感词数 / L > self.密度目标:
            return 0.3
        return 1.0

    # ──────────────────────────────────────────────
    # 目标更新（复用 目标决策器；α 由决策器输出自适应）
    # ──────────────────────────────────────────────

    def 更新目标(self, 用户文本: str = "", 思考链文本: str = "", 指令: str = "",
                角色=None, 轮次: int = 0):
        try:
            目标 = self.目标决策器.计算目标(
                用户当前=用户文本 or None, 思考链文本=思考链文本, 指令=指令,
                角色=角色, 轮次=轮次)
            self.v_target = np.asarray(目标.v_target, dtype=np.float32)
            self.密度目标 = 目标.情感词密度目标
            # P2 贯穿：α 取 决策器潮汐α 的自适应输出，再乘用户倍率并封顶
            潮汐α = float(目标.决策日志.get("潮汐α", self.α基) or self.α基)
            self.α当前 = min(self.α上限, max(0.0, 潮汐α * self.α倍率))
            return 目标
        except Exception as e:  # noqa: BLE001
            print(f"[超融合解码器] 目标计算失败，本次不注入：{e}")
            self.v_target = None
            return None

    # ──────────────────────────────────────────────
    # 生成循环（接口对齐 锚点解码器.生成）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 生成(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        logits_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        tokenizer=None,
        轮次回调: Optional[Callable[[int, object], None]] = None,
        用户文本: str = "",
        思考链文本: str = "",
        指令: str = "",
        角色=None,
        轮次: int = 0,
    ):
        temperature = 1.0 if temperature is None else temperature
        top_p = 0.9 if top_p is None else top_p
        top_k = 50 if top_k is None else top_k
        repetition_penalty = 1.05 if repetition_penalty is None else repetition_penalty
        tokenizer = tokenizer or self.tokenizer
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        self.β = self.β基
        self.更新目标(用户文本, 思考链文本, 指令, 角色=角色, 轮次=轮次)

        # 状态重置
        past_key_values = None
        已生成 = input_ids.clone()
        已生成token集合: Set[int] = set()
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表 = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        熵列表: List[float] = []

        for 步 in range(max_new_tokens):
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(模型输入, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if repetition_penalty != 1.0:
                for tid in 已生成token集合:
                    logits[0, tid] /= repetition_penalty

            # ── 超融合注入（DMR 主通道 + 可选锚点加性）──
            logits = self.注入偏置(logits, self.α当前)

            if logits_callback is not None:
                logits_callback(步, logits)

            熵列表.append(计算熵(logits))

            if self.最小长度 > 0 and self._生成文本 and len(self._生成文本) < self.最小长度:
                if eos_token_id is not None and eos_token_id < self.vocab_size:
                    logits[0, eos_token_id] = float('-inf')

            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < top_k_values[:, -1].unsqueeze(-1)] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            已生成 = torch.cat([已生成, 下一个token], dim=-1)
            已生成token集合.add(下一个token.item())
            self._已生成token列表.append(下一个token.item())

            self._更新句状态(下一个token.item())
            if len(self._已生成token列表) >= 8 and len(self._已生成token列表) % self.退化窗口 == 0:
                self._兜底监测()

            if 轮次回调 is not None:
                轮次回调(步, self)
            if 下一个token.item() == eos_token_id:
                break
            if self.句子停止 and self._句子停止():
                break

        统计 = {
            "平均熵": round(sum(熵列表) / len(熵列表), 4) if 熵列表 else 0.0,
            "重复率": 计算重复率(self._已生成token列表),
            "情感命中率": self._情感命中率(self._已生成token列表),
            "α": round(self.α当前, 4),
            "γ": self.γ,
            "T_emo": self.T_emo,
            "β": round(self.β or 0.0, 4),
            "v_target": None if self.v_target is None
            else [round(float(x), 4) for x in self.v_target],
            "触发兜底次数": self.触发兜底次数,
        }
        return 已生成, 统计

    # ──────────────────────────────────────────────
    # 句状态 / 兜底 / 指标
    # ──────────────────────────────────────────────

    def _更新句状态(self, token_id: int) -> None:
        文本 = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if not 文本:
            return
        if token_id in self._情感token集:
            self._句情感词数 += 1
        self._生成文本 += 文本
        if self.句分隔符.search(文本):
            self._当前句文本 = ""
            self._句情感词数 = 0
            self._句子数 += 1
            self._兜底监测()
        else:
            self._当前句文本 += 文本

    def _兜底监测(self) -> None:
        """在线退化兜底：重复率 > 阈值 或 经验熵 < 0.6 → α×0.5；连续 3 次 → α=0"""
        最近 = self._已生成token列表[-self.退化窗口:]
        if len(最近) < 8:
            return
        重复率 = 1.0 - len(set(tuple(最近[i:i + 2]) for i in range(len(最近) - 1))) / max(len(最近) - 1, 1)
        熵 = 经验熵(最近)
        if 重复率 > self.兜底阈值 or 熵 < 0.6:
            self.触发兜底次数 += 1
            self._兜底计数 += 1
            if self._兜底计数 >= 3:
                self.α当前 = 0.0
                self.β = 0.0
            else:
                self.α当前 *= 0.5
                if self.β:
                    self.β *= 0.5
        else:
            self._兜底计数 = 0

    def _句子停止(self) -> bool:
        if not self._生成文本:
            return False
        if len(self._生成文本) >= self.最短字数 and self._句子数 >= self.最长句数:
            return True
        if self._句子数 >= self.最长句数 + 1:
            return True
        if len(self._生成文本) >= self.最大字数:
            return True
        return False

    def _构建情感token集(self) -> Set[int]:
        """单 token 情感词 id 集合：锚点库词集 + cnsenti 词库（若有）"""
        集: Set[int] = set()
        for 维, 词列表 in self.锚点库.词集.items():
            for 词 in 词列表:
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    集.add(ids[0])
        try:
            感知器 = self.目标决策器.感知器
            for 词 in getattr(感知器, "_正面词", set()) | getattr(感知器, "_负面词", set()):
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    集.add(ids[0])
        except Exception:  # noqa: BLE001
            pass
        return 集

    def _情感命中率(self, token列表: List[int]) -> float:
        if not token列表:
            return 0.0
        命中 = sum(1 for t in token列表 if t in self._情感token集)
        return round(命中 / len(token列表), 4)

    def 重置(self) -> None:
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表 = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        self.v_target = None
        self.β = self.β基

    def __repr__(self):
        return (f"超融合解码器(DSA={self.开启DSA}(γ={self.γ}), "
                f"DMR={self.开启DMR}(α基={self.α基},T_emo={self.T_emo}), "
                f"锚点偏置={self.开启锚点偏置}(β={self.β基}), "
                f"K={self.K})")


def 经验熵(token列表: List[int]) -> float:
    """已生成文本的 token 级经验分布熵"""
    if not token列表:
        return 0.0
    n = len(token列表)
    c = Counter(token列表)
    return float(-sum((v / n) * math.log(v / n) for v in c.values()))


if __name__ == "__main__":
    print("超融合解码器模块自检（无模型，仅接口检查）")
    print("UFD = P1(hidden_state内心) × P4(锚点打分表) 动态方向 + P3 乘性重加权形式 + P2 自适应")
