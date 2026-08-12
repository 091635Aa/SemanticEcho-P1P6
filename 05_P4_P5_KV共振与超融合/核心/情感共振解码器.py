# -*- coding: utf-8 -*-
"""P5 KV 情感共振解码器（KV-Emotion Resonance Decoding, KER）

按 P5_KV情感共振解码_设计方案.md：
操作空间 = 注意力缓存空间（KV cache）——第五个空间，与 P1(表示)/P3(概率)/P4(嵌入) 均不同。

核心机制（不动 logits、不动权重，只调 KV 缓存里的注意力分配）：
  ① 稠密情感定位：g(p) = S[token_p]·v_eff（P4 稠密打分表，全词表覆盖）
  ② key 调制：    K_l[:,:,p,:] *= (1 + κ·clip(g,0,1))   → 注意力向情感 token 共振
  ③ value 调制：  V_l[:,:,p,:] *= (1 + κ_v·clip(g,0,1))  → 情感记忆放大（可选）
  ④ 决策层：      v_eff = normalize((1-γ)·v_target + γ·v_dyn)（P1 内心状态融合）
                  κ = κ基 × (1+Δarousal) × min(1, 活跃度×2.5)（P2 自适应）
  ⑤ 可选叠加：    P4 锚点 logits 注入（logits += β·tanh(S@v_eff/T)），与 KV 调制正交

接口签名对齐 锚点解码器.生成()/注入偏置()。
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


class 情感共振解码器:
    """P5 KER：KV 缓存情感调制 + 可选 P4 logits 注入（P1×P2×P3×P4 混合）"""

    def __init__(
        self,
        model,
        tokenizer,
        锚点库: 锚点库,
        目标决策器: 目标决策器,
        开启KV调制: bool = True,
        开启V调制: bool = False,
        开启DSA: bool = True,          # P1 内心状态融合（v_dyn）
        开启锚点注入: bool = False,     # 可选 P4 logits 注入（叠加验证）
        γ: float = 0.3,                # DSA 融合权重
        κ基: float = 0.15,             # key 调制强度基础
        κ上限: float = 0.5,            # key 调制强度封顶
        κ_v基: float = 0.06,           # value 调制强度基础（开启V调制时）
        情感阈值: float = 0.08,         # 情感 token 定位阈值（cos 打分）
        调制层数: int = 4,             # 调制最后 N 层
        β: Optional[float] = 0.8,      # 可选 P4 logits 注入强度
        T_anchor: float = 0.3,
        退化窗口: int = 40,
        兜底阈值: float = 0.6,
        句分隔符: str = r"[。！？!?；;\n～…~]",
        密度基: float = 0.06,
        密度增益: float = 0.10,
        密度上限: float = 0.25,
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
        self.开启KV调制 = bool(开启KV调制)
        self.开启V调制 = bool(开启V调制)
        self.开启DSA = bool(开启DSA)
        self.开启锚点注入 = bool(开启锚点注入)
        self.γ = γ
        self.κ基 = κ基
        self.κ上限 = κ上限
        self.κ_v基 = κ_v基
        self.情感阈值 = 情感阈值
        self.调制层数 = 调制层数
        self.β = β
        self.β基 = β
        self.T_anchor = T_anchor
        self.退化窗口 = 退化窗口
        self.兜底阈值 = 兜底阈值
        self.句分隔符 = re.compile(句分隔符)
        self.密度基 = 密度基
        self.密度增益 = 密度增益
        self.密度上限 = 密度上限
        self.最短字数 = 最短字数
        self.最大字数 = 最大字数
        self.最长句数 = 最长句数
        self.句子停止 = 句子停止
        self.最小长度 = 最小长度

        self.device = model.device
        self.vocab_size = int(model.config.vocab_size)

        # ── P4 载体：锚点矩阵 E（K,d）+ 稠密打分表 S（V,K fp16）──
        self.锚点矩阵 = 锚点库.构建()
        self.打分表 = 锚点库.预计算打分表()
        self.K = self.锚点矩阵.shape[0]

        # ── P1 内心状态捕获（轻量只读 hook，仅 DSA 开启时注册）──
        self._钩子列表: List = []
        self.当前hidden_state: Optional[torch.Tensor] = None
        if self.开启DSA:
            self._注册钩子()

        # ── 目标状态 ──
        self.v_target: Optional[np.ndarray] = None
        self.v_eff: Optional[torch.Tensor] = None
        self.κ当前 = 0.0
        self.κ_v当前 = 0.0
        self.密度目标 = self.密度基

        # ── 句状态 / 兜底 ──
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表: List[int] = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        self._已调制位置 = 0

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
        except Exception as e:  # noqa: BLE001
            print(f"[情感共振解码器] DSA hook 注册失败，降级为静态方向：{e}")
            self.开启DSA = False

    def _移除钩子(self) -> None:
        for handle in self._钩子列表:
            handle.remove()
        self._钩子列表.clear()

    def __del__(self):
        self._移除钩子()

    # ──────────────────────────────────────────────
    # 决策层：v_eff（P1 内心融合）+ κ（P2 自适应）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _动态方向(self, v_target_np: np.ndarray) -> torch.Tensor:
        """v_eff = normalize((1-γ)·v_target + γ·v_dyn)，v_dyn[k] = cos(h_t, e_k)"""
        if self.当前hidden_state is None:
            return torch.as_tensor(v_target_np, dtype=torch.float32, device=self.device)
        h = self.当前hidden_state.float()
        hn = h / (h.norm() + 1e-9)
        A = self.锚点矩阵.float() / (self.锚点矩阵.float().norm(dim=-1, keepdim=True) + 1e-9)
        v_dyn = hn @ A.T
        v_t = torch.as_tensor(v_target_np, dtype=torch.float32, device=self.device)
        v_eff = (1.0 - self.γ) * v_t + self.γ * v_dyn
        return (v_eff / (v_eff.norm() + 1e-9)).float()

    def 更新目标(self, 用户文本: str = "", 思考链文本: str = "", 指令: str = "",
                角色=None, 轮次: int = 0):
        try:
            目标 = self.目标决策器.计算目标(
                用户当前=用户文本 or None, 思考链文本=思考链文本, 指令=指令,
                角色=角色, 轮次=轮次)
            self.v_target = np.asarray(目标.v_target, dtype=np.float32)
            self.密度目标 = 目标.情感词密度目标
            # P2 自适应：κ = κ基 × (1+Δ唤醒) × min(1, 活跃度×2.5)，封顶
            日志 = 目标.决策日志 or {}
            Δ唤醒 = float(日志.get("Δ唤醒", 0.0) or 0.0)
            活跃度 = float(日志.get("活跃度", 0.0) or 0.0)
            目标强度 = float(日志.get("目标强度", 0.0) or 0.0)
            κ = self.κ基 * (1.0 + Δ唤醒) * min(1.0, max(活跃度, 目标强度) * 2.5)
            self.κ当前 = min(self.κ上限, max(0.0, κ))
            self.κ_v当前 = min(self.κ上限, self.κ_v基 * (1.0 + Δ唤醒))
            return 目标
        except Exception as e:  # noqa: BLE001
            print(f"[情感共振解码器] 目标计算失败：{e}")
            self.v_target = None
            return None

    # ──────────────────────────────────────────────
    # 表达层：KV 缓存情感调制（核心）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _调制KV缓存(self, past_key_values, token_ids: torch.Tensor):
        """对 KV 缓存做情感调制（只调制新位置，避免旧位置尺度重复叠加）。

        g(p) = S[token_p]·v_eff → clip [0,1]；
        K[p] *= (1 + κ·g)；可选 V[p] *= (1 + κ_v·g)，作用于最后 调制层数 层。

        兼容 transformers 5.x：past_key_values 为 Cache 对象，迭代产出
        (key, value, ...) 元组（元素数 2~3），统一按 [0]/[1] 取张量，就地缩放。
        """
        if past_key_values is None or not self.开启KV调制 or self.v_target is None:
            return past_key_values
        if self.κ当前 <= 0:
            return past_key_values
        ids = token_ids[0]
        T = len(ids)
        开始 = self._已调制位置
        if T <= 开始:
            return past_key_values
        # 只对 [开始, T) 新位置计算情感强度（已调制位置不再重调 → 无尺度叠加）
        v = self.v_eff.to(self.打分表.dtype)
        S_ids = self.打分表[ids[开始:]]
        scores = (S_ids @ v).float()
        g = torch.clamp(scores, 0.0, 1.0)
        掩码 = g > self.情感阈值
        self._已调制位置 = T
        if not 掩码.any():
            return past_key_values
        尺度k = torch.where(掩码, 1.0 + self.κ当前 * g, torch.ones_like(g))
        sk = 尺度k.reshape(1, 1, -1, 1)                        # (1,1,N,1)
        sv = None
        if self.开启V调制 and self.κ_v当前 > 0:
            尺度v = torch.where(掩码, 1.0 + self.κ_v当前 * g, torch.ones_like(g))
            sv = 尺度v.reshape(1, 1, -1, 1)
        层条目 = list(past_key_values)
        n = len(层条目)
        起始 = max(0, n - self.调制层数)
        for l in range(起始, n):
            k = 层条目[l][0]
            if k is None:
                continue
            k[:, :, 开始:, :] *= sk.to(k.dtype)               # 就地缩放情感位置 key
            if sv is not None and 层条目[l][1] is not None:
                层条目[l][1][:, :, 开始:, :] *= sv.to(层条目[l][1].dtype)
        return past_key_values

    # ──────────────────────────────────────────────
    # 可选 P4 logits 注入（叠加验证）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _锚点注入(self, logits: torch.Tensor) -> torch.Tensor:
        """P4 锚点加性偏置：logits += β·tanh(S@v_eff/T_anchor)"""
        if not self.开启锚点注入 or self.β <= 0 or self.v_eff is None:
            return logits
        密度系数 = self._密度系数()
        v = self.v_eff.to(self.打分表.dtype)
        a = self.打分表 @ v
        return logits + (self.β * 密度系数) * torch.tanh(a / self.T_anchor).unsqueeze(0)

    def _密度系数(self) -> float:
        if self._句情感词数 <= 0:
            return 1.0
        L = max(len(self._当前句文本), 1)
        if self._句情感词数 / L > self.密度目标:
            return 0.3
        return 1.0

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
        if self.v_target is not None:
            self.v_eff = self._动态方向(self.v_target)

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
        self._已调制位置 = 0
        熵列表: List[float] = []

        for 步 in range(max_new_tokens):
            # KV 调制缓存（含上一轮调制结果 → 本轮前向直接使用）
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(模型输入, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # P5 核心：对 KV 缓存做情感调制（首 token 之后生效）
            past_key_values = self._调制KV缓存(past_key_values, 已生成)

            if repetition_penalty != 1.0:
                for tid in 已生成token集合:
                    logits[0, tid] /= repetition_penalty

            # 可选 P4 logits 注入（叠加）
            logits = self._锚点注入(logits)

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
            "κ": round(self.κ当前, 4),
            "κ_v": round(self.κ_v当前, 4),
            "γ": self.γ,
            "调制层数": self.调制层数,
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
        最近 = self._已生成token列表[-self.退化窗口:]
        if len(最近) < 8:
            return
        重复率 = 1.0 - len(set(tuple(最近[i:i + 2]) for i in range(len(最近) - 1))) / max(len(最近) - 1, 1)
        熵 = 经验熵(最近)
        if 重复率 > self.兜底阈值 or 熵 < 0.6:
            self.触发兜底次数 += 1
            self._兜底计数 += 1
            if self._兜底计数 >= 3:
                self.κ当前 = 0.0
                self.β = 0.0
            else:
                self.κ当前 *= 0.5
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
        self._已调制位置 = 0
        self.v_target = None
        self.β = self.β基

    def __repr__(self):
        return (f"情感共振解码器(KV调制={self.开启KV调制}(κ基={self.κ基},层={self.调制层数}), "
                f"V调制={self.开启V调制}, DSA={self.开启DSA}(γ={self.γ}), "
                f"锚点注入={self.开启锚点注入}(β={self.β基}), K={self.K})")


def 经验熵(token列表: List[int]) -> float:
    if not token列表:
        return 0.0
    n = len(token列表)
    c = Counter(token列表)
    return float(-sum((v / n) * math.log(v / n) for v in c.values()))


if __name__ == "__main__":
    print("P5 KV 情感共振解码器模块自检")
    print("P5 = P3感知 + (P1内心×P4打分表)方向 + P2自适应κ + KV缓存调制(新空间)")
