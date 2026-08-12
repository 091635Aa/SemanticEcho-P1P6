# -*- coding: utf-8 -*-
import json, io
def add(ep, idx, extra):
    p = rf'f:\lora外挂\语义回响_系列\{ep}\script.json'
    s = json.load(io.open(p, encoding='utf-8-sig'))
    sc = s['scenes'][:idx] + extra + s['scenes'][idx:]
    s['scenes'] = sc
    io.open(p, 'w', encoding='utf-8').write(json.dumps(s, ensure_ascii=False, indent=2))
    print(ep, len(sc))

add('ep03_anchor', 5, [
    {"kicker":"P3 · ORTHOGONAL","title":"Three channels, stackable","points":["Anchor x Echo x Tide","Independent injection paths","Local 7B full stack +55.6%"],
     "en":"P3 is also stackable. The mixed anchor fuses anchor, echo and tide on three orthogonal channels. On a local 7 billion judge, the full stack beats baseline by 55.6 percent, with two independent judges agreeing on direction.","zh":"P3 还可以叠加。混合锚点把锚点、回响、潮汐放在三条正交通道上融合。本地 7B 裁判下，全开比基线提升 55.6%，两个独立裁判方向一致。"},
    {"kicker":"P3 · DEPLOY","title":"Degrades gracefully","points":["Local embeddings: main path","Logprobs-only: cloud fallback","Pure prompt: zero internal access"],
     "en":"Deployment is flexible. With local weights, it reads embeddings directly. On cloud APIs, it falls back to logprobs, with entropy difference under one percent. In the most locked-down mode, it degrades to pure prompt, zero internal access.","zh":"部署很灵活。有本地权重时直读嵌入；云端 API 时降级用 logprobs，熵差不到 1%；在封锁最严的模式下，退化为纯提示词，零内部访问。"},
])
add('ep04_kv', 5, [
    {"kicker":"P4 · SEED","title":"Seeds matter","points":["Win rate is seed-sensitive","Each mode uses its own seed","Future: multi-seed average"],
     "en":"One honest note on numbers. The evaluation is seed sensitive. Each mode runs with its own random seed, so the absolute win rate swings a lot between seeds. The right way to read these numbers is multi-seed averaging, which is the next planned step.","zh":"关于数字有一个诚实提醒：评测对随机种子敏感。每种模式用独立种子运行，绝对胜率在不同种子间波动很大。正确的读法是多种子平均，这也是下一步计划。"},
    {"kicker":"P4 · IMPLEMENTATION","title":"Runtime cache, zero writes","points":["Scale keys in-place","Track positions to avoid stacking","No persisted changes"],
     "en":"The implementation detail. The transformer cache is scaled in place, and token positions are tracked so the same factor is never applied twice. Nothing is persisted. Every run starts clean.","zh":"实现细节：transformer 缓存在原地缩放，并跟踪 token 位置防止同一系数被叠加两次。不留任何持久化，每次运行都从干净状态开始。"},
])
add('ep05_ufd', 5, [
    {"kicker":"P5 · COST","title":"Almost free","points":["Anchor matrix: K x d","VRAM increase ~ 0","933MB projection replaced"],
     "en":"And the cost is tiny. P1's 933 megabyte projection is replaced by a small K by d anchor matrix. The VRAM increase is effectively zero, and the extra latency is a lightweight hook plus a dense multiply.","zh":"而代价很小。P1 那 933MB 的随机投影被一个小型 K×d 锚点矩阵替代。显存增量几乎为零，额外延迟只有一个轻量钩子和一次稠密乘法。"},
    {"kicker":"P5 · ROADMAP","title":"From fragments to the whole","points":["P1-P4 are fragments","P5 is the synthesis demo","Full benchmarks: next step"],
     "en":"The roadmap. P1 through P4 are fragments, each proven in its own space. P5 proves they can fuse into one chain. The remaining work is the full five-benchmark, multi-seed, multi-model evaluation. That is the path to the complete body.","zh":"路线图：P1 到 P4 是碎片，各自在自己的空间被验证。P5 证明它们能融合成一条链。剩余工作是完整的五基准、多种子、多模型评测，这是通往完全体的路。"},
])
