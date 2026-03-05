"""
TSP VLA Agent — 基于 InternVL3.5-8B-Flash + VLA-Adapter 的 视觉-语言-动作 模型
================================================================================

架构概览 (借鉴 VLA-Adapter: arxiv 2509.09372)
----------------------------------------------
InternVL3.5-8B-Flash 本身包含:
  ┌───────────────────────────────────┐
  │  InternViT-6B  (看图)             │
  │       ↓                           │
  │  MLP Projector  (对齐维度)        │
  │       ↓                           │
  │  Qwen3-8B LLM  (理解 + 推理)     │
  │  ← 注入 Action Query Embeddings  │
  └───────────────────────────────────┘
                   ↓
  收集 所有层 hidden states → 分离 task tokens + action tokens
                   ↓
  ┌───────────────────────────────────┐
  │  Bridge Action Head               │
  │  (N 层 BridgeAttentionBlock)      │
  │                                   │
  │  每层: 三路注意力                  │
  │    1. x ← self-attention          │
  │    2. x ← cross-attend action_h   │
  │    3. x ← cross-attend task_h     │
  │       (gated, 可学习 gate)        │
  │  + 残差 + LayerNorm + FFN         │
  └───────────────────────────────────┘
                   ↓
  LayerNorm → Linear → Sigmoid → [x_min, y_min, x_max, y_max]

三种使用模式
-----------
1. **文本模式** — 返回字符串，方便调试
   >>> agent.text_chat(image, "选一块需要优化的区域")

2. **动作模式** — 返回归一化坐标，用于端到端训练 (VLA)
   >>> agent.get_action(image, image_width=1000, image_height=1000)

3. **训练模式** — 端到端训练 Bridge Action Head + Action Queries + LoRA
   >>> loss = agent.train_step(image, prompt, gt_coords)
"""

import math
import os
import io
from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================
# 缓存配置 — 让 HuggingFace 把文件下载到指定目录
# ============================================================
_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/workspace/codes/vitsp/.hf_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", _CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", _CACHE_DIR)


# ============================================================
# 工具函数：图像预处理
# ============================================================
def build_image_transform(input_size: int = 448):
    """
    返回一个 torchvision transform 流水线，
    把任意尺寸的 PIL Image → [3, 448, 448] 的标准化 Tensor。
    """
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def plotly_fig_to_pil(fig) -> Image.Image:
    """把 Plotly figure 转成 PIL Image（需要 kaleido）"""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", engine="kaleido")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def normalize_coords_to_pixels(
    norm_coords: torch.Tensor,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    把 [0,1] 范围内的 4 个归一化值 → 实际像素坐标，
    并保证 x_min < x_max, y_min < y_max。
    """
    x1 = int(norm_coords[0].item() * width)
    y1 = int(norm_coords[1].item() * height)
    x2 = int(norm_coords[2].item() * width)
    y2 = int(norm_coords[3].item() * height)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


# ============================================================
# RoPE — 旋转位置编码 (用于 Bridge Attention)
# ============================================================
def _apply_rope(q, k, cos, sin):
    """
    对 q, k 施加旋转位置编码。
    q, k: (B, H, T, D)  D 必须是偶数
    cos, sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


# ============================================================
# Bridge Attention Block — VLA-Adapter 核心模块
# ============================================================
class BridgeAttentionBlock(nn.Module):
    """
    带门控的三路注意力残差块 (参考 VLA-Adapter MLPResNetBlock_Pro)。

    三路注意力:
      1. Self-Attention:  x 内部自注意力
      2. Action Cross-Attn: x 对 h_a (action hidden states) 做交叉注意力
      3. Task Cross-Attn:   x 对 h_t (task/视觉 patch hidden states) 做交叉注意力 (gated)

    所有路共享 Q 投影，但 K/V 各自独立投影 (Pro 版设计)。
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # Q (统一)
        self.q_proj = nn.Linear(dim, dim)
        # Self-Attention K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        # Action Cross-Attention K, V
        self.k_action = nn.Linear(dim, dim)
        self.v_action = nn.Linear(dim, dim)
        # Task Cross-Attention K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # 可学习的门控因子，控制 task 信息注入强度，零初始化
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, h_a=None, h_t=None):
        """
        x:   (B, T, D)  — action query 当前隐状态
        h_a: (B, Ka, D) — action hidden states (来自某一层)
        h_t: (B, Kt, D) — task hidden states / 视觉 patch (来自某一层)
        """
        g = torch.tanh(self.gating_factor)

        B, T, C = x.shape
        K_a = h_a.size(1) if h_a is not None else 0
        K_t = h_t.size(1) if h_t is not None else 0

        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Q
        q = reshape_heads(self.q_proj(x), B, T)
        # Self K, V
        k_self = reshape_heads(self.k_self(x), B, T)
        v_self = reshape_heads(self.v_self(x), B, T)

        # RoPE on self-attention
        cos_main, sin_main = self.rope(T, device=x.device, dtype=x.dtype)
        q, k_self = _apply_rope(q, k_self, cos_main, sin_main)

        attn_scores = [torch.matmul(q, k_self.transpose(-2, -1))]
        v_list = [v_self]

        # Action cross-attention
        if h_a is not None and K_a > 0:
            k_act = reshape_heads(self.k_action(h_a), B, K_a)
            v_act = reshape_heads(self.v_action(h_a), B, K_a)
            cos_a, sin_a = self.rope(K_a, device=x.device, dtype=x.dtype)
            _, k_act = _apply_rope(k_act, k_act, cos_a, sin_a)
            attn_scores.append(torch.matmul(q, k_act.transpose(-2, -1)))
            v_list.append(v_act)

        # Task cross-attention (gated)
        if h_t is not None and K_t > 0:
            k_tsk = reshape_heads(self.k_task(h_t), B, K_t)
            v_tsk = reshape_heads(self.v_task(h_t), B, K_t)
            cos_t, sin_t = self.rope(K_t, device=x.device, dtype=x.dtype)
            _, k_tsk = _apply_rope(k_tsk, k_tsk, cos_t, sin_t)
            attn_scores.append(torch.matmul(q, k_tsk.transpose(-2, -1)) * g)
            v_list.append(v_tsk)

        # 合并注意力分数并 softmax
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        v_combined = torch.cat(v_list, dim=2)
        output = torch.matmul(attn_weights, v_combined)  # (B, H, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # 残差 + FFN
        x = self.ffn(output + x)
        return x


# ============================================================
# Bridge Action Head — 替代原来的简单 ActionHead
# ============================================================
class BridgeActionHead(nn.Module):
    """
    VLA-Adapter 风格的 Action Head:
      1. 输入投影: input_dim → hidden_dim
      2. N 层 BridgeAttentionBlock (每层接收不同 LLM 层的 h_t 和 h_a)
      3. 输出投影: hidden_dim → action_dim, 再 Sigmoid 归一化到 [0,1]

    输入:
      multi_layer_hidden: [B, num_layers, num_task_tokens + num_action_tokens, D]
      — 从 LLM 各层提取的 (task + action) hidden states

    输出:
      [B, action_dim]  (归一化坐标 0~1)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        action_dim: int = 4,
        num_action_tokens: int = 4,
        num_blocks: int = 12,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_action_tokens = num_action_tokens
        self.num_blocks = num_blocks

        # 输入投影: 把 action_dim*hidden_dim 展平后投影回 hidden_dim
        self.input_norm = nn.LayerNorm(action_dim * hidden_dim)
        self.input_proj = nn.Linear(action_dim * hidden_dim, hidden_dim)
        self.act = nn.ReLU()

        # Bridge Attention blocks
        self.blocks = nn.ModuleList([
            BridgeAttentionBlock(hidden_dim, num_heads)
            for _ in range(num_blocks)
        ])

        # 输出头
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, multi_layer_hidden, num_task_tokens: int) -> torch.Tensor:
        """
        multi_layer_hidden: [B, num_layers, num_task+num_action, D]
        num_task_tokens: 视觉 patch 的数量

        返回: [B, action_dim]  (sigmoid 归一化)
        """
        B = multi_layer_hidden.shape[0]
        D = self.hidden_dim

        # 初始化 action query 输入 (全零，将由 bridge attention 填充)
        # shape: [B, 1, action_dim * hidden_dim] → 投影为 [B, 1, hidden_dim]
        x = torch.zeros(B, 1, self.action_dim * D,
                         device=multi_layer_hidden.device,
                         dtype=multi_layer_hidden.dtype)
        x = self.input_norm(x)
        x = self.act(self.input_proj(x))  # [B, 1, D]

        # 逐层 Bridge Attention, 每个 block 使用不同 LLM 层的特征
        num_available_layers = multi_layer_hidden.shape[1]
        for i, block in enumerate(self.blocks):
            # 选择对应层的特征 (如果 blocks > layers，循环使用)
            layer_idx = i % num_available_layers
            layer_feat = multi_layer_hidden[:, layer_idx, :, :]  # [B, N_t+N_a, D]

            h_t = layer_feat[:, :num_task_tokens, :]      # task features
            h_a = layer_feat[:, num_task_tokens:, :]       # action features

            x = block(x, h_a=h_a, h_t=h_t)

        # 输出投影
        x = self.output_norm(x[:, 0, :])      # [B, D] — 取第一个 (唯一的) token
        x = self.output_proj(x)                # [B, action_dim]
        return torch.sigmoid(x)


# ============================================================
# (保留旧 ActionHead 用于向后兼容)
# ============================================================
class ActionHead(nn.Module):
    """旧版简单 Action Head (用于向后兼容或快速测试)"""

    def __init__(self, hidden_size: int, action_dim: int = 4, mid_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.GELU(),
            nn.Linear(mid_size, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# InternVL VLA Agent  —  主角 (VLA-Adapter 架构)
# ============================================================
class InternVL_VLA_Agent(nn.Module):
    """
    基于 InternVL3.5-8B-Flash + VLA-Adapter 的 TSP 子区域选择 Agent。

    核心改进 (相比旧版):
    1. Action Query Embeddings — 可学习 embedding，注入 LLM，标记 "要输出动作" 的位置
    2. 多层 Hidden States 提取 — 不只取最后一层，而是收集所有层的 (task + action) 特征
    3. Bridge Attention Action Head — 通过三路交叉注意力 (self, action, task) 精细化动作预测

    用法速查
    --------
    # 创建 agent
    agent = InternVL_VLA_Agent("/path/to/InternVL3_5-8B-Flash")

    # 文本模式 (兼容旧接口)
    text = agent.text_chat(pil_image, "请选择子区域")
    resp, pt, ct = agent.vision_chat(fig, prior, 1, [], 0, 1000, 0, 1000)

    # VLA 动作模式
    coords = agent.get_action(pil_image, image_width=1000, image_height=1000)

    # 训练模式
    loss = agent.train_step(pil_image, "Select a sub-region.", gt_norm_coords)
    """

    def __init__(
        self,
        model_path: str = "/workspace/codes/vitsp/InternVL3_5-8B-Flash",
        action_dim: int = 4,
        num_action_tokens: int = 4,
        num_bridge_blocks: int = 12,
        num_attn_heads: int = 8,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_path = model_path
        self.action_dim = action_dim
        self.num_action_tokens = num_action_tokens
        self.num_bridge_blocks = num_bridge_blocks
        self.num_attn_heads = num_attn_heads
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_device = device

        # 延迟加载占位符
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._action_head = None       # BridgeActionHead
        self._action_queries = None    # nn.Embedding
        self._loaded = False

    # ----------------------------------------------------------
    # 延迟加载
    # ----------------------------------------------------------
    def _lazy_load(self):
        """第一次调用时才真正加载模型到 GPU，避免多进程 CUDA 上下文问题。"""
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModel

        print(f"[VLA Agent] 正在加载 InternVL3.5-Flash 从 {self.model_path} ...")

        # ① 图像预处理器
        self._transform = build_image_transform(input_size=448)

        # ② 分词器
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False,
        )

        # ③ InternVL 主模型
        self._model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        use_cuda = (self.target_device == "cuda" and torch.cuda.is_available())
        if use_cuda:
            self._model = self._model.cuda()

        # ④ 冻结主干
        if self.freeze_backbone:
            for p in self._model.parameters():
                p.requires_grad = False

        # ⑤ 可选: LoRA 微调 LLM backbone
        if self.use_lora:
            self._apply_lora()

        # ⑥ 获取 LLM 隐藏层维度
        llm_hidden = self._get_llm_hidden_size()

        # ⑦ Action Query Embeddings (零初始化，可学习)
        self._action_queries = nn.Embedding(self.num_action_tokens, llm_hidden)
        nn.init.zeros_(self._action_queries.weight)

        # ⑧ Bridge Action Head
        self._action_head = BridgeActionHead(
            hidden_dim=llm_hidden,
            action_dim=self.action_dim,
            num_action_tokens=self.num_action_tokens,
            num_blocks=self.num_bridge_blocks,
            num_heads=self.num_attn_heads,
        )

        if use_cuda:
            self._action_queries = self._action_queries.cuda()
            self._action_head = self._action_head.cuda()

        self._loaded = True
        num_trainable = sum(p.numel() for p in self.trainable_parameters())
        print(f"[VLA Agent] 加载完成！LLM hidden_size = {llm_hidden}, "
              f"可训练参数量 = {num_trainable:,}")

    def _apply_lora(self):
        """对 LLM backbone 应用 LoRA 微调。"""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            print("[VLA Agent] 警告: peft 未安装，跳过 LoRA。pip install peft")
            self.use_lora = False
            return

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        self._model.language_model = get_peft_model(
            self._model.language_model, lora_config
        )
        print(f"[VLA Agent] LoRA 已应用 (rank={self.lora_rank})")
        self._model.language_model.print_trainable_parameters()

    def _get_llm_hidden_size(self) -> int:
        """从 InternVL config 里自动读取 LLM 隐藏层维度"""
        cfg = self._model.config
        if hasattr(cfg, "llm_config"):
            return cfg.llm_config.hidden_size
        if hasattr(cfg, "text_config"):
            return cfg.text_config.hidden_size
        return 4096

    def _get_num_llm_layers(self) -> int:
        """获取 LLM transformer 层数"""
        cfg = self._model.config
        if hasattr(cfg, "llm_config"):
            return getattr(cfg.llm_config, "num_hidden_layers", 36)
        if hasattr(cfg, "text_config"):
            return getattr(cfg.text_config, "num_hidden_layers", 36)
        return 36

    def trainable_parameters(self):
        """返回所有可训练参数的迭代器 (用于构建 optimizer)"""
        self._lazy_load()
        # Action queries
        yield from self._action_queries.parameters()
        # Bridge action head
        yield from self._action_head.parameters()
        # LoRA 参数 (如果有)
        if self.use_lora:
            for p in self._model.language_model.parameters():
                if p.requires_grad:
                    yield p

    # ----------------------------------------------------------
    # 图像预处理
    # ----------------------------------------------------------
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → [1, 3, 448, 448] bfloat16 Tensor"""
        self._lazy_load()
        pv = self._transform(image).unsqueeze(0).to(dtype=torch.bfloat16)
        if self.target_device == "cuda" and torch.cuda.is_available():
            pv = pv.cuda()
        return pv

    # ==========================================================
    # 模式一：文本输出（兼容旧的 API 调用方式）
    # ==========================================================
    def text_chat(self, image: Image.Image, prompt: str) -> str:
        """输入一张图 + 一段文字提示，返回模型生成的文本。"""
        self._lazy_load()
        pv = self.preprocess_image(image)
        gen_cfg = dict(max_new_tokens=1024, do_sample=False)
        return self._model.chat(self._tokenizer, pv, f"<image>\n{prompt}", gen_cfg)

    def vision_chat(
        self,
        fig,
        prior_selection: str,
        num_region: int,
        pending_coords: list,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> Tuple[str, int, int]:
        """
        ★ 直接兼容 llm.py 里 LocalInternVL.vision_chat() 的接口 ★
        返回: (回复文本, prompt_token数, completion_token数)
        """
        self._lazy_load()

        region_names = {1: "one", 2: "two", 3: "three"}
        n_str = region_names.get(num_region, "one")

        image = plotly_fig_to_pil(fig)
        pv = self.preprocess_image(image)

        pending_str = ", ".join(
            f"<coordinates> x_min={c[0]}, x_max={c[1]}, "
            f"y_min={c[2]}, y_max={c[3]} </coordinates>"
            for c in pending_coords
        )

        prompt = (
            f"<image>\nYou are tasked with improving an existing solution to a "
            f"Traveling Salesman Problem (TSP) by selecting a sub-region where "
            f"the routes can be significantly optimized.\n"
            f"The boundary of the map is x_min={x_min-10000}, x_max={x_max+10000}, "
            f"y_min={y_min-10000}, y_max={y_max+10000}.\n"
            f"Please return {n_str} non-overlapping sub-rectangle(s) that would "
            f"most reduce total travel distance.\n"
            f"Select areas as large as you could to cover more nodes.\n"
            f"Output format:\n"
            f"<coordinates> x_min= 1,000, x_max= 2,000, y_min= 1,000, y_max=2,000 </coordinates>\n"
            f"\nAvoid selecting: {pending_str}\n"
            f"\nPrevious trajectory:\n{prior_selection}"
        )

        gen_cfg = dict(max_new_tokens=1024, do_sample=False)
        try:
            resp = self._model.chat(self._tokenizer, pv, prompt, gen_cfg)
            print(f"[VLA Agent] 文本输出: {resp}")
            pt = len(self._tokenizer.encode(prompt))
            ct = len(self._tokenizer.encode(resp)) if resp else 0
            return resp, pt, ct
        except Exception as e:
            print(f"[VLA Agent] 推理出错: {e}")
            return "", 0, 0

    # ==========================================================
    # 模式二：动作输出 (VLA-Adapter forward)
    # ==========================================================
    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        VLA-Adapter 核心 forward:
          图像 Tensor + 文本 Token IDs → 归一化坐标 [batch, action_dim]

        流程:
          1. 图像 → vision_model → projector → 视觉 patch embeddings
          2. 文本 → token embedding
          3. 在文本 embedding 末尾注入 Action Query Embeddings
          4. 拼接 [视觉 patches, 文本+action queries] → 送入 LLM
          5. 收集所有层 hidden states，分离 task tokens 和 action tokens
          6. 送入 BridgeActionHead → 坐标输出
        """
        self._lazy_load()
        B = pixel_values.shape[0]

        # 1) 视觉特征 → 投影
        vis_out = self._model.vision_model(pixel_values)
        vis_embeds = vis_out.last_hidden_state                # [B, num_patches, vit_dim]
        vis_embeds = self._model.mlp1(vis_embeds)             # [B, num_patches, llm_dim]
        num_vis_tokens = vis_embeds.shape[1]

        # 2) 文本 embedding
        embed_layer = self._model.language_model.get_input_embeddings()
        if self.use_lora:
            # peft 包装后，需要通过 base_model 访问
            base_lm = self._model.language_model
            if hasattr(base_lm, 'model'):
                embed_layer = base_lm.model.get_input_embeddings()
        txt_embeds = embed_layer(prompt_ids)                  # [B, seq_len, llm_dim]

        # 3) 生成 Action Query embeddings 并拼接到文本末尾
        aq_indices = torch.arange(self.num_action_tokens, device=pixel_values.device)
        action_query_embeds = self._action_queries(aq_indices)   # [num_action_tokens, llm_dim]
        action_query_embeds = action_query_embeds.unsqueeze(0).expand(B, -1, -1)
        txt_with_aq = torch.cat([txt_embeds, action_query_embeds], dim=1)

        num_text_tokens = txt_embeds.shape[1]
        num_aq = self.num_action_tokens

        # 4) 拼接 [视觉patches | 文本 | action queries] → LLM
        combined = torch.cat([vis_embeds, txt_with_aq], dim=1)
        # combined shape: [B, num_vis + num_text + num_aq, llm_dim]

        lm = self._model.language_model
        llm_out = lm(
            inputs_embeds=combined,
            output_hidden_states=True,
        )

        # 5) 收集所有层 hidden states，分离 task 和 action 部分
        #    hidden_states: tuple of (num_layers+1) tensors, 每个 [B, total_seq, D]
        #    位置布局: [vis_tokens | text_tokens | action_query_tokens]
        all_hidden = llm_out.hidden_states  # tuple

        # 选择要使用的层 (均匀采样或全部使用)
        num_layers_available = len(all_hidden)
        if num_layers_available > self.num_bridge_blocks:
            # 均匀采样
            indices = torch.linspace(0, num_layers_available - 1,
                                     self.num_bridge_blocks).long().tolist()
        else:
            indices = list(range(num_layers_available))

        multi_layer_feats = []
        for idx in indices:
            h = all_hidden[idx]  # [B, total_seq, D]
            # task tokens = 视觉 patch 位置
            task_h = h[:, :num_vis_tokens, :]
            # action tokens = 最后 num_aq 个位置
            action_h = h[:, -num_aq:, :]
            combined_h = torch.cat([task_h, action_h], dim=1)  # [B, N_t+N_a, D]
            multi_layer_feats.append(combined_h.unsqueeze(1))  # [B, 1, N_t+N_a, D]

        multi_layer_hidden = torch.cat(multi_layer_feats, dim=1)
        # shape: [B, num_selected_layers, num_vis+num_aq, D]

        # 6) Bridge Action Head → 归一化坐标
        return self._action_head(multi_layer_hidden, num_task_tokens=num_vis_tokens)

    # ----------------------------------------------------------
    # 训练接口
    # ----------------------------------------------------------
    def train_step(
        self,
        image: Image.Image,
        prompt: str,
        gt_norm_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步训练: 图像 + prompt + ground truth 归一化坐标 → L1 loss。

        Args:
            image: PIL Image
            prompt: 文字提示
            gt_norm_coords: [4] 归一化坐标, 值域 [0,1]

        Returns:
            loss: L1 loss tensor (可直接 backward)
        """
        self._lazy_load()
        self.train()

        pv = self.preprocess_image(image)
        ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        if self.target_device == "cuda" and torch.cuda.is_available():
            ids = ids.cuda()
            gt_norm_coords = gt_norm_coords.cuda()

        pred = self.forward(pv, ids)  # [1, 4]
        gt = gt_norm_coords.unsqueeze(0).to(pred.dtype)  # [1, 4]

        return F.l1_loss(pred, gt)

    # ----------------------------------------------------------
    # 高层动作接口（给外部调用）
    # ----------------------------------------------------------
    def get_action(
        self,
        image: Image.Image,
        prompt: str = "Select a sub-region to optimize.",
        image_width: int = 512,
        image_height: int = 512,
    ) -> Tuple[int, int, int, int]:
        """
        输入 PIL Image → 输出像素级坐标 (x_min, y_min, x_max, y_max)。
        """
        self._lazy_load()
        self.eval()

        with torch.no_grad():
            pv = self.preprocess_image(image)
            ids = self._tokenizer(prompt, return_tensors="pt").input_ids
            if self.target_device == "cuda" and torch.cuda.is_available():
                ids = ids.cuda()

            norm = self.forward(pv, ids)[0]  # [4]
            return normalize_coords_to_pixels(norm, image_width, image_height)

    def get_action_from_plotly(
        self,
        fig,
        prompt: str = "Select a sub-region to optimize.",
        image_width: int = 512,
        image_height: int = 512,
    ) -> Tuple[int, int, int, int]:
        """便捷接口：直接传 Plotly figure"""
        return self.get_action(
            plotly_fig_to_pil(fig), prompt, image_width, image_height,
        )

    # ----------------------------------------------------------
    # 保存 / 加载可训练参数
    # ----------------------------------------------------------
    def save_trainable(self, path: str):
        """只保存可训练部分: action_queries, action_head, (lora weights)"""
        self._lazy_load()
        state = {
            "action_queries": self._action_queries.state_dict(),
            "action_head": self._action_head.state_dict(),
        }
        if self.use_lora:
            lora_state = {
                k: v for k, v in self._model.language_model.state_dict().items()
                if "lora" in k.lower()
            }
            state["lora"] = lora_state
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)
        print(f"[VLA Agent] 可训练参数已保存到 {path}")

    def load_trainable(self, path: str):
        """加载可训练部分"""
        self._lazy_load()
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._action_queries.load_state_dict(state["action_queries"])
        self._action_head.load_state_dict(state["action_head"])
        if self.use_lora and "lora" in state:
            missing, unexpected = self._model.language_model.load_state_dict(
                state["lora"], strict=False
            )
            if unexpected:
                print(f"[VLA Agent] LoRA 加载: {len(unexpected)} unexpected keys")
        print(f"[VLA Agent] 可训练参数已从 {path} 加载")