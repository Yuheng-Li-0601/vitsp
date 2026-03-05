"""
InternVL VLA Agent 测试集
========================

测试分为三层：
  1. 纯 CPU / 不需要模型的单元测试 (test_unit_*)
     → ActionHead 形状、坐标转换、图像预处理
  2. 使用 Mock 模拟模型的集成测试 (test_mock_*)
     → 验证数据流和接口正确性，不需要 GPU / 不需要下载模型
  3. 真实模型测试 (test_real_*)
     → 需要 GPU + 本地模型权重，跑 CI 时可跳过

运行方式
--------
    # 只跑不需要 GPU 的测试（推荐日常开发用）
    pytest test_vla_agent.py -k "not test_real" -v

    # 跑全部测试（需要 GPU + 模型权重）
    pytest test_vla_agent.py -v
"""

import sys
import os
import pytest
import torch
import torch.nn as nn
from PIL import Image
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# 确保项目根目录在 sys.path 里
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from LLM_TSP.vla_agent import (
    ActionHead,
    InternVL_VLA_Agent,
    build_image_transform,
    normalize_coords_to_pixels,
    plotly_fig_to_pil,
)


# ============================================================
# 辅助工具
# ============================================================
def _make_dummy_image(width=256, height=256, color=(255, 0, 0)):
    """生成一张纯色 PIL Image，用于测试"""
    return Image.new("RGB", (width, height), color)


# ============================================================
# 第一层：纯单元测试（CPU，无模型依赖）
# ============================================================
class TestActionHead:
    """测试 Action Head 网络独立工作是否正常"""

    def test_output_shape(self):
        """输出维度应该是 [batch, action_dim]"""
        head = ActionHead(hidden_size=128, action_dim=4, mid_size=64)
        x = torch.randn(2, 128)  # batch=2
        out = head(x)
        assert out.shape == (2, 4), f"期望 (2,4)，得到 {out.shape}"

    def test_output_range_sigmoid(self):
        """Sigmoid 输出应该在 [0, 1] 范围内"""
        head = ActionHead(hidden_size=64, action_dim=4)
        # 用很大的输入值来测试边界情况
        x = torch.randn(100, 64) * 10
        out = head(x)
        assert out.min() >= 0.0, f"最小值 {out.min()} < 0"
        assert out.max() <= 1.0, f"最大值 {out.max()} > 1"

    def test_different_action_dims(self):
        """支持不同的 action_dim（比如 2D / 6D）"""
        for dim in [2, 4, 6, 8]:
            head = ActionHead(hidden_size=256, action_dim=dim)
            out = head(torch.randn(1, 256))
            assert out.shape == (1, dim)

    def test_gradient_flow(self):
        """确保梯度可以反向传播到 Action Head 的参数"""
        head = ActionHead(hidden_size=64, action_dim=4)
        x = torch.randn(1, 64, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        # 检查每一层的参数都有梯度
        for name, p in head.named_parameters():
            assert p.grad is not None, f"参数 {name} 没有梯度"
            assert p.grad.abs().sum() > 0, f"参数 {name} 梯度全为零"


class TestNormalizeCoordsToPixels:
    """测试坐标从 [0,1] → 像素值的转换"""

    def test_basic_conversion(self):
        """(0.1, 0.2, 0.8, 0.9) × (1000, 500) → (100, 100, 800, 450)"""
        coords = torch.tensor([0.1, 0.2, 0.8, 0.9])
        result = normalize_coords_to_pixels(coords, width=1000, height=500)
        assert result == (100, 100, 800, 450)

    def test_auto_swap_min_max(self):
        """如果模型输出 x1 > x2，应该自动交换"""
        # 故意让 x1 > x2
        coords = torch.tensor([0.9, 0.8, 0.1, 0.2])
        result = normalize_coords_to_pixels(coords, width=100, height=100)
        x_min, y_min, x_max, y_max = result
        assert x_min <= x_max, f"x_min({x_min}) > x_max({x_max})"
        assert y_min <= y_max, f"y_min({y_min}) > y_max({y_max})"

    def test_zero_coords(self):
        """全零坐标 → (0,0,0,0)"""
        coords = torch.tensor([0.0, 0.0, 0.0, 0.0])
        result = normalize_coords_to_pixels(coords, width=512, height=512)
        assert result == (0, 0, 0, 0)

    def test_one_coords(self):
        """全 1 坐标 → (w, h, w, h)"""
        coords = torch.tensor([1.0, 1.0, 1.0, 1.0])
        result = normalize_coords_to_pixels(coords, width=512, height=512)
        assert result == (512, 512, 512, 512)


class TestBuildImageTransform:
    """测试图像预处理流水线"""

    def test_output_shape(self):
        """输出应该是 [3, input_size, input_size]"""
        transform = build_image_transform(input_size=448)
        img = _make_dummy_image(100, 200)
        tensor = transform(img)
        assert tensor.shape == (3, 448, 448)

    def test_different_input_sizes(self):
        """不同的 input_size 应该都能正确处理"""
        for size in [224, 336, 448]:
            transform = build_image_transform(input_size=size)
            tensor = transform(_make_dummy_image())
            assert tensor.shape == (3, size, size)

    def test_rgba_to_rgb(self):
        """RGBA 图像应该自动转为 RGB"""
        rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        transform = build_image_transform()
        tensor = transform(rgba_img)
        assert tensor.shape[0] == 3  # 通道数为 3

    def test_grayscale_to_rgb(self):
        """灰度图应该自动转为 RGB"""
        gray_img = Image.new("L", (100, 100), 128)
        transform = build_image_transform()
        tensor = transform(gray_img)
        assert tensor.shape[0] == 3


# ============================================================
# 第二层：Mock 集成测试（验证数据流，不需要 GPU / 模型权重）
# ============================================================
def _build_mock_agent():
    """
    创建一个 mock 版本的 InternVL_VLA_Agent。
    用假的模型和分词器替代真实的 8B 参数模型，
    这样测试就不需要 GPU 也不需要下载权重了。
    """
    agent = InternVL_VLA_Agent.__new__(InternVL_VLA_Agent)
    nn.Module.__init__(agent)

    agent.model_path = "mock_model"
    agent.action_dim = 4
    agent.freeze_backbone = True
    agent.target_device = "cpu"

    # 模拟 InternVL 的各个子模块
    hidden_size = 128

    # vision_model: 接收图像，输出 [batch, num_patches, vision_dim]
    mock_vision = MagicMock()
    mock_vision.return_value = SimpleNamespace(
        last_hidden_state=torch.randn(1, 16, hidden_size)
    )

    # mlp1 (projector): vision_dim → llm_dim
    mock_mlp1 = nn.Identity()  # 简单起见，维度一样

    # language_model
    mock_lm = MagicMock()
    mock_lm.get_input_embeddings.return_value = nn.Embedding(1000, hidden_size)
    mock_lm.return_value = SimpleNamespace(
        hidden_states=[torch.randn(1, 20, hidden_size)]  # 只有 1 层
    )

    # 组装 mock model
    mock_model = MagicMock()
    mock_model.vision_model = mock_vision
    mock_model.mlp1 = mock_mlp1
    mock_model.language_model = mock_lm
    mock_model.config = SimpleNamespace(
        llm_config=SimpleNamespace(hidden_size=hidden_size)
    )

    # mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    mock_tokenizer.return_value = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3, 4]])
    )

    agent._model = mock_model
    agent._tokenizer = mock_tokenizer
    agent._transform = build_image_transform()
    agent._action_head = ActionHead(hidden_size, action_dim=4)
    agent._loaded = True

    return agent


class TestMockForward:
    """使用 mock 模型测试 forward 数据流"""

    def test_forward_shape(self):
        """forward 应该输出 [batch, 4]"""
        agent = _build_mock_agent()
        pv = torch.randn(1, 3, 448, 448)
        ids = torch.tensor([[1, 2, 3]])
        out = agent.forward(pv, ids)
        assert out.shape == (1, 4), f"期望 (1,4)，得到 {out.shape}"

    def test_forward_range(self):
        """forward 输出值应该在 [0, 1]"""
        agent = _build_mock_agent()
        pv = torch.randn(1, 3, 448, 448)
        ids = torch.tensor([[1, 2, 3]])
        out = agent.forward(pv, ids)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestMockGetAction:
    """使用 mock 模型测试 get_action 高层接口"""

    def test_returns_four_ints(self):
        """get_action 应该返回 4 个整数"""
        agent = _build_mock_agent()
        img = _make_dummy_image()
        result = agent.get_action(img, image_width=1000, image_height=1000)
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)

    def test_coords_in_range(self):
        """坐标应该在 [0, width/height] 范围内"""
        agent = _build_mock_agent()
        w, h = 500, 300
        img = _make_dummy_image()
        x_min, y_min, x_max, y_max = agent.get_action(img, image_width=w, image_height=h)
        assert 0 <= x_min <= x_max <= w
        assert 0 <= y_min <= y_max <= h

    def test_min_less_than_max(self):
        """x_min <= x_max 且 y_min <= y_max 应该永远成立"""
        agent = _build_mock_agent()
        for _ in range(10):  # 多跑几次（随机初始化的 action head）
            x_min, y_min, x_max, y_max = agent.get_action(
                _make_dummy_image(), image_width=1000, image_height=1000
            )
            assert x_min <= x_max
            assert y_min <= y_max


class TestMockTextChat:
    """使用 mock 测试文本聊天接口"""

    def test_text_chat_calls_model(self):
        """text_chat 应该调用 model.chat()"""
        agent = _build_mock_agent()
        agent._model.chat = MagicMock(return_value="test response")
        result = agent.text_chat(_make_dummy_image(), "选一个区域")
        assert result == "test response"
        agent._model.chat.assert_called_once()

    def test_text_chat_includes_image_token(self):
        """prompt 里应该包含 <image> tag"""
        agent = _build_mock_agent()
        agent._model.chat = MagicMock(return_value="ok")
        agent.text_chat(_make_dummy_image(), "hello")
        # 检查传给 model.chat 的 prompt 参数
        call_args = agent._model.chat.call_args
        prompt_arg = call_args[0][2]  # 第三个位置参数
        assert "<image>" in prompt_arg


class TestMockVisionChat:
    """使用 mock 测试 vision_chat 接口（兼容 LocalInternVL）"""

    def test_vision_chat_returns_tuple(self):
        """vision_chat 应该返回 (str, int, int)"""
        agent = _build_mock_agent()
        agent._model.chat = MagicMock(return_value="<coordinates> x_min=100 </coordinates>")

        # 创建一个假的 plotly figure
        mock_fig = MagicMock()
        mock_fig.write_image = MagicMock(side_effect=lambda buf, **kw: _write_fake_png(buf))

        resp, pt, ct = agent.vision_chat(mock_fig, "", 1, [], 0, 1000, 0, 1000)
        assert isinstance(resp, str)
        assert isinstance(pt, int)
        assert isinstance(ct, int)


class TestAgentInit:
    """测试 Agent 初始化逻辑"""

    def test_lazy_load_not_triggered(self):
        """创建 Agent 时不应该加载模型"""
        agent = InternVL_VLA_Agent(model_path="nonexistent_path", device="cpu")
        assert agent._loaded is False
        assert agent._model is None

    def test_config_stored(self):
        """构造参数应该正确保存"""
        agent = InternVL_VLA_Agent(
            model_path="/test/path",
            action_dim=6,
            freeze_backbone=False,
            device="cpu",
        )
        assert agent.model_path == "/test/path"
        assert agent.action_dim == 6
        assert agent.freeze_backbone is False
        assert agent.target_device == "cpu"


class TestGetLLMHiddenSize:
    """测试从 config 自动获取 hidden_size"""

    def test_llm_config_path(self):
        """优先从 llm_config.hidden_size 读取"""
        agent = _build_mock_agent()
        agent._model.config = SimpleNamespace(
            llm_config=SimpleNamespace(hidden_size=4096)
        )
        assert agent._get_llm_hidden_size() == 4096

    def test_text_config_path(self):
        """如果没有 llm_config，fallback 到 text_config"""
        agent = _build_mock_agent()
        agent._model.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=2048)
        )
        assert agent._get_llm_hidden_size() == 2048

    def test_fallback_default(self):
        """都没有的话，默认返回 4096"""
        agent = _build_mock_agent()
        agent._model.config = SimpleNamespace()
        assert agent._get_llm_hidden_size() == 4096


# ============================================================
# 第三层：真实模型测试（需要 GPU + 模型权重）
# 运行时用: pytest test_vla_agent.py -k "test_real" -v
# ============================================================
REAL_MODEL_PATH = "/workspace/codes/vitsp/InternVL3_5-8B-Flash"
_has_gpu = torch.cuda.is_available()
_has_model = os.path.isdir(REAL_MODEL_PATH)

skip_real = pytest.mark.skipif(
    not (_has_gpu and _has_model),
    reason=f"需要 GPU({_has_gpu}) + 模型权重({_has_model})",
)


@skip_real
class TestRealAgent:
    """用真实模型跑的端到端测试"""

    @pytest.fixture(scope="class")
    def real_agent(self):
        agent = InternVL_VLA_Agent(model_path=REAL_MODEL_PATH, device="cuda")
        agent._lazy_load()
        return agent

    def test_loaded(self, real_agent):
        assert real_agent._loaded is True
        assert real_agent._model is not None

    def test_text_chat(self, real_agent):
        img = _make_dummy_image(512, 512)
        resp = real_agent.text_chat(img, "Describe this image briefly.")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_get_action(self, real_agent):
        img = _make_dummy_image(512, 512)
        coords = real_agent.get_action(img, image_width=1000, image_height=1000)
        assert len(coords) == 4
        x_min, y_min, x_max, y_max = coords
        assert 0 <= x_min <= x_max <= 1000
        assert 0 <= y_min <= y_max <= 1000

    def test_forward(self, real_agent):
        pv = real_agent.preprocess_image(_make_dummy_image())
        ids = real_agent._tokenizer("test", return_tensors="pt").input_ids.cuda()
        out = real_agent.forward(pv, ids)
        assert out.shape == (1, 4)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ============================================================
# 辅助函数
# ============================================================
def _write_fake_png(buf):
    """写一张最小的 fake PNG 到 BytesIO（给 mock plotly 用）"""
    img = Image.new("RGB", (100, 100), (0, 0, 255))
    img.save(buf, format="PNG")
    buf.seek(0)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "not test_real"])
