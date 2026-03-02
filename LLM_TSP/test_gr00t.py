import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# 1. 指定你刚刚下载的本地文件夹路径
groot_model_path = "./GR00T-N1.6-3B"

print("正在将 GR00T 加载到 96GB Blackwell 显存中...")

# 2. 原生加载模型 (注意：trust_remote_code=True 是必须的，因为 VLA 通常有自定义的 Action Head)
model = AutoModelForCausalLM.from_pretrained(
    groot_model_path,
    torch_dtype=torch.bfloat16, # 强烈建议用 bfloat16，Blackwell 对此支持极佳，省一半显存
    trust_remote_code=True,
    device_map="cuda" # 自动扔到你的 GPU 上
)

# 3. 加载对应的处理器 (用于把图片和 Prompt 转成 Tensor)
processor = AutoProcessor.from_pretrained(
    groot_model_path, 
    trust_remote_code=True
)

print(f"GR00T 加载成功！当前占用显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")