import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

class TSP_GR00T_Agent(nn.Module):
    def __init__(self, base_vlm_path="OpenGVLab/InternVL3_5-8B-Flash", action_dim=4):
        super().__init__()
        
        # 1. 视觉感知流 (Vision Encoder) - 相当于 GR00T 的眼睛
        # 实际开发中，这里可以加载你下载的 InternVL 的视觉模块
        print("Loading Vision Encoder...")
        self.vision_encoder = AutoModel.from_pretrained(base_vlm_path, subfolder="vision_model", trust_remote_code=True)
        # 冻结视觉特征提取器，节省显存
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # 2. 模态对齐投影层 (Projector)
        vision_hidden_size = self.vision_encoder.config.hidden_size
        llm_hidden_size = 4096 # 假设你的 LLM 隐藏层维度
        self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        
        # 3. 决策大脑 (LLM Backbone) - 相当于 GR00T 处理状态序列的 Transformer
        print("Loading LLM Backbone...")
        self.llm = AutoModelForCausalLM.from_pretrained(base_vlm_path, trust_remote_code=True)
        # 可以使用 LoRA 技术只微调 LLM 的部分参数
        
        # 4. 核心创新：动作流 Action Head (完美复刻 GR00T)
        # 我们不输出文本，而是直接输出 4 个连续维度的坐标 [x_min, y_min, x_max, y_max]
        self.action_head = nn.Sequential(
            nn.Linear(llm_hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
            nn.Sigmoid() # 关键！用 Sigmoid 将坐标归一化到 0.0 到 1.0 之间
        )

    def forward(self, image_tensor, state_prompt_ids):
        """
        前向传播：融合视觉与状态信息，输出连续动作
        """
        # 提取视觉特征
        vision_embeds = self.vision_encoder(image_tensor).last_hidden_state
        vision_embeds = self.projector(vision_embeds)
        
        # 提取文本状态特征 (比如当前的阶段信息)
        text_embeds = self.llm.get_input_embeddings()(state_prompt_ids)
        
        # 拼接视觉和文本的 Embedding (或者按照 GR00T 的交叉注意力机制)
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        
        # 送入 LLM 主干网络提取深层决策特征
        outputs = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # 取最后一个 Token 的特征作为全局表示，喂给 Action Head
        global_feature = last_hidden_state[:, -1, :]
        
        # 输出连续的 4D 动作向量 (值域在 0-1 之间)
        normalized_coords = self.action_head(global_feature)
        
        return normalized_coords

    def get_action_for_vitsp(self, image_tensor, image_width=512, image_height=512):
        """
        这个函数用于直接替换你 llm_tsp_async.py 里调用 API 的部分
        """
        self.eval()
        with torch.no_grad():
            # 伪造一个简单的 state prompt
            dummy_prompt = torch.tensor([[1, 2, 3]]).to(image_tensor.device) 
            
            # 获取 0-1 之间的归一化坐标: [batch, 4] -> [x1, y1, x2, y2]
            norm_coords = self(image_tensor, dummy_prompt)[0]
            
            # 将归一化坐标还原为真实像素坐标
            x1 = int(norm_coords[0].item() * image_width)
            y1 = int(norm_coords[1].item() * image_height)
            x2 = int(norm_coords[2].item() * image_width)
            y2 = int(norm_coords[3].item() * image_height)
            
            # 确保 x1 < x2 且 y1 < y2
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            return (x_min, y_min, x_max, y_max)