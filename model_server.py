import os
import base64
import io
from typing import Optional
from PIL import Image

# 1. 环境变量配置（必须在 import transformers 之前）
base_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(base_dir, ".hf_cache")
os.makedirs(cache_dir, exist_ok=True)

os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_MODULES_CACHE'] = os.path.join(cache_dir, "modules")
os.environ['XDG_CACHE_HOME'] = cache_dir

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import uvicorn

app = FastAPI()

# 2. 图像预处理函数 (从 llm.py 移植)
def build_transform(input_size=448):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

# 初始化转换器
img_transform = build_transform(input_size=448)

# 3. 加载模型
MODEL_PATH = "./InternVL3_5-8B-Flash"
print(f"正在加载模型: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True, 
    fix_mistral_regex=True
)

model = AutoModel.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True, 
    device_map="auto",
    low_cpu_mem_usage=True
).eval()

# 4. 定义请求结构
class PromptRequest(BaseModel):
    prompt: str
    image_base64: Optional[str] = None  # 可选的 Base64 图像

@app.post("/generate")
async def generate(request: PromptRequest):
    try:
        pixel_values = None
        
        # 如果请求中包含图像
        if request.image_base64:
            # 解码 Base64 字符串
            img_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # 转换为模型所需的 Tensor
            pixel_values = img_transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
            
            # 注意：InternVL 的 Prompt 通常需要以 <image>\n 开头
            if "<image>" not in request.prompt:
                request.prompt = f"<image>\n{request.prompt}"

        # 调用模型推理
        generation_config = {"max_new_tokens": 1024, "do_sample": False}
        response = model.chat(tokenizer, pixel_values, request.prompt, generation_config)
        
        return {"response": response}
        
    except Exception as e:
        print(f"推理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)