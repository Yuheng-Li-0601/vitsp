import time
import base64
import io
from PIL import Image
from openai import OpenAI
import numpy as np

# 你的配置参数
API_KEY = 'sk-VsV9Kt41SBXzRymbW8Jma1m11jgNr2RNu5ZcrI9r0sosmTMZ'
BASE_URL = "https://chat.intern-ai.org.cn/api/v1/"
MODEL_NAME = "internvl3.5-latest" # 测试 Fast Thinking 模型

def generate_dummy_base64_image():
    """生成一张测试用的随机彩色图片（模拟 TSP 散点图的尺寸）"""
    # 假设你的 TSP 图分辨率是 1000x1000
    img_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    buf = io.BytesIO()
    # 模拟你代码里的 JPEG 保存格式
    img.save(buf, format='JPEG', quality=85) 
    img_bytes = buf.getvalue()
    
    # 打印图片大小，这非常关键！
    size_kb = len(img_bytes) / 1024
    print(f"-> [网络指标] 准备上传的图片大小: {size_kb:.2f} KB")
    
    return base64.b64encode(img_bytes).decode('utf-8')

def test_vlm_latency():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    base64_image = generate_dummy_base64_image()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please return 2 non-overlapping sub-rectangle coordinates in this map. Keep it very brief."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    print("\n发起 API 请求，开始计时...")
    t0 = time.time()
    
    try:
        # 开启 stream=True，这是测速的核心！
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True 
        )
        
        first_token_time = None
        full_content = ""
        token_count = 0
        
        for chunk in response:
            # 记录收到第一个字符的时间
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - t0
                print(f"-> [耗时拆解 1] 首字返回时间 (TTFT): {ttft:.2f} 秒 (包含网络上传图 + VLM 读图计算)")
                print("-> 开始流式接收内容: ", end="", flush=True)

            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content
                token_count += 1
                
        t_end = time.time()
        print("\n\n" + "="*40)
        
        # 计算生成指标
        generation_time = t_end - first_token_time
        total_time = t_end - t0
        tps = token_count / generation_time if generation_time > 0 else 0
        
        print(f"[耗时拆解 2] 纯生成时间: {generation_time:.2f} 秒")
        print(f"[耗时拆解 3] 生成速度: ~{tps:.1f} tokens/秒")
        print(f"[总结] 总 API 耗时: {total_time:.2f} 秒")

    except Exception as e:
        print(f"\nAPI 调用失败: {e}")

if __name__ == "__main__":
    test_vlm_latency()