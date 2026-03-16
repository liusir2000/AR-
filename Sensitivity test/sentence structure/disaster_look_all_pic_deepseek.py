import os
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

def get_all_image_paths(root_dir):
    """递归获取所有图片文件路径"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_paths = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    
    print(f"找到 {len(image_paths)} 张图片")
    return image_paths

def deepseek_look(road):
    # 构建对话（包含图像输入）
    conversation = [
        {
            "role": "用户",
            "content": "<image_placeholder>这是什么灾害？从洪水、冰雹、大雾、大风、泥石流、大雪、闪电、龙卷、内涝中选择一个最符合的答案。",
            "images": [road]  # 图像文件路径
        },
        {
            "role": "助手",
            "content": ""  # 预留回复位置
        }
    ]
    # 加载图像并准备输入数据
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # 生成图像嵌入
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # 模型推理生成回复
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,  # 最大生成token数
        do_sample=False,  # 关闭采样（确定性生成）
        use_cache=True
    )

    # 解码并输出结果
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

# 指定模型路径
model_path = "/user_homes/chuhai/.cache/modelscope/hub/models/deepseek-ai/deepseek-vl-7b-chat/"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# 加载模型并设置运行设备
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

image_root = '/user_homes/chuhai/ldw2/hwx/disaster/data/'
result_file = open('/user_homes/chuhai/ldw2/hwx/disaster/prompt1/result/deepseek_vl.txt', 'w', encoding='utf-8')
# 写入文件头（可选）
result_file.write("# image_path,real_answer,predicted_result\n")

all_image_paths = get_all_image_paths(image_root)
for i, image_path in enumerate(all_image_paths):
    if i % 100 == 0:  # 每100张图片打印一次进度
        print(f"处理进度: {i}/{len(all_image_paths)} ({i/len(all_image_paths)*100:.1f}%)")
    raw_result = deepseek_look(image_path)
    real_answer = os.path.basename(image_path).split('_')[0]
    result_file.write(f"{image_path},{real_answer},{raw_result}\n")
    result_file.flush()  # 立即写入，防止数据丢失
# 关闭所有文件
result_file.close()

print('所有图片已完成！')