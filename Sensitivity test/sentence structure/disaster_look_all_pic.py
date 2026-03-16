import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*")
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch
import os
from transformers import ViltProcessor, ViltForQuestionAnswering

# ============ 强化的离线模式设置 ============
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = '/user_homes/chuhai/.cache/huggingface/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/user_homes/chuhai/.cache/huggingface/hub'

# 禁用所有网络连接
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 取消任何代理设置（这可能是问题的根源）
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

print("=== 离线模式设置完成 ===")
print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 创建结果目录
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

# 模型预测函数
def getTQ_albef_feature_extractor_base(filename, model, vis_processors, txt_processors):
    disaster_classes = ["flood", "hail", "dense fog", "strong wind", "mudslide", "heavy snow", "thunder", "tornado", "waterlogging"]
    disaster_descriptions = ['a photo of '+ adisaster +'.' for adisaster in disaster_classes]
    
    raw_image = Image.open(filename).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) 
    similarities = []
    for desc in disaster_descriptions:
        text_input = txt_processors["eval"](desc)
        sample = {"image": image, "text_input": [text_input]}    
        features_image = model.extract_features(sample, mode="image")
        features_text = model.extract_features(sample, mode="text")
        similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
        similarities.append(similarity)
    best_match_idx = torch.argmax(torch.tensor(similarities)).item()
    best_match_class = disaster_classes[best_match_idx]
    return best_match_class

def getTQ_clip_general(filename, model, vis_processors, txt_processors, model_name):
    texts = ["flood", "hail", "dense fog", "strong wind", "mudslide", "heavy snow", "thunder", "tornado", "waterlogging"]
    
    raw_image = Image.open(filename).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    text_descriptions = [f"a photo of {t}." for t in texts]
    
    similarities = []
    for text_desc in text_descriptions:
        text_input = txt_processors["eval"](text_desc)
        sample = {"image": image, "text_input": [text_input]}
        
        with torch.no_grad():
            features = model.extract_features(sample)
            similarity = features.image_embeds_proj @ features.text_embeds_proj.t()
            similarities.append(similarity.item())
    
    best_match_idx = torch.argmax(torch.tensor(similarities)).item()
    return texts[best_match_idx]

def getTQ_albef_vqa_vqav2(filename, model, vis_processors, txt_processors):
    questionStr = "what disaster is this ?"
    
    question = txt_processors["eval"](questionStr)
    raw_image = Image.open(filename).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)  
    # 修改answer_list为 ["flood", "hail", "heavyfog", "heavywind", "mudslide", "snow", "thunder", "tornado", "waterlogging"]
    answer = model.predict_answers(
        samples={"image": image, "text_input": question}, answer_list=["flood", "hail", "dense fog", "strong wind", "mudslide", "heavy snow", "thunder", "tornado", "waterlogging"],
        inference_method="rank"
    )
    return answer[0]

def getTQ_blip_vqa_aokvqa(filename, model, vis_processors, txt_processors):
    questionStr = "what disaster is this ?"
    
    question = txt_processors["eval"](questionStr)
    raw_image = Image.open(filename).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) 
    answer = model.predict_answers({"image": image, "text_input": question}, inference_method="generate")
    return answer[0]

def getTQ_blip_vqa_okvqa(filename, model, vis_processors, txt_processors):
    questionStr = "what disaster is this ?"
    
    question = txt_processors["eval"](questionStr) 
    raw_image = Image.open(filename).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    answer = model.predict_answers({"image": image, "text_input": question}, inference_method="generate")
    return answer[0]

def getTQ_vilt_b32_finetuned_vqa(filename, model, processor):
    questionStr = "what disaster is this ?"
    
    question = questionStr  
    raw_image = Image.open(filename).convert('RGB')
    inputs = processor(
        raw_image, 
        question, 
        return_tensors="pt",  
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    answer = model.config.id2label[predicted_class_idx]
    return answer

def getTQ_blip_vqa_vqav2(filename, model, vis_processors, txt_processors):
    questionStr = "what disaster is this ?"
    
    question = txt_processors["eval"](questionStr)  
    raw_image = Image.open(filename).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)       
    answer = model.predict_answers({"image": image, "text_input": question}, inference_method="generate")
    return answer[0]

def load_all_models():
    """加载所有需要的模型"""
    models = {}
    
    # 1. blip_vqa_vqav2
    print("1. 加载 blip_vqa_vqav2...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa", 
        model_type="vqav2", 
        is_eval=True, device=device
    )
    print(f"   模型设备: {next(model.parameters()).device}")
    models[1] = {
        'name': 'blip_vqa_vqav2',
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'type': 'vqa',
        'prompt_type': 'question'
    }
    
    # 2. vilt_b32_finetuned_vqa
    print("2. 加载 vilt_b32_finetuned_vqa...")
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = model.to(device)
    print(f"   模型设备: {next(model.parameters()).device}")
    models[2] = {
        'name': 'vilt_b32_finetuned_vqa',
        'model': model,
        'processor': processor,
        'type': 'vqa',
        'prompt_type': 'question'
    }
    
    # 3. blip_vqa_okvqa
    print("3. 加载 blip_vqa_okvqa...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa",  
        model_type="okvqa", 
        is_eval=True, device=device
    )
    print(f"   模型设备: {next(model.parameters()).device}")
    models[3] = {
        'name': 'blip_vqa_okvqa',
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'type': 'vqa',
        'prompt_type': 'question'
    }
    
    # 4. blip_vqa_aokvqa
    print("4. 加载 blip_vqa_aokvqa...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa", 
        model_type="aokvqa",  
        is_eval=True, device=device
    )
    print(f"   模型设备: {next(model.parameters()).device}")
    models[4] = {
        'name': 'blip_vqa_aokvqa',
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'type': 'vqa',
        'prompt_type': 'question'
    }
    
    # 5. albef_vqa_vqav2
    print("5. 加载 albef_vqa_vqav2...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="albef_vqa",  
        model_type="vqav2",  
        is_eval=True, device=device
    )
    print(f"   模型设备: {next(model.parameters()).device}")
    models[5] = {
        'name': 'albef_vqa_vqav2',
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'type': 'vqa',
        'prompt_type': 'question'
    }
    
    # 6. clip_ViT_B_32
    print("6. 加载 clip_ViT_B_32...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="clip",
        model_type="ViT-B-32",
        is_eval=True,
        device=device
    )
    print(f"   模型设备: {next(model.parameters()).device}")
    models[6] = {
        'name': 'clip_ViT_B_32',
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'type': 'feature',
        'prompt_type': 'text_list'
    }
    
    # 7. clip_ViT_B_16
    print("7. 加载 clip_ViT_B_16...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="clip",
        model_type="ViT-B-16",
        is_eval=True,
        device=device
    )
    print(f"   模型设备: {next(model.parameters()).device}")
    models[7] = {
        'name': 'clip_ViT_B_16',
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'type': 'feature',
        'prompt_type': 'text_list'
    }
    
    # 8. clip_ViT_L_14
    print("8. 加载 clip_ViT_L_14...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="clip",
        model_type="ViT-L-14",
        is_eval=True,
        device=device
    )
    print(f"   模型设备: {next(model.parameters()).device}")
    models[8] = {
        'name': 'clip_ViT_L_14',
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'type': 'feature',
        'prompt_type': 'text_list'
    }
    
    # 9. albef_feature_extractor_base
    print("9. 加载 albef_feature_extractor_base...")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="albef_feature_extractor",
        model_type="base",
        is_eval=True,
        device=device
    )
    print(f"   模型设备: {next(model.parameters()).device}")
    models[9] = {
        'name': 'albef_feature_extractor_base',
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'type': 'feature',
        'prompt_type': 'text_list'
    }
    return models

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

def predict_disaster_for_image(image_path, models_dict, result_files):
    """对单张图片进行所有模型的天气预测，并将结果写入文件"""
    real_answer = os.path.basename(image_path).split('_')[0]
    
    # 处理所有模型
    for model_id, model_info in models_dict.items():
        try:
            model_name = model_info['name']
            file_handle = result_files[model_id]
            
            # 根据模型类型调用相应的预测函数
            if model_id == 1:
                raw_result = getTQ_blip_vqa_vqav2(image_path, model_info['model'], 
                                                  model_info['vis_processors'], model_info['txt_processors'])
            elif model_id == 2:
                raw_result = getTQ_vilt_b32_finetuned_vqa(image_path, model_info['model'], 
                                                          model_info['processor'])
            elif model_id == 3:
                raw_result = getTQ_blip_vqa_okvqa(image_path, model_info['model'], 
                                                  model_info['vis_processors'], model_info['txt_processors'])
            elif model_id == 4:
                raw_result = getTQ_blip_vqa_aokvqa(image_path, model_info['model'], 
                                                   model_info['vis_processors'], model_info['txt_processors'])
            elif model_id == 5:
                raw_result = getTQ_albef_vqa_vqav2(image_path, model_info['model'], 
                                                   model_info['vis_processors'], model_info['txt_processors'])
            elif model_id in [6, 7, 8]:
                raw_result = getTQ_clip_general(image_path, model_info['model'], 
                                              model_info['vis_processors'], model_info['txt_processors'], model_name)
            elif model_id == 9:
                raw_result = getTQ_albef_feature_extractor_base(image_path, model_info['model'], 
                                                              model_info['vis_processors'], model_info['txt_processors'])
            
            # 写入结果到文件
            file_handle.write(f"{image_path},{real_answer},{raw_result}\n")
            file_handle.flush()  # 立即写入，防止数据丢失
            
        except Exception as e:
            print(f"模型 {model_info['name']} 预测图片 {image_path} 时出错: {e}")
            # 出错时写入错误信息
            result_files[model_id].write(f"{image_path},{real_answer},error\n")
            result_files[model_id].flush()

def essemable(image_paths):
    """主函数：加载模型并对所有图片进行预测"""
    # 加载所有模型
    models_dict = load_all_models()
    
    # 为每个模型创建结果文件
    result_files = {}
    for model_id, model_info in models_dict.items():
        filename = os.path.join(result_dir, f"{model_info['name']}.txt")
        result_files[model_id] = open(filename, 'w', encoding='utf-8')
        # 写入文件头（可选）
        result_files[model_id].write("# image_path,real_answer,predicted_result\n")
    
    print(f"开始处理 {len(image_paths)} 张图片...")
    
    # 逐张图片处理
    for i, image_path in enumerate(image_paths):
        if i % 100 == 0:  # 每100张图片打印一次进度
            print(f"处理进度: {i}/{len(image_paths)} ({i/len(image_paths)*100:.1f}%)")
        
        try:
            predict_disaster_for_image(image_path, models_dict, result_files)
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")
            # 对所有模型记录错误
            for model_id in models_dict:
                result_files[model_id].write(f"{image_path},error,error\n")
                result_files[model_id].flush()
    
    # 关闭所有文件
    for file_handle in result_files.values():
        file_handle.close()
    
    print("所有图片处理完成！")

if __name__ == '__main__':
    # 获取所有图片路径
    image_root = '../data/'  # 修改为您的图片根目录
    all_image_paths = get_all_image_paths(image_root)
    
    if not all_image_paths:
        print("未找到任何图片文件，请检查路径设置")
    else:
        # 开始处理
        essemable(all_image_paths)