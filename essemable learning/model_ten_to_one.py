import os

# 创建result文件夹
if not os.path.exists('model_data'):
    os.makedirs('model_data')

# 定义真实情况的替换规则
real_replacement_rules = {
    'strongwind': 'strong wind',
    'landwind': 'strong wind',
    'seawind': 'strong wind', 
    'winddisaster': 'strong wind',
    'densefog': 'dense fog',
    'heavysnow': 'heavy snow'
}

# 定义预测情况的分类映射
classification_mapping = {
    'waterlogging': ['rain', 'rain storm', 'rainbow', 'salt water', 'sink', 'sink drain', 'sink faucet', 'sink full', 'water', 'waterlogging'],
    'flood': ['flood', 'flooding', 'drowning'],
    'mudslide': ['earthquake', 'drought', 'mudslide'],
    'thunder': ['fire', 'fast', 'forest fire','light', 'lightning', 'lightning storm', 'lightning strike', 'thunder', 'thunderstorm'],
    'hail': ['fall', 'ice', 'bird droppings', 'crystal sink', 'golfing', 'ice cubes', 'hail'],
    'tornado': ['tornado'],
    'strong wind': ['accident', 'big wave', 'crashing waves', 'dandelion', 'dead', 'hurricane', 'kite', 'kite flying', 'shipwreck', 'storm', 'surfing', 'tropical', 'tropical storm', 'tsunami', 'wind', 'wave', 'waves', 'strong wind', 'strongwind'],
    'dense fog': ['air pollution', 'fog', 'pollution', 'smog', 'foggy', 'steam', 'dense fog', 'densefog', 'haze'],
    'heavy snow': ['avalanche', 'blizzard', 'christmas', 'freeze', 'frozen lake', 'hypothermia', 'ice cream', "it's snowing", 'skiing', 'ski accident', 'snow', 'snow storm', 'snowball', 'snowstorm', 'winter', 'heavy snow', 'heavysnow']
}

# 定义所有可能的天气条件（不包括unknown）
weather_conditions = [
    'waterlogging', 'flood', 'mudslide', 'thunder', 'hail', 
    'tornado', 'strong wind', 'dense fog', 'heavy snow'
]

# 创建天气条件到数字的映射字典（0代表unknown）
weather_to_num = {'unknown': 0}
for i, condition in enumerate(weather_conditions, 1):
    weather_to_num[condition] = i

# 创建反向映射字典，用于快速查找预测结果的分类
reverse_mapping = {}
for category, items in classification_mapping.items():
    for item in items:
        reverse_mapping[item] = category

# 定义模型文件名列表
model_files = [
    'albef_feature_extractor_base.txt',
    'albef_vqa_vqav2.txt',
    'blip_vqa_aokvqa.txt',
    'blip_vqa_okvqa.txt',
    'blip_vqa_vqav2.txt',
    'clip_ViT_B_16.txt',
    'clip_ViT_B_32.txt',
    'clip_ViT_L_14.txt',
    'vilt_b32_finetuned_vqa.txt',
    'deepseek-vl-7b-chat.txt'
]

def process_file(file_path):
    """处理单个文件，返回该模型的预测结果数字列表和真实结果数字列表"""
    pred_nums = []
    real_nums = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过标题行
        next(f)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 分割行数据
            parts = line.split(',')
            if len(parts) < 3:
                continue
            
            real_answer, predicted_result = parts[1], parts[2]
            
            # 处理真实情况的替换
            if real_answer in real_replacement_rules:
                real_answer = real_replacement_rules[real_answer]
            
            # 将真实结果转换为数字
            real_num = weather_to_num.get(real_answer, 0)
            
            # 处理预测结果的替换和分类
            predicted_result = predicted_result.strip()
            if predicted_result in reverse_mapping:
                predicted_result = reverse_mapping[predicted_result]
            else:
                predicted_result = 'unknown'
            
            # 将预测结果转换为数字
            pred_num = weather_to_num.get(predicted_result, 0)
            
            pred_nums.append(pred_num)
            real_nums.append(real_num)
    
    return pred_nums, real_nums

def main():
    # 存储所有模型的预测结果
    all_predictions = []
    # 存储真实结果（只存储一次）
    ground_truth = None
    
    # 处理每个模型文件
    for model_file in model_files:
        file_path = os.path.join('model_result', model_file)
        
        if os.path.exists(file_path):
            print(f"正在处理: {model_file}")
            pred_nums, real_nums = process_file(file_path)
            
            # 如果是第一个模型，保存真实结果
            if ground_truth is None:
                ground_truth = real_nums
            
            # 添加到所有预测结果中
            all_predictions.append(pred_nums)
        else:
            print(f"文件不存在: {file_path}")
            # 如果文件不存在，用0填充
            if all_predictions:
                all_predictions.append([0] * len(all_predictions[0]))
            else:
                # 如果还没有任何数据，无法确定长度，暂时跳过
                continue
    
    # 确保所有列表长度相同
    if not all_predictions or not ground_truth:
        print("没有有效数据")
        return
    
    # 检查数据一致性
    num_samples = len(ground_truth)
    for i, pred_list in enumerate(all_predictions):
        if len(pred_list) != num_samples:
            print(f"警告: 模型{i}的预测结果数量({len(pred_list)})与真实结果数量({num_samples})不匹配")
            # 截断或填充到相同长度
            if len(pred_list) > num_samples:
                all_predictions[i] = pred_list[:num_samples]
            else:
                all_predictions[i] = pred_list + [0] * (num_samples - len(pred_list))
    
    # 组合预测结果
    combined_results = []
    for i in range(num_samples):
        # 组合9个模型的预测结果
        combined_pred = ''
        for model_idx in range(len(model_files)):
            if model_idx < len(all_predictions):
                combined_pred += str(all_predictions[model_idx][i])
            else:
                combined_pred += '0'  # 用0填充缺失的模型
        
        # 添加真实结果
        combined_results.append(f"{combined_pred},{ground_truth[i]}")
    
    # 保存结果到CSV文件
    output_file = os.path.join('model_data', 'model10_v2.0_encoding.csv')
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in combined_results:
            f.write(line + '\n')
    
    print(f"结果已保存到: {output_file}")
    print(f"共处理了 {num_samples} 个样本")

if __name__ == "__main__":
    main()