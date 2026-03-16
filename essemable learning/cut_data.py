import os
import csv
import random

def split_dataset_by_ground_truth():
    """根据真实标签分割数据集为训练集和测试集"""
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 输入文件路径
    input_file = os.path.join('model_data', 'model10_v2.0_encoding.csv')
    
    # 输出文件路径
    train_file = os.path.join('model_data', 'train_set_v4.0.csv')
    test_file = os.path.join('model_data', 'test_set_v4.0.csv')
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 按真实标签分组存储数据
    data_by_label = {}
    
    # 读取数据并分组
    print(f"正在读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        line_count = 0
        
        for row in csv_reader:
            if len(row) == 2:
                predictions, ground_truth = row
                
                # 验证数据格式
                if len(predictions) == 10 and ground_truth.isdigit():
                    label = int(ground_truth)
                    
                    # 只处理1-9的标签
                    if 1 <= label <= 9:
                        if label not in data_by_label:
                            data_by_label[label] = []
                        
                        data_by_label[label].append(row)
                        line_count += 1
                else:
                    print(f"警告: 第{line_count+1}行数据格式不正确: {row}")
    
    print(f"共读取 {line_count} 行数据")
    
    # 统计每个标签的数据量
    print("\n各标签数据统计:")
    for label in sorted(data_by_label.keys()):
        count = len(data_by_label[label])
        print(f"标签 {label}: {count} 个样本")
    
    # 分割数据集
    train_data = []
    test_data = []
    
    for label, samples in data_by_label.items():
        # 随机打乱当前标签的数据
        random.shuffle(samples)
        
        # 计算分割点
        split_idx = int(len(samples) * 0.7)
        
        # 分割数据
        label_train = samples[:split_idx]
        label_test = samples[split_idx:]
        
        # 添加到总数据集
        train_data.extend(label_train)
        test_data.extend(label_test)
        
        print(f"标签 {label}: 训练集 {len(label_train)} 个样本, 测试集 {len(label_test)} 个样本")
    
    # 再次打乱训练集和测试集（可选，避免同一标签的数据连续出现）
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # 保存训练集
    print(f"\n正在保存训练集: {train_file}")
    with open(train_file, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(train_data)
    
    # 保存测试集
    print(f"正在保存测试集: {test_file}")
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(test_data)
    
    print(f"训练集已保存: {train_file} ({len(train_data)} 个样本)")
    print(f"测试集已保存: {test_file} ({len(test_data)} 个样本)")
    
    # 统计最终数据集分布
    print("\n最终数据集分布:")
    print(f"总样本数: {len(train_data) + len(test_data)}")
    print(f"训练集比例: {len(train_data) / (len(train_data) + len(test_data)):.2%}")
    print(f"测试集比例: {len(test_data) / (len(train_data) + len(test_data)):.2%}")

def main():
    """主函数"""
    print("开始分割数据集...")
    print("=" * 50)
    
    # 创建result文件夹（如果不存在）
    if not os.path.exists('result'):
        os.makedirs('result')
        print("创建了result文件夹")
    
    # 分割数据集
    split_dataset_by_ground_truth()
    
    print("=" * 50)
    print("数据集分割完成!")

if __name__ == "__main__":
    main()