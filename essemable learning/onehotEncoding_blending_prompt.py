import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Creating a tensor from a list of numpy.ndarrays.*')

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Training on CPU.")
    device = torch.device("cpu")
    
    
# # 假设你已经有了一个模型 model
# model = YourModel()
 
# # 将模型移至GPU
# model.to(device)
 
# # 如果你有数据，例如输入数据 input_data 和目标数据 target_data
# input_data = torch.randn(10, 3, 224, 224)  # 示例数据，需要根据实际情况调整
# target_data = torch.randint(0, 10, (10,))  # 示例标签，需要根据实际情况调整
 
# # 将数据移至GPU
# input_data, target_data = input_data.to(device), target_data.to(device)


def convert_to_onehot(codes):
    """
    将天气类别编码转换为onehot向量，并拼接成9×4的二维向量。

    参数：
    codes (str): 长度为9的字符串，每个字符是0至3之间的数字。

    返回：
    numpy.ndarray: 9×4的二维onehot向量。
    """
    codes = '%010d' % int(codes)
    if len(codes) != 10:
        raise ValueError("输入编码的长度必须为10！")

    # 确保所有字符是数字且在0到3的范围内
    if not all(char.isdigit() and 0 <= int(char) <= 9 for char in codes):
        raise ValueError("编码必须是0到9之间的数字字符串！")

    # 将字符串转换为整数列表
    # codes1= codes[0:2]+codes[3]+codes[5:]
    codes1= codes[0:10]
    codes = [int(char) for char in codes1]

    # 使用NumPy进行onehot编码
    onehot_matrix = np.eye(10)[codes]
    return onehot_matrix


def convert_label_to_onehot(label):
    """
    将单个标签转换为onehot向量。

    参数：
    label (int): 标签值，范围在0-3之间。

    返回：
    numpy.ndarray: 1×4的onehot向量。
    """
    label = int(label)
    if label < 0 or label > 9:
        raise ValueError("标签值必须在1到9之间！")
    return np.eye(9)[label-1]#去除占位符


def load_data(filepath):
    """
    从文件中加载数据，提取编码和真实值。

    参数：
    filepath (str): 文件路径。

    返回：
    list: 编码的onehot向量。
    list: 真实值的onehot向量。
    """
    inputs = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            code, label = line.strip().split(',')
            onehot_vector = convert_to_onehot(code)
            onehot_label = convert_label_to_onehot(label)
            inputs.append(onehot_vector)
            labels.append(onehot_label)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# 定义神经网络
class DisasterNet(nn.Module):
    def __init__(self):
        super(DisasterNet, self).__init__()
        #self.fc1 = nn.Linear(36, 128)  # 输入层（9×4展开为36）
        self.fc1 = nn.Linear(100, 128)  # 输入层（9×5展开为36）
        self.fc2 = nn.Linear(128, 256)  # 第一个隐藏层
        self.fc3 = nn.Linear(256, 128)  # 第二个隐藏层
        self.fc4 = nn.Linear(128, 9)  # 输出层，不考虑占位符0，因此输出为9位

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

os.makedirs('./model_blending',exist_ok=True)
# 加载数据
train_file = "train_set_v4.0.csv"
inputs, labels = load_data(train_file)
inputs, labels = inputs.to(device), labels.to(device)

# 初始化模型、损失函数和优化器
model = DisasterNet()
model.to(device)
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print("训练完成！")

# 保存模型
torch.save(model.state_dict(), "./model_blending/disaster_blending_v4.0.pth")
print("模型已保存！")

# 测试模型
test_file = "test_set_v4.0.csv"
test_inputs, test_labels = load_data(test_file)
test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    test_outputs = model(test_inputs)
    predicted = torch.argmax(test_outputs, dim=1).cpu().numpy()
    actual = torch.argmax(test_labels, dim=1).cpu().numpy()
    
    #修正去除占位符影响造成的改动
    predicted = predicted + 1
    actual = actual + 1   

    accuracy = (predicted == actual).sum() / actual.shape[0]
    precision = precision_score(actual, predicted, average='macro')
    recall = recall_score(actual, predicted, average='macro')
    f1 = f1_score(actual, predicted, average='macro')

    print(f"测试准确率: {accuracy * 100:.2f}%")
    print(f"查准率: {precision:.4f}")
    print(f"查全率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

# 记录结果到result.txt
with open("./model_blending/result_final_v4.0.txt", "w") as f:
    f.write("编码 预测值 真实值\n")
    for i, (p, a) in enumerate(zip(predicted, actual)):
        f.write(f"{i} {p} {a}\n")
print("测试结果已保存")
