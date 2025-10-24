"""
当保存和加载模型时，需要熟悉三个核心功能：

torch.save：将序列化对象保存到磁盘。此函数使用Python的pickle模块进行序列化。使用此函数可以保存如模型、tensor、字典等各种对象。
torch.load：使用pickle的unpickling功能将pickle对象文件反序列化到内存。此功能还可以有助于设备加载数据。
torch.nn.Module.load_state_dict：使用反序列化函数 state_dict 来加载模型的参数字典。
"""


"""part 1 : 什么是状态字典？state_dict 是什么？"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path
import torch.nn.functional as F  # 添加F导入，因为forward方法中使用了F.relu

# 在 PyTorch 中最常见的模型保存使用‘.pt’或者是‘.pth’作为模型文件扩展名。
save_path = Path(__file__).parent / "model_save" / "model.pth"
save_model_path = Path(__file__).parent / "model_save" / "model_total.pth"
checkpoint_path = Path(__file__).parent / "model_save" / "checkpoint.pth"
# 定义模型
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        # 卷积层1：输入通道3（RGB图像），输出通道6，卷积核大小5x5
        # 正向传播：提取图像的低级特征（边缘、纹理等）
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 最大池化层：窗口大小2x2，步长2
        # 正向传播：降低特征图尺寸，保留最显著特征，增强平移不变性
        # 反向传播：梯度只传递给最大值位置，其他位置梯度为0
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2：输入通道6，输出通道16，卷积核大小5x5
        # 正向传播：提取更高级的特征组合
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层1：将卷积特征展平后连接到120个神经元
        # 正向传播：进行特征组合和抽象
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2：120个神经元到84个神经元
        # 正向传播：进一步特征抽象
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3：84个神经元到10个输出（假设是10分类问题）
        # 正向传播：输出最终的分类概率
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 正向传播：数据从输入层逐层传递到输出层
        # 卷积层1 + ReLU激活 + 池化层
        # ReLU激活函数：f(x)=max(0,x)，解决梯度消失问题，计算简单
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积层2 + ReLU激活 + 池化层
        x = self.pool(F.relu(self.conv2(x)))
        # 展平操作：将多维特征图转换为一维向量，为全连接层做准备
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层1 + ReLU激活
        x = F.relu(self.fc1(x))
        # 全连接层2 + ReLU激活
        x = F.relu(self.fc2(x))
        # 输出层：不激活，输出原始logits（后续配合交叉熵损失函数）
        x = self.fc3(x)
        return x


# 初始化模型
model = TheModelClass()
# 打印模型的参数：查看模型的所有可学习参数（权重和偏置）
print("Model's parameters:")
for param in model.parameters():
    print(param)


# 初始化优化器：使用随机梯度下降（SGD）
# lr=0.001：学习率，控制参数更新步长
# momentum=0.9：动量，加速收敛并减少震荡
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练循环：完整的深度学习训练流程
# epoch：遍历整个数据集的次数
for epoch in range(2):
    running_loss = 0.0
    # 遍历训练数据加载器
    for i, data in enumerate(trainloader, 0):
        # 1. 获取输入数据和标签
        inputs, labels = data
        
        # 2. 梯度清零：清空上一轮计算的梯度，避免梯度累积
        optimizer.zero_grad()
        
        # 3. 正向传播：模型根据输入数据做出预测
        # 解决"模型如何做预测"的问题
        outputs = model(inputs)
        
        # 4. 计算损失：衡量预测与真实标签的差距
        # 解决"预测有多不准"的问题
        loss = criterion(outputs, labels)
        
        # 5. 反向传播：计算损失对每个参数的梯度
        # 解决"如何从错误中学习"的问题
        # 使用链式法则自动计算所有参数的偏导数
        loss.backward()
        
        # 6. 参数更新：根据梯度调整模型参数
        # 执行 w = w - lr * ∂loss/∂w
        optimizer.step()
        
        # 累计损失用于监控训练进度
        running_loss += loss.item()



# 打印模型的状态字典：包含所有可学习参数的当前值
# state_dict是PyTorch保存和加载模型的核心数据结构
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的状态字典：包含优化器的状态信息
# 如动量缓存、学习率调度状态等
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])




"""part2 : 保存和加载模型"""

# 保存模型参数（只保存state_dict，不保存模型结构）
# 这种方式更灵活，可以在不同代码结构中加载参数
# os.makedirs(save_path, exist_ok=True)  
torch.save(model.state_dict(), save_path)


# 加载模型参数
# 需要先创建相同结构的模型实例
# model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(save_path))
# 在运行推理之前，务必调用model.eval()设置dropout和batch normalization层为评估模式
# 评估模式会固定dropout和BN的统计量，确保推理结果的一致性
model.eval()

# 保存和加载整个模型（包括模型结构和参数）
# 这种方式简单但不够灵活，模型类定义必须在当前作用域可用
torch.save(model, save_model_path)
# 模型类必须在此之前被定义
model = torch.load(save_model_path)
model.eval()


""" part3: 保存和加载 Checkpoint 用于推理/继续训练"""

""" 
# 保存
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)


# 加载
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()

"""




"""part4: 在一个文件中保存多个模型 """

""" 
# 保存
torch.save({
            'epoch': epoch,
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            'loss': loss,
            }, save_path) 

# 加载
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(save_path)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()

"""



"""part5 : 使用不同模型参数下的热启动模式"""

"""

torch.save(modelA.state_dict(), PATH)

modelB = TheModelBClass(*args, **kwargs)
#在迁移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见的情况。 
#都可以通过在load_state_dict()函数中将strict参数设置为 False 来忽略非匹配键的函数。
#如果要将参数从一个层加载到另一个层，但是某些键不匹配，主要修改正在加载的 state_dict 中的参数键的名称以匹配要在加载到模型中的键即可。s
modelB.load_state_dict(torch.load(PATH), strict=False)


"""



""" part6: 通过设备保存/加载模型 """
"""

#保存在CPU上, 加载在CPU上
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))


# 保存在CPU上, 加载在GPU上
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# 确保在你提供给模型的任何输入张量上调用input = input.to(device)


"""