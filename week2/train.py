import copy
import time
import torch
import matplotlib.pyplot as plt
from model import ResNet18
import torch.nn as nn
import pandas as pd
import os
from data_loader import train_val_data_process

# 1. 禁止系统把内存数据写到C盘（最关键！）
os.environ["TORCH_HOME"] = "D:/torch_temp"  # 改模型缓存到D盘
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_CACHE_DISABLE"] = "1"     # 禁用不必要的磁盘缓存

# 2. 强制 PyTorch 只用物理内存，绝对不读写硬盘虚拟内存
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# 3. 降低内存压力，防止系统被迫使用C盘
import warnings
warnings.filterwarnings('ignore')

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):  # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设定训练所用到的设备，有GPU用GPU没有用CPU

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器（梯度下降法） lr->学习率，model.parameters()->模型参数

    criterion = nn.CrossEntropyLoss()   # 定义损失函数，分类问题一般用交叉熵损失函数，回归问题一般用均方差损失函数

    model = model.to(device)  # 将模型放入到训练设备中

    best_model_wts = copy.deepcopy(model.state_dict())  # 复制当前模型的参数

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()

            with torch.no_grad():  #可以减少内存使用并加速验证过程
                # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                output = model(b_x)
                # 查找每一行中最大值对应的行标
                pre_lab = torch.argmax(output, dim=1)
                # 计算每一个batch的损失函数
                loss = criterion(output, b_y)
            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重
        if val_acc_all[-1] > best_acc:
            # 保存当前的最高精确度
            best_acc = val_acc_all[-1]
            # 保存当前的最高精确度参数
            best_model_wts = copy.deepcopy(model.state_dict())
        # 训练耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use % 60))

    # 选择最优参数
    torch.save(best_model_wts, 'best_model.pth')

    train_process = pd.DataFrame(data={"epoch" : range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.tight_layout()  # 自动调整布局，避免子图重叠
    plt.show()

if __name__ == "__main__":
    data_dir="E:/SignMelody/SignAlphaSet"   #数据集的路径
    train_dataloader,val_dataloader,num_classes=train_val_data_process(data_dir)

    # 将模型实例化
    resnet18 = ResNet18()
    train_process = train_model_process(resnet18, train_dataloader, val_dataloader, 1)
    matplot_acc_loss(train_process)