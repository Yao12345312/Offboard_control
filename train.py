import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from model import MobileNetV2
import os
import json
import torchvision.models.mobilenet
import matplotlib.pyplot as plt
import numpy as np


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

data_transform = {
    "train" : transforms.Compose([transforms.RandomResizedCrop(224),   # 随机裁剪
                                  transforms.RandomHorizontalFlip(),   # 随机翻转
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val" : transforms.Compose([transforms.Resize(256),      # 长宽比不变，最小边长缩放到256
                                transforms.CenterCrop(224),  # 中心裁剪到 224x224
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 获取数据集所在的根目录
data_root = os.path.abspath(os.getcwd())
# 获取动物数据集路径
image_path = data_root + "/data_set/"

# 加载数据集
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])

# 获取训练集图像数量
train_num = len(train_dataset)

# 获取分类的名称
animal_list = train_dataset.class_to_idx
print(f"数据集类别: {animal_list}")

# 采用遍历方法，将分类名称的key与value反过来
cla_dict = dict((val, key) for key, val in animal_list.items())

# 将字典cla_dict编码为json格式
json_str = json.dumps(cla_dict, indent=4)
with open("class_indices.json", "w") as json_file:
    json_file.write(json_str)

batch_size = 16
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = DataLoader(validate_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

print(f"训练集图像数量: {train_num}")
print(f"验证集图像数量: {val_num}")

# 定义模型
net = MobileNetV2(num_classes=len(cla_dict))   # 实例化模型
net.to(device)

# 尝试加载预训练权重
model_weight_path = "./CV/mobilenet_v2-b0353104.pth"
if os.path.exists(model_weight_path):
    print("加载预训练权重...")
    pre_weights = torch.load(model_weight_path, map_location=device)
    # 删除分类权重
    pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    print(f"缺失的键: {len(missing_keys)}, 意外的键: {len(unexpected_keys)}")
    
    # 冻结除最后全连接层以外的所有权重
    for param in net.features.parameters():
        param.requires_grad = False
    print("已冻结特征提取层")
else:
    print("未找到预训练权重，使用随机初始化")

loss_function = nn.CrossEntropyLoss()   # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 定义优化器

# 用于存储训练历史
train_losses = []
train_accuracies = []
val_accuracies = []
epochs_list = []

# 设置存储权重路径
save_path = './CV/mobilenetV2.pth'
best_acc = 0.0
num_epochs = 100

print(f"\n开始训练，共{num_epochs}轮...")

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
    
    # 训练阶段
    net.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for step, data in enumerate(train_loader, start=0):
        # 获取数据的图像和标签
        images, labels = data
        
        # 将历史损失梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels.to(device)).sum().item()
        
        # 打印训练进度
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(f"\r训练进度: {int(rate * 100):3d}%[{a}->{b}] Loss: {loss.item():.4f}", end="")
    
    # 计算训练准确率
    train_acc = train_correct / train_total
    avg_train_loss = running_loss / len(train_loader)
    
    # 验证阶段
    net.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            _, predicted = torch.max(outputs, 1)
            val_total += test_labels.size(0)
            val_correct += (predicted == test_labels.to(device)).sum().item()
    
    val_acc = val_correct / val_total
    
    # 保存训练历史
    epochs_list.append(epoch + 1)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), save_path)
        print(f"\n✓ 新的最佳模型已保存! 验证准确率: {val_acc:.4f}")
    
    # 打印epoch结果
    print(f"\nEpoch {epoch + 1:3d}: 训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
    
    # # 每10个epoch绘制一次图表
    # if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
    #     plot_training_history(epochs_list, train_losses, train_accuracies, val_accuracies, epoch + 1)

print(f"\n训练完成! 最佳验证准确率: {best_acc:.4f}")

def plot_training_history(epochs, train_losses, train_accs, val_accs, current_epoch):
    """绘制训练历史图表"""
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'训练历史 (Epoch {current_epoch})', fontsize=16, fontweight='bold')
    
    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失', marker='o', markersize=4)
    ax1.set_title('训练损失变化', fontsize=14, fontweight='bold')
    ax1.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 设置损失图的样式
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'g-', linewidth=2, label='训练准确率', marker='s', markersize=4)
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='验证准确率', marker='^', markersize=4)
    ax2.set_title('准确率变化', fontsize=14, fontweight='bold')
    ax2.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率 (Accuracy)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1)
    
    # 设置准确率图的样式
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 添加最新数值标注
    if len(epochs) > 0:
        latest_epoch = epochs[-1]
        latest_loss = train_losses[-1]
        latest_train_acc = train_accs[-1]
        latest_val_acc = val_accs[-1]
        
        # 在损失图上标注最新值
        ax1.annotate(f'{latest_loss:.4f}', 
                    xy=(latest_epoch, latest_loss), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 在准确率图上标注最新值
        ax2.annotate(f'训练: {latest_train_acc:.4f}', 
                    xy=(latest_epoch, latest_train_acc), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax2.annotate(f'验证: {latest_val_acc:.4f}', 
                    xy=(latest_epoch, latest_val_acc), 
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # 保存图表
    plot_filename = f'./CV/training_history_epoch_{current_epoch:03d}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 训练图表已保存: {plot_filename}")
    
    # 显示图表
    plt.show()
    plt.close()

def save_training_log(epochs, train_losses, train_accs, val_accs):
    """保存训练日志到文件"""
    log_filename = './CV/training_log.txt'
    
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write("=== MobileNetV2 动物分类训练日志 ===\n\n")
        f.write(f"训练轮次: {len(epochs)}\n")
        f.write(f"最佳验证准确率: {max(val_accs):.4f}\n")
        f.write(f"最终训练损失: {train_losses[-1]:.4f}\n")
        f.write(f"最终训练准确率: {train_accs[-1]:.4f}\n")
        f.write(f"最终验证准确率: {val_accs[-1]:.4f}\n\n")
        
        f.write("详细训练记录:\n")
        f.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Acc\n")
        f.write("-" * 50 + "\n")
        
        for i, epoch in enumerate(epochs):
            f.write(f"{epoch:3d}\t{train_losses[i]:.6f}\t{train_accs[i]:.6f}\t{val_accs[i]:.6f}\n")
    
    print(f"📝 训练日志已保存: {log_filename}")

# 最终绘制完整的训练历史
plot_training_history(epochs_list, train_losses, train_accuracies, val_accuracies, num_epochs)

# 保存训练日志
save_training_log(epochs_list, train_losses, train_accuracies, val_accuracies)

print("🎉 所有训练任务完成!")