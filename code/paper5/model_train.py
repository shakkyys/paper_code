import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import os

# # Step 1: Define the data preprocessing
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# 加载CIFAR10数据集

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化网络
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)  #CIFR10分类任务为10 
model = model.to(device)

# 定义超参数
batch_size = 10
num_epochs = 10
learning_rate = 0.001
# 对抗样本数
adver_nums = 1000

# 定义优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

history = {'train_loss': [], 'train_accuracy': [], 'test_accuracy': []}

# 训练模型
def train(model, optimizer):
    history = {'train_loss': [], 'train_accuracy': [], 'test_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total

        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {epoch_accuracy:.2f}%")

        # 在每个epoch结束后进行测试并保存测试准确率
        test_accuracy = test(model)
        history['test_accuracy'].append(test_accuracy)

    # 绘制并保存训练和测试的曲线
    plot_and_save_curves(history)

    return history

def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy

def plot_and_save_curves(history):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='Training Loss')
    plt.plot(range(1, num_epochs + 1), history['train_accuracy'], label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), history['test_accuracy'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    if not os.path.exists("train_images"):
        os.makedirs("train_images")
    plt.savefig('train_images/train_curves.png')


# 调用函数进行训练和保存曲线
history = train(model, optimizer)

# Step 7: Save the model
torch.save(model.state_dict(), "resnet18_cifar10.pth")