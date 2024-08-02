import os.path

import dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import model
import MINSTDataSet


if torch.cuda.is_available():
    print(True)
else:
    print(False)


# 转换数据集
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])



# 加载MNIST数据集
'''
train_dataset = datasets.MNIST(root='./MNIST',train=True,transform=transform,download=True)
test_dataset = datasets.MNIST(root='./MNIST',train=False,transform=transform,download=True)
'''
train_dataset = MINSTDataSet.DealDataSet(root='./', train=True, transform=transform)
test_dataset = MINSTDataSet.DealDataSet(root='./', train=False, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


model_file_path = 'model_params.pth'
model = model.SimpleMLP()
# 初始化网络和优化器
if os.path.exists(model_file_path):
    model.load_state_dict(torch.load('model_params.pth'))


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

# 测试模型
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练和测试模型
for epoch in range(1, 50):  # 训练50个epoch
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)

torch.save(model.state_dict(), 'model_params.pth')
